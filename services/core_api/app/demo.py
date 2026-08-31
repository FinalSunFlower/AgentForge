from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .models import UsageDaily, UsageSession

DEMO_DAILY_BUDGET_REASON = "demo_daily_budget"
DEMO_DAILY_BUDGET_MESSAGE = (
    "Today's public demo token budget is exhausted. Run the stack locally "
    "(see Quick start) or come back tomorrow."
)
DEMO_IP_RATE_MESSAGE = (
    "This IP has reached the public demo run limit. Try again later or run locally."
)

_WINDOW_SECONDS = 3_600
_local_hits: dict[str, deque[float]] = {}
_local_lock = asyncio.Lock()


@dataclass(frozen=True)
class GlobalUsage:
    total_tokens: int
    runs: int


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


def reset_demo_rate_limiter() -> None:
    _local_hits.clear()


async def get_global_usage_today(session: AsyncSession) -> GlobalUsage:
    """Sum UsageDaily across all users, plus sessions the worker has not rolled up yet."""
    today = datetime.now(UTC).date().isoformat()
    day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    daily_tokens = await session.scalar(
        select(
            func.coalesce(func.sum(UsageDaily.input_tokens + UsageDaily.output_tokens), 0)
        ).where(UsageDaily.day == today)
    )
    daily_runs = await session.scalar(
        select(func.coalesce(func.sum(UsageDaily.runs), 0)).where(UsageDaily.day == today)
    )
    pending_tokens = await session.scalar(
        select(
            func.coalesce(func.sum(UsageSession.input_tokens + UsageSession.output_tokens), 0)
        ).where(
            UsageSession.aggregated_at.is_(None),
            UsageSession.created_at >= day_start,
        )
    )
    pending_runs = await session.scalar(
        select(func.count(UsageSession.id)).where(
            UsageSession.aggregated_at.is_(None),
            UsageSession.created_at >= day_start,
        )
    )
    return GlobalUsage(
        total_tokens=int(daily_tokens or 0) + int(pending_tokens or 0),
        runs=int(daily_runs or 0) + int(pending_runs or 0),
    )


async def check_demo_budget(session: AsyncSession) -> bool:
    settings = get_settings()
    if not settings.demo_mode:
        return True
    usage = await get_global_usage_today(session)
    return usage.total_tokens < settings.daily_token_budget


async def allow_ip(
    ip: str,
    *,
    limit: int,
    window_seconds: int = _WINDOW_SECONDS,
    redis_url: str | None = None,
) -> bool:
    now = time.time()
    cutoff = now - window_seconds
    if redis_url:
        from redis.asyncio import Redis

        client = Redis.from_url(redis_url, decode_responses=True)
        key = f"demo:ip:{ip}"
        member = f"{now:.6f}:{time.monotonic_ns()}"
        try:
            pipe = client.pipeline()
            pipe.zremrangebyscore(key, 0, cutoff)
            pipe.zcard(key)
            count = int((await pipe.execute())[1])
            if count >= limit:
                return False
            pipe = client.pipeline()
            pipe.zadd(key, {member: now})
            pipe.expire(key, window_seconds + 1)
            await pipe.execute()
            return True
        finally:
            await client.aclose()
    async with _local_lock:
        hits = _local_hits.setdefault(ip, deque())
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= limit:
            return False
        hits.append(now)
        return True


async def enforce_demo_ip_limit(ip: str) -> bool:
    settings = get_settings()
    if not settings.demo_mode:
        return True
    return await allow_ip(
        ip,
        limit=settings.demo_runs_per_ip_per_hour,
        window_seconds=_WINDOW_SECONDS,
        redis_url=settings.redis_url,
    )
