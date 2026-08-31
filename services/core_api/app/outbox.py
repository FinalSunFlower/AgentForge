from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .models import OutboxEvent


class RedisNotConfigured(RuntimeError):
    pass


async def enqueue(
    session: AsyncSession,
    *,
    aggregate_type: str,
    aggregate_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> OutboxEvent:
    event = OutboxEvent(
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        event_type=event_type,
        payload_json=payload,
    )
    session.add(event)
    await session.flush()
    return event


async def claim_pending_events(
    session: AsyncSession, *, batch_size: int = 100
) -> list[OutboxEvent]:
    """Lock a pending batch. PostgreSQL skips rows already locked by a peer."""
    query = (
        select(OutboxEvent)
        .where(OutboxEvent.status == "pending", OutboxEvent.available_at <= datetime.now(UTC))
        .order_by(OutboxEvent.created_at)
        .limit(batch_size)
    )
    dialect = session.bind.dialect.name if session.bind is not None else ""
    if dialect == "postgresql":
        query = query.with_for_update(skip_locked=True)
    elif dialect == "sqlite":
        await session.execute(text("BEGIN IMMEDIATE"))
    return list(await session.scalars(query))


async def publish_claimed_events(session: AsyncSession, events: list[OutboxEvent]) -> int:
    redis_url = get_settings().redis_url
    if not redis_url:
        raise RedisNotConfigured("REDIS_URL is required for the outbox relay")
    if not events:
        return 0

    client = Redis.from_url(redis_url, decode_responses=True)
    event_attempts = [(event.id, event.attempts) for event in events]
    try:
        for event in events:
            await client.xadd(
                f"agentforge:{event.aggregate_type}",
                {
                    "event_id": str(event.id),
                    "event_type": event.event_type,
                    "aggregate_id": event.aggregate_id,
                    "payload": json.dumps(event.payload_json, ensure_ascii=False),
                },
            )
            event.status = "published"
            event.published_at = datetime.now(UTC)
            event.attempts += 1
        await session.commit()
    except Exception:
        await session.rollback()
        now = datetime.now(UTC)
        for event_id, attempts in event_attempts:
            next_attempt = attempts + 1
            await session.execute(
                update(OutboxEvent)
                .where(OutboxEvent.id == event_id)
                .values(
                    status="dead_letter" if next_attempt >= 8 else "pending",
                    attempts=OutboxEvent.attempts + 1,
                    available_at=now + timedelta(seconds=min(300, 2 ** min(next_attempt, 8))),
                )
            )
        await session.commit()
        raise
    finally:
        await client.aclose()
    return len(events)


async def relay_once(session: AsyncSession, *, batch_size: int = 100) -> int:
    events = await claim_pending_events(session, batch_size=batch_size)
    return await publish_claimed_events(session, events)
