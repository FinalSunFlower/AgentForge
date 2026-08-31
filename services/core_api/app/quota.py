from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Run, RunStatus, Thread, User


@dataclass(frozen=True)
class PlanLimit:
    max_concurrent_runs: int
    max_input_chars: int


PLAN_LIMITS: dict[str, PlanLimit] = {
    "free": PlanLimit(max_concurrent_runs=2, max_input_chars=32_000),
    "plus": PlanLimit(max_concurrent_runs=5, max_input_chars=64_000),
    "pro": PlanLimit(max_concurrent_runs=20, max_input_chars=128_000),
}


class QuotaExceeded(Exception):
    def __init__(self, code: str, limit: int) -> None:
        self.code = code
        self.limit = limit
        super().__init__(code)


async def enforce_run_quota(
    session: AsyncSession, user: User, thread_id: UUID, content: str
) -> None:
    limits = PLAN_LIMITS.get(user.plan, PLAN_LIMITS["free"])
    if len(content) > limits.max_input_chars:
        raise QuotaExceeded("input_size_exceeded", limits.max_input_chars)
    active = await session.scalar(
        select(func.count(Run.id))
        .join(Run.thread)
        .where(
            Thread.user_id == user.id,
            Run.status.in_([RunStatus.CREATED, RunStatus.RUNNING]),
        )
    )
    if int(active or 0) >= limits.max_concurrent_runs:
        raise QuotaExceeded("concurrent_run_limit_exceeded", limits.max_concurrent_runs)
