from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select, text

from services.core_api.app.models import Run, Thread, UsageDaily, UsageSession


async def aggregate_usage_once(session, *, batch_size: int = 500) -> int:
    bind = session.bind
    dialect = bind.dialect.name if bind is not None else ""
    if dialect == "sqlite":
        await session.execute(text("BEGIN IMMEDIATE"))
    query = (
        select(UsageSession)
        .where(UsageSession.aggregated_at.is_(None))
        .order_by(UsageSession.created_at)
        .limit(batch_size)
    )
    if dialect == "postgresql":
        query = query.with_for_update(skip_locked=True)
    rows = list(await session.scalars(query))
    count = 0
    for usage in rows:
        owner = await session.scalar(
            select(Thread.user_id)
            .join(Run, Run.thread_id == Thread.id)
            .where(Run.id == usage.run_id)
        )
        if owner is None:
            usage.aggregated_at = datetime.now(UTC)
            continue
        day = usage.created_at.astimezone(UTC).date().isoformat()
        aggregate = await session.scalar(
            select(UsageDaily).where(UsageDaily.user_id == owner, UsageDaily.day == day)
        )
        if aggregate is None:
            aggregate = UsageDaily(
                user_id=owner, day=day, runs=0, input_tokens=0, output_tokens=0, cost_micros=0
            )
            session.add(aggregate)
        aggregate.runs += 1
        aggregate.input_tokens += usage.input_tokens
        aggregate.output_tokens += usage.output_tokens
        aggregate.cost_micros += usage.cost_micros
        usage.aggregated_at = datetime.now(UTC)
        count += 1
    await session.commit()
    return count
