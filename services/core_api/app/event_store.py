from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.contracts.events import EventEnvelope, EventType

from .config import get_settings
from .models import Run, RunEvent
from .redis_events import publish_if_configured

# SQLite has no row-level locking. This process-wide lock prevents two local
# writers from selecting the same next sequence; PostgreSQL uses the database
# lock below for cross-process correctness.
_append_lock = asyncio.Lock()


async def append_event(
    session: AsyncSession,
    *,
    run_id: UUID,
    thread_id: UUID,
    event_type: EventType,
    payload: dict,
) -> EventEnvelope:
    async with _append_lock:
        dialect = session.bind.dialect.name if session.bind is not None else ""
        if dialect == "postgresql":
            await session.execute(select(Run.id).where(Run.id == run_id).with_for_update())
        current = await session.scalar(
            select(func.max(RunEvent.sequence)).where(RunEvent.run_id == run_id)
        )
        event = EventEnvelope(
            run_id=run_id,
            thread_id=thread_id,
            sequence=(current or 0) + 1,
            type=event_type,
            payload=payload,
        )
        session.add(
            RunEvent(
                run_id=run_id,
                sequence=event.sequence,
                event_id=str(event.event_id),
                event_type=event.type.value,
                payload_json=event.model_dump(mode="json"),
            )
        )
        await session.flush()
        from .outbox import enqueue

        await enqueue(
            session,
            aggregate_type="run",
            aggregate_id=str(run_id),
            event_type=event.type.value,
            payload=event.model_dump(mode="json"),
        )
        # Redis is an acceleration layer. A publish failure must not discard SQL facts.
        try:
            settings = get_settings()
            await publish_if_configured(
                event,
                settings.redis_url,
                max_length=settings.event_stream_maxlen,
                retention_seconds=settings.event_stream_retention_seconds,
            )
        except Exception:
            import logging

            logging.getLogger("agentforge.events").warning(
                "redis event mirror unavailable", exc_info=True
            )
        return event


async def iter_events(
    session: AsyncSession, run_id: UUID, after_sequence: int = 0
) -> AsyncIterator[EventEnvelope]:
    result = await session.scalars(
        select(RunEvent)
        .where(RunEvent.run_id == run_id, RunEvent.sequence > after_sequence)
        .order_by(RunEvent.sequence)
    )
    for row in result:
        yield EventEnvelope.model_validate(row.payload_json)
