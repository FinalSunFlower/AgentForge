import asyncio
import os
from uuid import UUID, uuid4

import pytest
from redis.asyncio import Redis
from sqlalchemy import select

from services.core_api.app.config import get_settings
from services.core_api.app.models import OutboxEvent
from services.core_api.app.outbox import claim_pending_events, enqueue, publish_claimed_events
from tests.pg_gate import make_session_factory, postgres_url, redis_url


@pytest.mark.asyncio
async def test_postgres_relays_claim_disjoint_rows() -> None:
    """Two open transactions must overlap before either commits.

    asyncio.gather alone is not enough: if the first relay commits before the
    second SELECT, the second sees published rows and SKIP LOCKED is never hit.
    The barrier holds both claims until both SELECTs have run.
    """
    with postgres_url() as database_url, redis_url() as redis_target:
        previous = os.environ.get("REDIS_URL")
        os.environ["REDIS_URL"] = redis_target
        get_settings.cache_clear()
        engine, factory = await make_session_factory(database_url)
        try:
            aggregate_id = str(uuid4())
            event_count = 12
            async with factory() as session:
                for index in range(event_count):
                    await enqueue(
                        session,
                        aggregate_type="integration",
                        aggregate_id=aggregate_id,
                        event_type=f"integration.{index}",
                        payload={"index": index},
                    )
                await session.commit()

            barrier = asyncio.Barrier(2)
            claims: list[list[UUID]] = []

            async def claim_while_peer_holds() -> int:
                async with factory() as session:
                    events = await claim_pending_events(session, batch_size=event_count)
                    assert session.bind is not None
                    assert session.bind.dialect.name == "postgresql"
                    assert all(event.status == "pending" for event in events)
                    claims.append([event.id for event in events])
                    await barrier.wait()
                    return await publish_claimed_events(session, events)

            counts = await asyncio.gather(claim_while_peer_holds(), claim_while_peer_holds())
            first, second = claims
            assert len(claims) == 2
            assert set(first).isdisjoint(set(second))
            assert len(first) + len(second) == event_count
            assert sum(counts) == event_count
            async with factory() as session:
                rows = list(
                    await session.scalars(
                        select(OutboxEvent).where(OutboxEvent.aggregate_id == aggregate_id)
                    )
                )
                assert len(rows) == event_count
                assert all(row.status == "published" for row in rows)

            client = Redis.from_url(redis_target, decode_responses=True)
            try:
                stream_entries = await client.xrange("agentforge:integration")
                ids = [
                    fields["event_id"]
                    for _, fields in stream_entries
                    if fields.get("aggregate_id") == aggregate_id
                ]
                assert len(ids) == event_count
                assert len(set(ids)) == event_count
            finally:
                await client.aclose()
        finally:
            await engine.dispose()
            if previous is None:
                os.environ.pop("REDIS_URL", None)
            else:
                os.environ["REDIS_URL"] = previous
            get_settings.cache_clear()
