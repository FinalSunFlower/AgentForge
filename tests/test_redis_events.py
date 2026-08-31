from uuid import uuid4

import fakeredis.aioredis
import pytest

from packages.contracts.events import EventEnvelope, EventType
from services.core_api.app.redis_events import RedisRunEventStore


@pytest.mark.asyncio
async def test_redis_run_event_store_replays_after_sequence() -> None:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = RedisRunEventStore(client)
    run_id, thread_id = uuid4(), uuid4()
    for sequence in (1, 2, 3):
        await store.publish(
            EventEnvelope(
                run_id=run_id,
                thread_id=thread_id,
                sequence=sequence,
                type=EventType.MESSAGE_DELTA,
                payload={"delta": str(sequence)},
            )
        )

    events = [event async for event in store.replay(run_id, after_sequence=1)]
    assert [event.sequence for event in events] == [2, 3]
    assert events[-1].payload["delta"] == "3"
    await client.aclose()
