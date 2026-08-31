from datetime import UTC, datetime
from uuid import uuid4

import fakeredis.aioredis
import pytest

import services.core_api.app.outbox as outbox_module
from services.core_api.app.config import get_settings
from services.core_api.app.db import SessionFactory, init_db
from services.core_api.app.outbox import RedisNotConfigured, enqueue, relay_once


@pytest.mark.asyncio
async def test_outbox_enqueue_is_durable_without_redis() -> None:
    await init_db()
    async with SessionFactory() as session:
        event = await enqueue(
            session,
            aggregate_type="test",
            aggregate_id=str(uuid4()),
            event_type="test.created",
            payload={"value": 1},
        )
        await session.commit()
        assert event.status == "pending"
        with pytest.raises(RedisNotConfigured, match="REDIS_URL"):
            await relay_once(session)


@pytest.mark.asyncio
async def test_sqlite_outbox_relay_publishes_and_marks_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(outbox_module.Redis, "from_url", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(get_settings(), "redis_url", "redis://fake")
    aggregate_id = str(uuid4())
    async with SessionFactory() as session:
        event = await enqueue(
            session,
            aggregate_type="local",
            aggregate_id=aggregate_id,
            event_type="local.created",
            payload={"value": 2},
        )
        await session.commit()
        for _ in range(20):
            if await relay_once(session, batch_size=200) == 0:
                break
            await session.refresh(event)
            if event.status == "published":
                break
        await session.refresh(event)
        assert event.status == "published"
    rows = await fake.xrange("agentforge:local")
    assert len(rows) >= 1
    assert any(fields["aggregate_id"] == aggregate_id for _, fields in rows)
    await fake.aclose()


@pytest.mark.asyncio
async def test_outbox_publish_failure_persists_retry_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenRedis:
        async def xadd(self, *_args, **_kwargs):
            raise ConnectionError("redis unavailable")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(outbox_module.Redis, "from_url", lambda *_args, **_kwargs: BrokenRedis())
    monkeypatch.setattr(get_settings(), "redis_url", "redis://broken")
    aggregate_id = str(uuid4())
    async with SessionFactory() as session:
        event = await enqueue(
            session,
            aggregate_type="retry",
            aggregate_id=aggregate_id,
            event_type="retry.created",
            payload={},
        )
        await session.commit()
        before = datetime.now(UTC)
        with pytest.raises(ConnectionError, match="redis unavailable"):
            await relay_once(session)
        await session.refresh(event)
        assert event.status == "pending"
        assert event.attempts >= 1
        available_at = (
            event.available_at
            if event.available_at.tzinfo
            else event.available_at.replace(tzinfo=UTC)
        )
        assert available_at >= before
