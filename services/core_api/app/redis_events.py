from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

from redis.asyncio import Redis

from packages.contracts.events import EventEnvelope


class RedisRunEventStore:
    """Redis Streams acceleration layer; SQL remains the source of truth."""

    def __init__(
        self, client: Redis, *, max_length: int = 10_000, retention_seconds: int = 86_400
    ) -> None:
        self.client = client
        self.max_length = max_length
        self.retention_seconds = retention_seconds

    @staticmethod
    def stream_key(run_id: UUID) -> str:
        return f"agentforge:run:{run_id}:events"

    async def publish(self, event: EventEnvelope) -> str:
        stream = self.stream_key(event.run_id)
        entry_id = await self.client.xadd(
            stream,
            {"sequence": str(event.sequence), "event": event.model_dump_json()},
            maxlen=self.max_length,
            approximate=True,
        )
        if self.retention_seconds > 0:
            await self.client.expire(stream, self.retention_seconds)
        return entry_id

    async def replay(self, run_id: UUID, after_sequence: int = 0) -> AsyncIterator[EventEnvelope]:
        entries = await self.client.xrange(self.stream_key(run_id))
        for _, fields in entries:
            sequence = int(fields["sequence"])
            if sequence > after_sequence:
                yield EventEnvelope.model_validate_json(fields["event"])

    async def oldest_sequence(self, run_id: UUID) -> int | None:
        entries = await self.client.xrange(self.stream_key(run_id), count=1)
        if not entries:
            return None
        return int(entries[0][1]["sequence"])


async def publish_if_configured(
    event: EventEnvelope,
    redis_url: str | None,
    max_length: int = 10_000,
    retention_seconds: int = 86_400,
) -> bool:
    if not redis_url:
        return False
    client = Redis.from_url(redis_url, decode_responses=True)
    try:
        await RedisRunEventStore(
            client, max_length=max_length, retention_seconds=retention_seconds
        ).publish(event)
        return True
    finally:
        await client.aclose()
