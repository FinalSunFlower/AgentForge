from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import select

from services.core_api.app.db import SessionFactory, init_db
from services.core_api.app.models import (
    Agent,
    Run,
    Thread,
    UsageDaily,
    UsageSession,
    User,
)
from services.worker.app.usage import aggregate_usage_once


@pytest.mark.asyncio
async def test_usage_worker_aggregates_each_session_once() -> None:
    await init_db()
    async with SessionFactory() as session:
        for previous in await session.scalars(
            select(UsageSession).where(UsageSession.aggregated_at.is_(None))
        ):
            previous.aggregated_at = datetime.now(UTC)
        await session.commit()
        user = User(email=f"{uuid4()}@example.com", display_name="Usage")
        agent = Agent(
            slug=f"usage-{uuid4()}", version="1.0.0", model_ref="test", system_policy="test"
        )
        session.add_all([user, agent])
        await session.flush()
        thread = Thread(user_id=user.id, agent_id=agent.id)
        session.add(thread)
        await session.flush()
        run = Run(thread_id=thread.id)
        session.add(run)
        await session.flush()
        session.add(UsageSession(run_id=run.id, input_tokens=10, output_tokens=5, cost_micros=2))
        await session.commit()
        assert await aggregate_usage_once(session) == 1
        assert await aggregate_usage_once(session) == 0
        daily = await session.scalar(select(UsageDaily).where(UsageDaily.user_id == user.id))
        assert daily is not None and daily.runs == 1 and daily.output_tokens == 5
