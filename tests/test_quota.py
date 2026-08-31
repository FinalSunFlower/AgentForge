from uuid import uuid4

import pytest

from services.core_api.app.db import SessionFactory, init_db
from services.core_api.app.models import Agent, Run, RunStatus, Thread, User
from services.core_api.app.quota import QuotaExceeded, enforce_run_quota


@pytest.mark.asyncio
async def test_free_plan_blocks_third_active_run() -> None:
    await init_db()
    async with SessionFactory() as session:
        user = User(email=f"{uuid4()}@example.com", display_name="Quota")
        agent = Agent(
            slug=f"quota-{uuid4()}", version="1.0.0", model_ref="test", system_policy="test"
        )
        session.add_all([user, agent])
        await session.flush()
        thread = Thread(user_id=user.id, agent_id=agent.id, title="Quota")
        session.add(thread)
        await session.flush()
        session.add_all(
            [
                Run(thread_id=thread.id, status=RunStatus.CREATED),
                Run(thread_id=thread.id, status=RunStatus.RUNNING),
            ]
        )
        await session.flush()
        with pytest.raises(QuotaExceeded, match="concurrent_run_limit_exceeded"):
            await enforce_run_quota(session, user, thread.id, "hello")
        await session.rollback()
