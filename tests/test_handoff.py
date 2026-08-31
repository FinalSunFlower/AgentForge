from uuid import uuid4

import pytest
from sqlalchemy import select

from packages.contracts.events import EventType
from services.agent_runtime.app.eval_harness import KEYWORD_ROUTER_CLAIM, run_supervisor_eval
from services.agent_runtime.app.executor import RunExecutor
from services.agent_runtime.app.handoff import SPECIALISTS
from services.agent_runtime.app.provider import ModelTurn, ToolCallDelta
from services.core_api.app.db import SessionFactory, init_db
from services.core_api.app.models import (
    Agent,
    Message,
    MessageRole,
    Run,
    RunEvent,
    RunStatus,
    Thread,
    User,
)
from services.core_api.app.seed import (
    ensure_builtin_tools,
    ensure_default_agent,
    ensure_handoff_agents,
)


def test_supervisor_eval_routes_to_specialists() -> None:
    traces, summary = run_supervisor_eval()
    assert summary["eval_kind"] == "deterministic_keyword_router"
    assert summary["claim"] == KEYWORD_ROUTER_CLAIM
    assert summary["task_success_rate"] == 1.0
    assert {item.predicted_args[0]["specialist"] for item in traces} == {"code_data", "retrieval"}


@pytest.mark.asyncio
async def test_executor_handoff_switches_tools_and_policy() -> None:
    await init_db()
    async with SessionFactory() as session:
        user = User(email=f"{uuid4()}@example.com", display_name="Handoff")
        await ensure_default_agent(session)
        await ensure_handoff_agents(session)
        await ensure_builtin_tools(session)
        session.add(user)
        await session.flush()
        supervisor = await session.scalar(select(Agent).where(Agent.slug == "supervisor"))
        assert supervisor is not None
        thread = Thread(user_id=user.id, agent_id=supervisor.id, title="Handoff")
        session.add(thread)
        await session.flush()
        session.add(Message(thread_id=thread.id, role=MessageRole.USER, content="Calculate 2 + 3"))
        run = Run(thread_id=thread.id)
        session.add(run)
        await session.commit()
        run_id = run.id
        thread_id = thread.id

    class HandoffProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tools):
            self.calls += 1
            names = {item["function"]["name"] for item in tools}
            if self.calls == 1:
                assert "handoff" in names
                assert "calculator" not in names
                return ModelTurn(
                    tool_calls=[
                        ToolCallDelta(
                            "tc-h",
                            "handoff",
                            '{"specialist":"code_data","reason":"math","brief":"Calculate 2 + 3"}',
                        )
                    ]
                )
            if self.calls == 2:
                assert "calculator" in names
                assert "retrieval" not in names
                return ModelTurn(
                    tool_calls=[ToolCallDelta("tc-c", "calculator", '{"expression":"2 + 3"}')]
                )
            return ModelTurn(text="The result is 5.", input_tokens=4, output_tokens=3)

    await RunExecutor(lambda _thread: HandoffProvider()).execute(run_id)
    async with SessionFactory() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETED
        events = list(await session.scalars(select(RunEvent).where(RunEvent.run_id == run_id)))
        assert any(event.event_type == EventType.AGENT_HANDOFF.value for event in events)
        thread = await session.get(Thread, thread_id)
        specialist = await session.scalar(select(Agent).where(Agent.slug == "code-data-specialist"))
        assert thread is not None and specialist is not None
        assert thread.agent_id == specialist.id
        assert SPECIALISTS["code_data"].tools

        session.add(Message(thread_id=thread.id, role=MessageRole.USER, content="Calculate 4 + 1"))
        follow = Run(thread_id=thread.id)
        session.add(follow)
        await session.commit()
        follow_id = follow.id

    class SpecialistProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tools):
            self.calls += 1
            names = {item["function"]["name"] for item in tools}
            assert "calculator" in names
            assert "handoff" not in names
            assert "retrieval" not in names
            if self.calls == 1:
                return ModelTurn(
                    tool_calls=[ToolCallDelta("tc-f", "calculator", '{"expression":"4 + 1"}')]
                )
            return ModelTurn(text="The result is 5.", input_tokens=3, output_tokens=2)

    await RunExecutor(lambda _thread: SpecialistProvider()).execute(follow_id)
    async with SessionFactory() as session:
        follow = await session.get(Run, follow_id)
        assert follow is not None
        assert follow.status == RunStatus.COMPLETED
