import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import select

from packages.contracts.events import EventType
from services.agent_runtime.app.executor import PROMPT_VERSION, RunExecutor
from services.agent_runtime.app.policy import RunBudget
from services.agent_runtime.app.provider import ModelTurn, ToolCallDelta
from services.core_api.app.db import SessionFactory, init_db
from services.core_api.app.models import (
    Message,
    MessageRole,
    Run,
    RunEvent,
    RunStatus,
    Thread,
    User,
)
from services.core_api.app.seed import ensure_builtin_tools, ensure_default_agent


async def _create_run(prompt: str = "2 + 3") -> tuple:
    await init_db()
    async with SessionFactory() as session:
        user = User(email=f"{uuid4()}@example.com", display_name="Gate test")
        agent = await ensure_default_agent(session)
        await ensure_builtin_tools(session)
        session.add(user)
        await session.flush()
        thread = Thread(user_id=user.id, agent_id=agent.id, title="Gate")
        session.add(thread)
        await session.flush()
        session.add(Message(thread_id=thread.id, role=MessageRole.USER, content=prompt))
        run = Run(thread_id=thread.id)
        session.add(run)
        await session.commit()
        return run.id, thread.id


class TwoStepProvider:
    def __init__(self, first: ModelTurn, second: ModelTurn | None = None) -> None:
        self.turns = [first]
        if second is not None:
            self.turns.append(second)

    async def complete(self, messages, tools):
        if not self.turns:
            return ModelTurn(text="done")
        return self.turns.pop(0)


@pytest.mark.asyncio
async def test_duplicate_tool_call_id_is_replayed_not_reexecuted() -> None:
    run_id, _ = await _create_run()
    provider = TwoStepProvider(
        ModelTurn(
            tool_calls=[
                ToolCallDelta("tc-dup", "calculator", '{"expression":"2 + 3"}'),
                ToolCallDelta("tc-dup", "calculator", '{"expression":"2 + 3"}'),
            ]
        ),
        ModelTurn(text="The result is 5.", input_tokens=4, output_tokens=3),
    )
    await RunExecutor(lambda _thread: provider).execute(run_id)
    async with SessionFactory() as session:
        events = list(await session.scalars(select(RunEvent).where(RunEvent.run_id == run_id)))
        statuses = [
            (event.payload_json.get("payload") or {}).get("status")
            for event in events
            if event.event_type == EventType.TOOL_RESULT.value
        ]
        assert "succeeded" in statuses
        assert "duplicate" in statuses


@pytest.mark.asyncio
async def test_run_timeout_persists_budget_exceeded() -> None:
    run_id, _ = await _create_run()

    class SlowProvider:
        async def complete(self, messages, tools):
            await asyncio.sleep(2)
            return ModelTurn(text="late")

    executor = RunExecutor(lambda _thread: SlowProvider())
    executor.budget = RunBudget(max_steps=8, max_tool_depth=6, timeout_seconds=1)
    await executor.execute(run_id)
    async with SessionFactory() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        assert run.status == RunStatus.BUDGET_EXCEEDED
        assert run.terminal_reason == "run_timeout"


@pytest.mark.asyncio
async def test_cancel_between_steps_is_honored() -> None:
    run_id, _ = await _create_run()

    class CancelOnSecond:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(0.15)
                return ModelTurn(
                    tool_calls=[ToolCallDelta("tc-1", "calculator", '{"expression":"2 + 3"}')]
                )
            return ModelTurn(text="should not matter")

    async def cancel_after_start() -> None:
        for _ in range(40):
            await asyncio.sleep(0.02)
            async with SessionFactory() as session:
                events = list(
                    await session.scalars(select(RunEvent).where(RunEvent.run_id == run_id))
                )
                if any(event.event_type == EventType.RUN_STARTED.value for event in events):
                    run = await session.get(Run, run_id)
                    assert run is not None
                    run.status = RunStatus.CANCELED
                    run.terminal_reason = "user_canceled"
                    await session.commit()
                    return

    await asyncio.gather(
        RunExecutor(lambda _thread: CancelOnSecond()).execute(run_id), cancel_after_start()
    )
    async with SessionFactory() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        assert run.status == RunStatus.CANCELED


@pytest.mark.asyncio
async def test_approved_readonly_sql_executes() -> None:
    run_id, _ = await _create_run("SELECT title FROM novels")
    provider = TwoStepProvider(
        ModelTurn(
            tool_calls=[
                ToolCallDelta(
                    "tc-sql", "readonly_sql", '{"query":"SELECT title FROM novels LIMIT 5"}'
                )
            ]
        ),
        ModelTurn(text="Listed titles.", input_tokens=6, output_tokens=3),
    )
    await RunExecutor(lambda _thread: provider).execute(run_id)
    async with SessionFactory() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETED
        assert run.prompt_version == PROMPT_VERSION
        events = list(await session.scalars(select(RunEvent).where(RunEvent.run_id == run_id)))
        sql_results = [
            event.payload_json.get("payload") or {}
            for event in events
            if event.event_type == EventType.TOOL_RESULT.value
            and (event.payload_json.get("payload") or {}).get("tool_name") == "readonly_sql"
        ]
        assert sql_results
        assert sql_results[0].get("status") == "succeeded"


@pytest.mark.asyncio
async def test_stream_provider_emits_message_deltas() -> None:
    run_id, _ = await _create_run("hello")

    class StreamProvider:
        async def complete_stream(self, messages, tools):
            yield ModelTurn(text="Hello ")
            yield ModelTurn(text="world.", input_tokens=3, output_tokens=2)

    await RunExecutor(lambda _thread: StreamProvider()).execute(run_id)
    async with SessionFactory() as session:
        events = list(
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.sequence)
            )
        )
        deltas = [
            (event.payload_json.get("payload") or {}).get("delta")
            for event in events
            if event.event_type == EventType.MESSAGE_DELTA.value
        ]
        assert deltas == ["Hello ", "world."]


def test_usage_cost_uses_ceil_micro_formula() -> None:
    from services.agent_runtime.app.config import get_settings

    settings = get_settings()
    original_in = settings.model_input_cost_per_1m_micros
    original_out = settings.model_output_cost_per_1m_micros
    settings.model_input_cost_per_1m_micros = 1_000_000
    settings.model_output_cost_per_1m_micros = 2_000_000
    try:
        usage = RunExecutor._usage(3, 2, 0.0, 0.0)
        assert usage["cost_micros"] == 7
    finally:
        settings.model_input_cost_per_1m_micros = original_in
        settings.model_output_cost_per_1m_micros = original_out
