import asyncio
import time

from services.agent_runtime.app.scheduler import ScheduledCall, schedule_batches


def test_independent_reads_share_one_batch() -> None:
    calls = [
        ScheduledCall("tc-1", "calculator", '{"expression":"1+1"}'),
        ScheduledCall("tc-2", "retrieval", '{"query":"planning"}'),
    ]
    batches = schedule_batches(calls)
    assert len(batches) == 1
    assert [item.tool_call_id for item in batches[0]] == ["tc-1", "tc-2"]


def test_write_and_dependent_calls_are_serialized() -> None:
    calls = [
        ScheduledCall("tc-1", "retrieval", '{"query":"planning"}'),
        ScheduledCall("tc-2", "calculator", '{"expression":"1+1","prior":"tc-1"}'),
        ScheduledCall("tc-3", "ledger_write", "{}", side_effect="write"),
        ScheduledCall("tc-4", "intent_router", '{"text":"hi"}'),
    ]
    batches = schedule_batches(calls)
    assert [item.tool_call_id for item in batches[0]] == ["tc-1"]
    assert [item.tool_call_id for item in batches[1]] == ["tc-2"]
    assert [item.tool_call_id for item in batches[2]] == ["tc-3"]
    assert [item.tool_call_id for item in batches[3]] == ["tc-4"]


async def test_schedule_batches_are_safe_to_gather() -> None:
    seen: list[str] = []

    async def work(name: str) -> str:
        await asyncio.sleep(0.05)
        seen.append(name)
        return name

    batches = schedule_batches(
        [
            ScheduledCall("a", "calculator", "{}"),
            ScheduledCall("b", "retrieval", "{}"),
        ]
    )
    started = time.perf_counter()
    for batch in batches:
        await asyncio.gather(*[work(item.tool_call_id) for item in batch])
    elapsed = time.perf_counter() - started
    assert elapsed < 0.09
    assert set(seen) == {"a", "b"}
