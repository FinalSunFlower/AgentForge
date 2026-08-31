from services.agent_runtime.app.eval_harness import (
    MEMORY_EVAL_CLAIM,
    memory_eval_report,
    needle_survives_compression,
)
from services.agent_runtime.app.memory import assemble_context, build_needle_thread, extract_facts


def test_extract_facts_keeps_codes_and_drops_chitchat() -> None:
    facts = extract_facts(
        [
            {"role": "user", "content": "hello there"},
            {"role": "user", "content": "Remember that the vault code is ORCHID-7729."},
            {"role": "assistant", "content": "ok"},
        ]
    )
    assert any("ORCHID-7729" in fact.text for fact in facts)
    assert all("hello there" not in fact.text for fact in facts)


def test_long_thread_compression_keeps_early_needle() -> None:
    assert needle_survives_compression() is True
    history = build_needle_thread(
        "Remember that the vault code is ORCHID-7729.", filler_turns=80, needle_at=3
    )
    assembled = assemble_context(
        history,
        system_policy="policy",
        max_chars=6_000,
        keep_recent=4,
        compress_after=8,
    )
    assert assembled.compressed is True
    assert assembled.report["dropped_turns"] > 50
    assert "ORCHID-7729" in "\n".join(item["content"] for item in assembled.messages)
    memory = memory_eval_report()
    assert memory["claim"] == MEMORY_EVAL_CLAIM
    assert all(memory["survived"].values())


def test_implicit_preference_is_outside_extractive_scope() -> None:
    history = build_needle_thread("I don't like meetings on Fridays.", filler_turns=80, needle_at=3)
    assembled = assemble_context(
        history,
        system_policy="policy",
        max_chars=6_000,
        keep_recent=4,
        compress_after=8,
    )
    blob = "\n".join(item["content"] for item in assembled.messages)
    assert assembled.compressed is True
    assert "Fridays" not in blob
