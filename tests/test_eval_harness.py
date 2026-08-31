from services.agent_runtime.app.eval_harness import (
    EMBEDDING_ROUTER_CLAIM,
    EVAL_TASKS,
    KEYWORD_ROUTER_CLAIM,
    MEMORY_EVAL_CLAIM,
    memory_eval_report,
    needle_survives_compression,
    retrieval_mode_report,
    run_embedding_router_eval,
    run_embedding_supervisor_eval,
    run_keyword_router_eval,
    run_supervisor_eval,
    score_trace,
    zero_overlap_report,
)


def test_eval_harness_scores_keyword_router_on_synthetic_suite() -> None:
    assert len(EVAL_TASKS) >= 24
    traces, summary = run_keyword_router_eval()
    assert summary["eval_kind"] == "deterministic_keyword_router"
    assert summary["claim"] == KEYWORD_ROUTER_CLAIM
    assert summary["tasks"] == len(EVAL_TASKS)
    assert summary["tool_selection_accuracy"] >= 0.9
    assert summary["param_accuracy"] >= 0.9
    assert summary["task_success_rate"] >= 0.9
    assert summary["avg_steps"] >= 1
    failed = [item.task_id for item in traces if not item.success]
    assert failed == []


def test_score_trace_detects_wrong_tool_and_wrong_args() -> None:
    task = EVAL_TASKS[0]
    wrong_tool = score_trace(
        task, ["retrieval"], [{}], steps=2, cost_micros=0, terminal="completed"
    )
    assert wrong_tool.tool_selection_correct is False
    assert wrong_tool.success is False
    wrong_args = score_trace(
        task,
        ["calculator"],
        [{"expression": "1 + 1"}],
        steps=2,
        cost_micros=10,
        terminal="completed",
    )
    assert wrong_args.param_correct is False


def test_retrieval_benchmark_is_reproducible() -> None:
    first = retrieval_mode_report()
    second = retrieval_mode_report()
    assert first == second
    assert set(first) == {
        "keyword",
        "bm25",
        "vector",
        "hybrid",
        "hybrid_rerank",
        "hybrid_late_interaction",
        "hybrid_cross_encoder",
    }
    assert first["hybrid_cross_encoder"]["queries"] == 19


def test_supervisor_and_memory_evals_are_deterministic() -> None:
    _, summary = run_supervisor_eval()
    assert summary["eval_kind"] == "deterministic_keyword_router"
    assert summary["claim"] == KEYWORD_ROUTER_CLAIM
    assert summary["task_success_rate"] == 1.0
    assert needle_survives_compression() is True
    memory = memory_eval_report()
    assert memory["claim"] == MEMORY_EVAL_CLAIM
    assert all(memory["survived"].values())
    gap = zero_overlap_report()
    assert gap["vector"]["recall_at_k"] > gap["keyword"]["recall_at_k"]


def test_embedding_router_is_measured_separately_from_keyword_baseline() -> None:
    _, keyword = run_keyword_router_eval()
    traces, embedding = run_embedding_router_eval()
    assert embedding["eval_kind"] == "minilm_embedding_router"
    assert embedding["claim"] == EMBEDDING_ROUTER_CLAIM
    assert embedding["tasks"] == keyword["tasks"]
    assert embedding["tool_selection_accuracy"] >= 0.9
    failed = [item.task_id for item in traces if not item.success]
    _, supervisor = run_embedding_supervisor_eval()
    assert supervisor["eval_kind"] == "minilm_embedding_router"
    assert supervisor["task_success_rate"] == 1.0
    assert keyword["eval_kind"] != embedding["eval_kind"]
    assert failed == []


def test_live_route_payload_is_labeled_not_live_llm() -> None:
    from services.agent_runtime.app.eval_harness import compare_route_payload

    payload = compare_route_payload("Calculate 2 + 3")
    assert payload["kind"] == "not_live_llm"
    assert payload["keyword"][0]["name"] == "calculator"
    assert "embedding" in payload


def test_hard_router_suite_is_separate_and_allowed_to_drop() -> None:
    from services.agent_runtime.app.eval_harness import (
        HARD_TASKS,
        run_hard_embedding_eval,
        run_hard_keyword_eval,
    )

    _, core = run_keyword_router_eval()
    _, hard_keyword = run_hard_keyword_eval()
    _, hard_embedding = run_hard_embedding_eval()
    assert len(HARD_TASKS) >= 6
    assert hard_keyword["suite"] == "hard"
    assert hard_embedding["suite"] == "hard"
    assert hard_keyword["tasks"] == float(len(HARD_TASKS))
    assert hard_keyword["task_success_rate"] < core["task_success_rate"]
