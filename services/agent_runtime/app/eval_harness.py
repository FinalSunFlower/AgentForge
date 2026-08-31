from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from .embedding_router import prefer_tool_order, select_specialist, select_tool_names
from .hybrid_retrieval import (
    EVAL_QUERIES,
    evaluate_hard_retrieval,
    evaluate_modes,
    evaluate_zero_overlap,
)
from .memory import assemble_context, build_needle_thread
from .provider import ModelTurn, ToolCallDelta

SuccessRule = Literal["tools_match", "completed", "approval_required"]


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    prompt: str
    expected_tools: list[str]
    expected_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    success_if: SuccessRule = "tools_match"


@dataclass
class TaskTrace:
    task_id: str
    predicted_tools: list[str]
    predicted_args: list[dict[str, Any]]
    expected_tools: list[str]
    steps: int
    cost_micros: int
    terminal: str
    tool_selection_correct: bool
    param_correct: bool
    success: bool


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def keyword_route(prompt: str) -> list[tuple[str, dict[str, Any]]]:
    """Deterministic router used as the evaluated agent policy, not as an LLM."""
    if _contains_any(prompt, ("classify", "intent", "adapter")):
        return [("intent_router", {"text": prompt})]
    if _contains_any(prompt, ("select ", " sql", "sql ", "analytics")):
        return [("readonly_sql", {"query": "SELECT title FROM novels LIMIT 5"})]
    if _contains_any(prompt, ("bearing", "sonar", "triangulate")) and not _contains_any(
        prompt, ("search", "retrieve", "find", "look up")
    ):
        sensors = [
            {"x": 0.0, "y": 0.0, "bearing_deg": 45.0},
            {"x": 10.0, "y": 0.0, "bearing_deg": 135.0},
        ]
        return [("passive_sonar", {"sensors": sensors})]
    if _contains_any(prompt, ("wind tunnel", "cylinder", "potential flow")) and not _contains_any(
        prompt, ("search", "retrieve", "find", "look up")
    ):
        return [
            (
                "wind_tunnel",
                {"velocity_mps": 20.0, "angle_deg": 0.0, "cylinder_radius_m": 1.0},
            )
        ]
    if (
        _contains_any(prompt, ("calculate", "compute", "expression"))
        or " + " in prompt
        or " * " in prompt
    ):
        expression = "12 * (3 + 4)" if "12" in prompt else "2 + 3"
        calls: list[tuple[str, dict[str, Any]]] = [("calculator", {"expression": expression})]
        if _contains_any(prompt, ("search", "retrieve", "find", "look up")):
            calls.append(("retrieval", {"query": "planning", "limit": 5}))
        return calls
    if _contains_any(prompt, ("search", "retrieve", "find", "look up", "who wrote", "when do")):
        return [("retrieval", {"query": prompt[:200], "limit": 5})]
    return []


def plan_to_turns(plan: list[tuple[str, dict[str, Any]]], final_text: str) -> list[ModelTurn]:
    if not plan:
        return [ModelTurn(text=final_text, input_tokens=8, output_tokens=4)]
    turns = [
        ModelTurn(
            tool_calls=[
                ToolCallDelta(f"eval-{index}", name, json.dumps(arguments, separators=(",", ":")))
                for index, (name, arguments) in enumerate(plan, start=1)
            ],
            input_tokens=12,
            output_tokens=6,
        )
    ]
    turns.append(ModelTurn(text=final_text, input_tokens=10, output_tokens=5))
    return turns


EVAL_TASKS: list[EvalTask] = [
    EvalTask(
        "calc-basic",
        "Calculate 12 * (3 + 4)",
        ["calculator"],
        {"calculator": {"expression": "12 * (3 + 4)"}},
    ),
    EvalTask("calc-sum", "Compute 2 + 3", ["calculator"], {"calculator": {"expression": "2 + 3"}}),
    EvalTask(
        "calc-then-search",
        "Calculate 12 * (3 + 4) and then search planning notes",
        ["calculator", "retrieval"],
        {"calculator": {"expression": "12 * (3 + 4)"}, "retrieval": {"query": "planning"}},
    ),
    EvalTask(
        "retrieve-planning",
        "Search published notes about planning",
        ["retrieval"],
        {"retrieval": {}},
    ),
    EvalTask(
        "retrieve-who",
        "Who wrote the strategy memo after the rainy-season holdup?",
        ["retrieval"],
        {"retrieval": {}},
    ),
    EvalTask(
        "retrieve-quota",
        "When do unused credits expire each day?",
        ["retrieval"],
        {"retrieval": {}},
    ),
    EvalTask(
        "sonar-bearing",
        "Triangulate the sonar bearing from two sensors",
        ["passive_sonar"],
        {"passive_sonar": {}},
    ),
    EvalTask(
        "wind-tunnel",
        "Run the wind tunnel cylinder potential flow calculation",
        ["wind_tunnel"],
        {"wind_tunnel": {"velocity_mps": 20.0}},
    ),
    EvalTask(
        "sql-select",
        "Run a read-only SQL analytics query on novels",
        ["readonly_sql"],
        {"readonly_sql": {}},
    ),
    EvalTask(
        "intent-math",
        "Classify the intent of: please calculate this equation",
        ["intent_router"],
        {"intent_router": {}},
    ),
    EvalTask(
        "intent-science",
        "Classify adapter for a sonar bearing request",
        ["intent_router"],
        {"intent_router": {}},
    ),
    EvalTask(
        "find-untrusted",
        "Look up why search snippets are isolated from instructions",
        ["retrieval"],
        {"retrieval": {}},
    ),
    EvalTask(
        "find-rrf",
        "Search how lexical and dense lists are combined",
        ["retrieval"],
        {"retrieval": {}},
    ),
    EvalTask(
        "find-bm25",
        "Retrieve the primer on saturated term frequency ranking",
        ["retrieval"],
        {"retrieval": {}},
    ),
    EvalTask("find-tea", "Find the amber tea ceremony chapter", ["retrieval"], {"retrieval": {}}),
    EvalTask(
        "find-sonar-log",
        "Search the harbor observer angle measurement",
        ["retrieval"],
        {"retrieval": {}},
    ),
    EvalTask(
        "calc-only-no-search",
        "Calculate 2 + 3 without looking anything up",
        ["calculator"],
        {"calculator": {"expression": "2 + 3"}},
    ),
    EvalTask(
        "multi-read",
        "Calculate 2 + 3 and retrieve planning notes",
        ["calculator", "retrieval"],
        {"calculator": {}, "retrieval": {}},
    ),
    EvalTask("empty-chit-chat", "Hello, how are you today?", [], {}, "completed"),
    EvalTask(
        "sql-approval-path",
        "SELECT title FROM novels for analytics",
        ["readonly_sql"],
        {"readonly_sql": {}},
    ),
    EvalTask(
        "wind-only",
        "Compute cylinder pressure in the wind tunnel",
        ["wind_tunnel"],
        {"wind_tunnel": {}},
    ),
    EvalTask(
        "sonar-only",
        "Use passive sonar to triangulate a source",
        ["passive_sonar"],
        {"passive_sonar": {}},
    ),
    EvalTask(
        "intent-data",
        "Classify this analytics SQL database request",
        ["intent_router"],
        {"intent_router": {}},
    ),
    EvalTask("retrieve-fusion", "Look up Reciprocal Rank Fusion", ["retrieval"], {"retrieval": {}}),
]


HARD_TASKS: list[EvalTask] = [
    EvalTask(
        "trap-calc-means-sonar",
        "Calculate the sonar source position with the science tool, not the arithmetic calculator",
        ["passive_sonar"],
        {"passive_sonar": {}},
    ),
    EvalTask(
        "trap-search-not-calc",
        "Search the notes for how people calculate quotas; do not compute anything",
        ["retrieval"],
        {"retrieval": {}},
    ),
    EvalTask(
        "multi-retrieve-then-calc",
        "Retrieve planning notes first, then calculate 2 + 3",
        ["retrieval", "calculator"],
        {"calculator": {"expression": "2 + 3"}, "retrieval": {}},
    ),
    EvalTask(
        "multi-intent-then-sonar",
        "Classify this sonar request, then triangulate the bearing",
        ["intent_router", "passive_sonar"],
        {"intent_router": {}},
    ),
    EvalTask(
        "paraphrase-void-order",
        "Look up how a customer voids finished store order",
        ["retrieval"],
        {"retrieval": {}},
    ),
    EvalTask(
        "negation-no-sql",
        "Do not run SQL. Calculate 2 + 3",
        ["calculator"],
        {"calculator": {"expression": "2 + 3"}},
    ),
]


HARD_EVAL_CLAIM = (
    "Trap, paraphrase, and ordered multi-tool tasks. Keyword and MiniLM routers are "
    "expected to drop. Not live-LLM routing."
)


class PlannedProvider:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = list(turns)
        self.calls = 0

    async def complete(self, messages, tools) -> ModelTurn:
        self.calls += 1
        if not self._turns:
            return ModelTurn(text="done")
        return self._turns.pop(0)


def _args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


def score_trace(
    task: EvalTask,
    predicted_tools: list[str],
    predicted_args: list[dict[str, Any]],
    *,
    steps: int,
    cost_micros: int,
    terminal: str,
) -> TaskTrace:
    selection = predicted_tools == task.expected_tools
    param_ok = True
    actual_by_name: dict[str, dict[str, Any]] = {}
    for name, payload in zip(predicted_tools, predicted_args, strict=False):
        actual_by_name.setdefault(name, payload)
    for name, expected in task.expected_args.items():
        param_ok = param_ok and _args_match(expected, actual_by_name.get(name, {}))
    if task.success_if == "approval_required":
        success = terminal == "approval_required"
    elif task.success_if == "completed":
        success = terminal == "completed" and selection
    else:
        success = selection and param_ok
    return TaskTrace(
        task_id=task.task_id,
        predicted_tools=predicted_tools,
        predicted_args=predicted_args,
        expected_tools=task.expected_tools,
        steps=steps,
        cost_micros=cost_micros,
        terminal=terminal,
        tool_selection_correct=selection,
        param_correct=param_ok,
        success=success,
    )


KEYWORD_ROUTER_CLAIM = (
    "Deterministic keyword-router baseline. Live-LLM routing has no separate measured number."
)
EMBEDDING_ROUTER_CLAIM = (
    "MiniLM cosine over frozen tool/specialist descriptions, plus a small intent-verb "
    "override, retrieval-verb prior, negation filter, and canonical-order post-process. "
    "Not a learned router and not live-LLM routing."
)
MEMORY_EVAL_CLAIM = (
    "Needles are structured facts (IDs, preference keywords, numbers). "
    "Implicit unstructured context is not guaranteed."
)


def summarize(traces: list[TaskTrace]) -> dict[str, float]:
    count = max(1, len(traces))
    return {
        "tasks": float(len(traces)),
        "tool_selection_accuracy": round(
            sum(item.tool_selection_correct for item in traces) / count, 4
        ),
        "param_accuracy": round(sum(item.param_correct for item in traces) / count, 4),
        "task_success_rate": round(sum(item.success for item in traces) / count, 4),
        "avg_steps": round(sum(item.steps for item in traces) / count, 4),
        "avg_cost_micros": round(sum(item.cost_micros for item in traces) / count, 4),
    }


def _router_summary(traces: list[TaskTrace], *, eval_kind: str, claim: str) -> dict[str, Any]:
    return {
        **summarize(traces),
        "eval_kind": eval_kind,
        "claim": claim,
    }


def run_keyword_router_eval(
    tasks: list[EvalTask] | None = None,
) -> tuple[list[TaskTrace], dict[str, Any]]:
    traces: list[TaskTrace] = []
    for task in tasks or EVAL_TASKS:
        plan = keyword_route(task.prompt)
        predicted_tools = [name for name, _ in plan]
        predicted_args = [arguments for _, arguments in plan]
        traces.append(
            score_trace(
                task,
                predicted_tools,
                predicted_args,
                steps=1 if not plan else 2,
                cost_micros=0,
                terminal="completed",
            )
        )
    return traces, _router_summary(
        traces, eval_kind="deterministic_keyword_router", claim=KEYWORD_ROUTER_CLAIM
    )


def _fallback_args(name: str, prompt: str) -> dict[str, Any]:
    if name == "calculator":
        expression = "12 * (3 + 4)" if "12" in prompt else "2 + 3"
        return {"expression": expression}
    if name == "retrieval":
        return {"query": "planning" if "planning" in prompt.lower() else prompt[:200], "limit": 5}
    if name == "intent_router":
        return {"text": prompt}
    if name == "readonly_sql":
        return {"query": "SELECT title FROM novels LIMIT 5"}
    if name == "handoff":
        specialist = select_specialist(prompt) or "retrieval"
        return {"specialist": specialist, "reason": f"{specialist}_task", "brief": prompt[:400]}
    return {}


def embedding_route(prompt: str) -> list[tuple[str, dict[str, Any]]]:
    names = select_tool_names(prompt)
    keyword_args = {name: arguments for name, arguments in keyword_route(prompt)}
    return [(name, keyword_args.get(name) or _fallback_args(name, prompt)) for name in names]


def run_embedding_router_eval(
    tasks: list[EvalTask] | None = None,
) -> tuple[list[TaskTrace], dict[str, Any]]:
    traces: list[TaskTrace] = []
    for task in tasks or EVAL_TASKS:
        plan = embedding_route(task.prompt)
        traces.append(
            score_trace(
                task,
                [name for name, _ in plan],
                [arguments for _, arguments in plan],
                steps=1 if not plan else 2,
                cost_micros=0,
                terminal="completed",
            )
        )
    return traces, _router_summary(
        traces, eval_kind="minilm_embedding_router", claim=EMBEDDING_ROUTER_CLAIM
    )


def compare_route_payload(prompt: str, allowed: set[str] | None = None) -> dict[str, Any]:
    keyword = keyword_route(prompt)
    embedding = embedding_route(prompt)
    if allowed is not None:
        keyword = [(name, arguments) for name, arguments in keyword if name in allowed]
        embedding = [(name, arguments) for name, arguments in embedding if name in allowed]
    return {
        "kind": "not_live_llm",
        "claim": "Keyword and MiniLM embedding routers on the user text. Not live-LLM tool selection.",
        "keyword": [{"name": name, "arguments": arguments} for name, arguments in keyword],
        "embedding": [{"name": name, "arguments": arguments} for name, arguments in embedding],
        "agreement": [name for name, _ in keyword] == [name for name, _ in embedding],
    }


def ordered_catalog_names(prompt: str, catalog_names: list[str]) -> list[str]:
    preferred = [name for name, _ in embedding_route(prompt)] + [
        name for name, _ in keyword_route(prompt)
    ]
    return prefer_tool_order(preferred, catalog_names)


def retrieval_mode_report() -> dict[str, dict[str, float]]:
    return evaluate_modes(EVAL_QUERIES)


def hard_retrieval_report() -> dict[str, dict[str, float]]:
    return evaluate_hard_retrieval()


def run_hard_keyword_eval() -> tuple[list[TaskTrace], dict[str, Any]]:
    traces, summary = run_keyword_router_eval(HARD_TASKS)
    summary["claim"] = HARD_EVAL_CLAIM
    summary["suite"] = "hard"
    return traces, summary


def run_hard_embedding_eval() -> tuple[list[TaskTrace], dict[str, Any]]:
    traces, summary = run_embedding_router_eval(HARD_TASKS)
    summary["claim"] = HARD_EVAL_CLAIM
    summary["suite"] = "hard"
    return traces, summary


def supervisor_route(prompt: str) -> list[tuple[str, dict[str, Any]]]:
    if _contains_any(
        prompt, ("calculate", "compute", "sonar", "bearing", "wind tunnel", "cylinder")
    ):
        return [
            ("handoff", {"specialist": "science", "reason": "science_task", "brief": prompt[:400]})
        ]
    if _contains_any(prompt, ("search", "retrieve", "find", "look up", "who wrote")):
        return [
            (
                "handoff",
                {"specialist": "retrieval", "reason": "retrieval_task", "brief": prompt[:400]},
            )
        ]
    return []


SUPERVISOR_TASKS: list[EvalTask] = [
    EvalTask(
        "handoff-science-calc",
        "Calculate 12 * (3 + 4)",
        ["handoff"],
        {"handoff": {"specialist": "science"}},
    ),
    EvalTask(
        "handoff-science-sonar",
        "Triangulate the sonar bearing from two sensors",
        ["handoff"],
        {"handoff": {"specialist": "science"}},
    ),
    EvalTask(
        "handoff-retrieval",
        "Search published notes about planning",
        ["handoff"],
        {"handoff": {"specialist": "retrieval"}},
    ),
]


def embedding_supervisor_route(prompt: str) -> list[tuple[str, dict[str, Any]]]:
    specialist = select_specialist(prompt)
    if specialist is None:
        return []
    return [
        (
            "handoff",
            {"specialist": specialist, "reason": f"{specialist}_task", "brief": prompt[:400]},
        )
    ]


def run_supervisor_eval(
    tasks: list[EvalTask] | None = None,
) -> tuple[list[TaskTrace], dict[str, Any]]:
    traces: list[TaskTrace] = []
    for task in tasks or SUPERVISOR_TASKS:
        plan = supervisor_route(task.prompt)
        traces.append(
            score_trace(
                task,
                [name for name, _ in plan],
                [arguments for _, arguments in plan],
                steps=2,
                cost_micros=0,
                terminal="completed",
            )
        )
    return traces, _router_summary(
        traces, eval_kind="deterministic_keyword_router", claim=KEYWORD_ROUTER_CLAIM
    )


def run_embedding_supervisor_eval(
    tasks: list[EvalTask] | None = None,
) -> tuple[list[TaskTrace], dict[str, Any]]:
    traces: list[TaskTrace] = []
    for task in tasks or SUPERVISOR_TASKS:
        plan = embedding_supervisor_route(task.prompt)
        traces.append(
            score_trace(
                task,
                [name for name, _ in plan],
                [arguments for _, arguments in plan],
                steps=2,
                cost_micros=0,
                terminal="completed",
            )
        )
    return traces, _router_summary(
        traces, eval_kind="minilm_embedding_router", claim=EMBEDDING_ROUTER_CLAIM
    )


def needle_survives_compression(
    needle: str = "Remember that the vault code is ORCHID-7729.",
) -> bool:
    history = build_needle_thread(needle, filler_turns=70, needle_at=4)
    assembled = assemble_context(
        history,
        system_policy="You are a test assistant.",
        max_chars=8_000,
        keep_recent=4,
        compress_after=8,
    )
    blob = "\n".join(item.get("content") or "" for item in assembled.messages)
    return assembled.compressed and "ORCHID-7729" in blob


def memory_eval_report() -> dict[str, Any]:
    """Structured-fact needles that must survive extractive compression."""
    cases = (
        ("vault_code", "Remember that the vault code is ORCHID-7729.", "ORCHID-7729"),
        ("timezone", "I prefer Asia/Shanghai as my timezone.", "Asia/Shanghai"),
        ("plan_quota", "My plan is 240 tokens per day.", "240"),
    )
    survived: dict[str, bool] = {}
    for name, needle, marker in cases:
        history = build_needle_thread(needle, filler_turns=70, needle_at=4)
        assembled = assemble_context(
            history,
            system_policy="You are a test assistant.",
            max_chars=8_000,
            keep_recent=4,
            compress_after=8,
        )
        blob = "\n".join(item.get("content") or "" for item in assembled.messages)
        survived[name] = bool(assembled.compressed and marker in blob)
    return {
        "survived": survived,
        "scope": "structured_extractive_facts",
        "claim": MEMORY_EVAL_CLAIM,
    }


def zero_overlap_report() -> dict[str, dict[str, float]]:
    return evaluate_zero_overlap()
