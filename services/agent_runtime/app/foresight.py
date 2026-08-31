from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from .hybrid_retrieval import RetrievalTrace, retrieve_detailed
from .research_tools import ArxivSearchInput, PlotInput, SQLAnalyticsInput
from .tools import ToolError, _evaluate


@dataclass(frozen=True)
class Action:
    name: str
    value: float


@dataclass(frozen=True)
class GateDecision:
    should_predict: bool
    confidence: float
    reason: str


@dataclass(frozen=True)
class Prediction:
    action: str
    predicted_value: float
    confidence: float


@dataclass(frozen=True)
class ToolSimulation:
    name: str
    ok: bool
    confidence: float
    kind: str
    predicted: dict[str, Any]


class ForesightPolicy(Protocol):
    async def decide(self, actions: list[Action]) -> GateDecision: ...

    async def predict(self, action: Action) -> Prediction: ...


class HeuristicForesightPolicy:
    """Variance gate over tool-outcome simulations.

    This is not RAP and not the academic tabular world-model package.
    """

    def __init__(self, uncertainty_threshold: float = 0.2) -> None:
        self.uncertainty_threshold = uncertainty_threshold

    async def decide(self, actions: list[Action]) -> GateDecision:
        if not actions:
            return GateDecision(False, 1.0, "no_candidates")
        values = [action.value for action in actions]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        confidence = 1.0 / (1.0 + variance)
        return GateDecision(
            variance > self.uncertainty_threshold, confidence, "candidate_value_variance"
        )

    async def predict(self, action: Action) -> Prediction:
        return Prediction(action.name, action.value, 1.0)


def preview_calculator(expression: str) -> float:
    simulation = simulate_calculator({"expression": expression})
    return 1.0 if simulation.ok else 0.0


def simulate_calculator(arguments: dict[str, Any]) -> ToolSimulation:
    expression = str(arguments.get("expression") or "")
    try:
        tree = ast.parse(expression, mode="eval")
        value = _evaluate(tree)
    except (SyntaxError, ZeroDivisionError, OverflowError, ToolError, TypeError, ValueError):
        return ToolSimulation("calculator", False, 1.0, "ast", {"error": "invalid_expression"})
    if not isinstance(value, float) or not math.isfinite(value) or abs(value) > 1e100:
        return ToolSimulation("calculator", False, 1.0, "ast", {"error": "result_out_of_range"})
    return ToolSimulation(
        "calculator", True, 1.0, "ast", {"value": value, "expression": expression}
    )


def simulate_sql(arguments: dict[str, Any]) -> ToolSimulation:
    try:
        payload = SQLAnalyticsInput.model_validate(arguments)
    except ValidationError as exc:
        return ToolSimulation("readonly_sql", False, 1.0, "sql_validate", {"error": str(exc)[:200]})
    return ToolSimulation(
        "readonly_sql",
        True,
        0.85,
        "sql_validate",
        {"would_execute": True, "query": payload.query, "read_only": True},
    )


def simulate_arxiv(arguments: dict[str, Any]) -> ToolSimulation:
    try:
        payload = ArxivSearchInput.model_validate(arguments)
    except ValidationError as exc:
        return ToolSimulation(
            "arxiv_search", False, 1.0, "catalog_preview", {"error": str(exc)[:200]}
        )
    query = payload.query.strip()
    return ToolSimulation(
        "arxiv_search",
        True,
        0.8,
        "catalog_preview",
        {"would_search": True, "query": query, "limit": payload.limit},
    )


def simulate_plot(arguments: dict[str, Any]) -> ToolSimulation:
    try:
        payload = PlotInput.model_validate(arguments)
    except ValidationError as exc:
        return ToolSimulation(
            "plot_generator", False, 1.0, "closed_form_estimate", {"error": str(exc)[:200]}
        )
    return ToolSimulation(
        "plot_generator",
        True,
        1.0,
        "closed_form_estimate",
        {"n": len(payload.ys), "title": payload.title},
    )


def simulate_retrieval(
    arguments: dict[str, Any], passages: list[Any] | None = None
) -> ToolSimulation:
    query = str(arguments.get("query") or "").strip()
    if not query:
        return ToolSimulation("retrieval", False, 1.0, "vector_preview", {"error": "empty_query"})
    if not passages:
        return ToolSimulation(
            "retrieval", True, 0.2, "vector_preview", {"hits": 0, "note": "no_corpus_yet"}
        )
    limit = int(arguments.get("limit") or 3)
    outcome = retrieve_detailed(query, passages, limit=limit, mode="vector")
    top = [hit.passage.passage_id for hit in outcome.hits]
    confidence = min(1.0, outcome.trace.vector_gap / 0.18) if outcome.hits else 0.1
    return ToolSimulation(
        "retrieval",
        True,
        round(confidence, 4),
        "vector_preview",
        {"top_ids": top, "vector_gap": round(outcome.trace.vector_gap, 6), "reranker": "skipped"},
    )


def simulate_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    retrieval_trace: RetrievalTrace | None = None,
    passages: list[Any] | None = None,
) -> ToolSimulation:
    if name == "calculator":
        return simulate_calculator(arguments)
    if name == "readonly_sql":
        return simulate_sql(arguments)
    if name == "arxiv_search":
        return simulate_arxiv(arguments)
    if name == "plot_generator":
        return simulate_plot(arguments)
    if name == "retrieval":
        if retrieval_trace is not None and passages is None:
            return ToolSimulation(
                "retrieval",
                True,
                min(1.0, retrieval_trace.vector_gap / 0.18),
                "vector_preview",
                {"vector_gap": round(retrieval_trace.vector_gap, 6)},
            )
        return simulate_retrieval(arguments, passages)
    return ToolSimulation(name, True, 0.5, "uninformative_prior", {})


def preview_tool(
    name: str, arguments: dict[str, Any], retrieval_trace: RetrievalTrace | None = None
) -> float:
    simulation = simulate_tool(name, arguments, retrieval_trace=retrieval_trace)
    if not simulation.ok:
        return 0.0
    return simulation.confidence
