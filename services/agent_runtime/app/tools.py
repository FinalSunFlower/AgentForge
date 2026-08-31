from __future__ import annotations

import ast
import operator
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolError(Exception):
    """A user-safe tool failure that can be returned as an Observation."""


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=200)


class RetrievalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=5, ge=1, le=20)


class HandoffInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    specialist: str = Field(pattern="^(code_data|retrieval|writer)$")
    reason: str = Field(min_length=1, max_length=400)
    brief: str = Field(min_length=1, max_length=2_000)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    risk: str
    timeout_seconds: float
    execute: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    side_effect: str = "none"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ToolError("exponent_too_large")
        return _OPERATORS[type(node.op)](left, right)
    raise ToolError("expression_contains_unsupported_syntax")


async def calculator(arguments: dict[str, Any]) -> dict[str, Any]:
    payload = CalculatorInput.model_validate(arguments)
    try:
        tree = ast.parse(payload.expression, mode="eval")
        value = _evaluate(tree)
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise ToolError("invalid_expression") from exc
    if not isinstance(value, float) or abs(value) > 1e100:
        raise ToolError("result_out_of_range")
    return {"expression": payload.expression, "value": value}


async def retrieval(arguments: dict[str, Any]) -> dict[str, Any]:
    from .hybrid_retrieval import load_published_passages, retrieve_detailed

    payload = RetrievalInput.model_validate(arguments)
    passages = await load_published_passages()
    outcome = retrieve_detailed(
        payload.query,
        passages,
        limit=payload.limit,
        mode="hybrid_late_interaction",
        skip_neural_when_confident=True,
    )
    return {
        "query": payload.query,
        "method": "hybrid_rrf_late_interaction",
        "embedding": "all-MiniLM-L6-v2",
        "reranker": outcome.trace.reranker,
        "rerank_skipped": outcome.trace.rerank_skipped,
        "vector_gap": round(outcome.trace.vector_gap, 6),
        "foresight_reason": outcome.trace.reason,
        "results": [hit.as_tool_result() for hit in outcome.hits],
    }


async def handoff(arguments: dict[str, Any]) -> dict[str, Any]:
    from .handoff import resolve_specialist

    payload = HandoffInput.model_validate(arguments)
    spec = resolve_specialist(payload.specialist)
    return {
        "specialist": spec.name,
        "reason": payload.reason,
        "brief": payload.brief,
        "tools": list(spec.tools),
    }


def builtin_tools() -> dict[str, ToolDefinition]:
    from .research_tools import (
        ArxivSearchInput,
        IntentInput,
        PlotInput,
        SQLAnalyticsInput,
        arxiv_search,
        intent_router,
        plot_generator,
        readonly_sql,
    )

    calculator_definition = ToolDefinition(
        name="calculator",
        description="Evaluate a deterministic arithmetic expression. No variables or function calls.",
        input_model=CalculatorInput,
        risk="low",
        timeout_seconds=2.0,
        execute=calculator,
    )
    retrieval_definition = ToolDefinition(
        name="retrieval",
        description=(
            "Hybrid search over published papers and sections. BM25 lexical retrieval "
            "plus MiniLM semantic embeddings, fused with RRF and MiniLM late-interaction "
            "MaxSim (ColBERT-style). Selective skip when the vector gap is already large. "
            "Returns passage_id-tagged snippets."
        ),
        input_model=RetrievalInput,
        risk="medium",
        timeout_seconds=3.0,
        execute=retrieval,
    )
    arxiv_definition = ToolDefinition(
        name="arxiv_search",
        description="Search a closed arXiv-style paper catalog by title and abstract keywords.",
        input_model=ArxivSearchInput,
        risk="medium",
        timeout_seconds=3.0,
        execute=arxiv_search,
    )
    plot_definition = ToolDefinition(
        name="plot_generator",
        description="Compute series statistics and a deterministic SVG polyline from x/y values.",
        input_model=PlotInput,
        risk="medium",
        timeout_seconds=3.0,
        execute=plot_generator,
    )
    sql_definition = ToolDefinition(
        name="readonly_sql",
        description="Run a bounded read-only SELECT against approved analytics tables.",
        input_model=SQLAnalyticsInput,
        risk="high",
        timeout_seconds=5.0,
        execute=readonly_sql,
    )
    intent_definition = ToolDefinition(
        name="intent_router",
        description="Classify a request and select a research, data, math, writing, or text adapter.",
        input_model=IntentInput,
        risk="low",
        timeout_seconds=1.0,
        execute=intent_router,
    )
    handoff_definition = ToolDefinition(
        name="handoff",
        description=(
            "Transfer this run to the code-data, retrieval, or writer specialist with a brief. "
            "The thread stays with that specialist after the transfer."
        ),
        input_model=HandoffInput,
        risk="low",
        timeout_seconds=1.0,
        execute=handoff,
        side_effect="write",
    )
    definitions = [
        calculator_definition,
        retrieval_definition,
        arxiv_definition,
        plot_definition,
        sql_definition,
        intent_definition,
        handoff_definition,
    ]
    return {definition.name: definition for definition in definitions}


def catalog_for_run(
    tools: dict[str, ToolDefinition],
    allowed: set[str],
    *,
    demo_mode: bool,
) -> dict[str, ToolDefinition]:
    """Restrict the model-visible catalog. Demo Mode drops high-risk tools entirely."""
    catalog = {name: definition for name, definition in tools.items() if name in allowed}
    if demo_mode:
        return {
            name: definition for name, definition in catalog.items() if definition.risk != "high"
        }
    return catalog
