from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tools import ToolDefinition


@dataclass(frozen=True)
class RunBudget:
    max_steps: int
    max_tool_depth: int
    timeout_seconds: int


class PolicyDenied(Exception):
    pass


def validate_tool_call(
    definition: ToolDefinition,
    arguments: dict[str, Any],
    *,
    depth: int,
    budget: RunBudget,
    allowed_tools: set[str] | None = None,
    high_risk_approved: bool = False,
) -> None:
    if allowed_tools is not None and definition.name not in allowed_tools:
        raise PolicyDenied("capability_not_granted")
    if depth > budget.max_tool_depth:
        raise PolicyDenied("tool_depth_exceeded")
    if definition.risk == "high" and not high_risk_approved:
        raise PolicyDenied("approval_required")
    definition.input_model.model_validate(arguments)
