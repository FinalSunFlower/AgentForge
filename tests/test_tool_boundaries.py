import pytest
from pydantic import ValidationError

from services.agent_runtime.app.policy import PolicyDenied, RunBudget, validate_tool_call
from services.agent_runtime.app.tools import ToolError, builtin_tools, calculator


@pytest.mark.asyncio
async def test_calculator_schema_and_adversarial_boundaries() -> None:
    with pytest.raises(ValidationError):
        await calculator({})
    with pytest.raises(ValidationError):
        await calculator({"expression": "1 + 1", "unexpected": True})
    with pytest.raises(ToolError, match="exponent_too_large"):
        await calculator({"expression": "10 ** 101"})
    with pytest.raises(ToolError, match="invalid_expression"):
        await calculator({"expression": "1 / 0"})


def test_tool_policy_enforces_depth_and_high_risk_approval() -> None:
    definition = builtin_tools()["calculator"]
    budget = RunBudget(max_steps=2, max_tool_depth=1, timeout_seconds=2)
    with pytest.raises(PolicyDenied, match="tool_depth_exceeded"):
        validate_tool_call(definition, {"expression": "1 + 1"}, depth=2, budget=budget)

    high_risk = definition.__class__(**{**definition.__dict__, "risk": "high"})
    with pytest.raises(PolicyDenied, match="approval_required"):
        validate_tool_call(high_risk, {"expression": "1 + 1"}, depth=1, budget=budget)

    validate_tool_call(
        high_risk,
        {"expression": "1 + 1"},
        depth=1,
        budget=budget,
        high_risk_approved=True,
    )

    with pytest.raises(PolicyDenied, match="capability_not_granted"):
        validate_tool_call(
            definition, {"expression": "1 + 1"}, depth=1, budget=budget, allowed_tools={"retrieval"}
        )
