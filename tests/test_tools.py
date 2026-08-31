import pytest

from services.agent_runtime.app.tools import ToolError, calculator


@pytest.mark.asyncio
async def test_calculator_accepts_arithmetic_without_eval() -> None:
    assert await calculator({"expression": "12 * (3 + 4)"}) == {
        "expression": "12 * (3 + 4)",
        "value": 84.0,
    }


@pytest.mark.asyncio
async def test_calculator_rejects_calls_and_variables() -> None:
    with pytest.raises(ToolError, match="unsupported_syntax"):
        await calculator({"expression": "__import__('os').getcwd()"})
    with pytest.raises(ToolError, match="unsupported_syntax"):
        await calculator({"expression": "answer + 1"})
