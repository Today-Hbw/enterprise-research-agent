import pytest

from app.models import ToolCall
from app.tools.python_worker import IsolatedPythonTool


@pytest.mark.asyncio
async def test_isolated_worker_evaluates_expression() -> None:
    tool = IsolatedPythonTool(timeout_seconds=2, max_output_bytes=1024)
    result = await tool.execute(
        ToolCall(
            name="python_execute",
            arguments={"expression": "revenue - cost", "variables": {"revenue": 12, "cost": 7}},
        )
    )

    assert result.success is True
    assert result.data == {"result": 5}


@pytest.mark.asyncio
async def test_isolated_worker_rejects_import_and_filesystem_access() -> None:
    tool = IsolatedPythonTool(timeout_seconds=2, max_output_bytes=1024)
    result = await tool.execute(
        ToolCall(name="python_execute", arguments={"expression": "__import__('os').getcwd()"})
    )

    assert result.success is False
    assert "rejected" in result.summary.lower()


@pytest.mark.asyncio
async def test_isolated_worker_rejects_unbounded_exponentiation() -> None:
    tool = IsolatedPythonTool(timeout_seconds=2, max_output_bytes=1024)
    result = await tool.execute(
        ToolCall(name="python_execute", arguments={"expression": "2 ** 1000000"})
    )

    assert result.success is False
