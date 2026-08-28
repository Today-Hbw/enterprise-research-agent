import pytest

from app.models import ToolCall, ToolPermission
from app.tools.stubs import build_stub_registry


@pytest.mark.asyncio
async def test_registry_exposes_all_stub_tools() -> None:
    registry = build_stub_registry(timeout_seconds=1)
    names = {spec.name for spec in registry.specs()}

    assert names == {
        "knowledge_search",
        "web_search",
        "http_fetch",
        "schema_search",
        "execute_sql",
        "python_execute",
        "browser",
        "mcp_invoke",
    }
    assert all(spec.is_stub for spec in registry.specs())


@pytest.mark.asyncio
async def test_sql_stub_rejects_mutation() -> None:
    registry = build_stub_registry(timeout_seconds=1)
    result = await registry.execute(
        ToolCall(name="execute_sql", arguments={"sql": "DELETE FROM fact_purchase_orders"})
    )

    assert result.success is False
    assert "SELECT" in result.summary


@pytest.mark.asyncio
async def test_registry_rejects_tools_above_configured_permission_ceiling() -> None:
    registry = build_stub_registry(timeout_seconds=1, max_permission=ToolPermission.LOW)

    assert {spec.name for spec in registry.specs()} == {
        "knowledge_search",
        "web_search",
        "http_fetch",
        "mcp_invoke",
    }

    result = await registry.execute(ToolCall(name="browser", arguments={"task": "export"}))

    assert result.success is False
    assert result.error == "Tool permission high exceeds configured maximum low."


@pytest.mark.asyncio
async def test_registry_allows_tools_at_configured_permission_ceiling() -> None:
    registry = build_stub_registry(timeout_seconds=1, max_permission=ToolPermission.LOW)

    result = await registry.execute(ToolCall(name="web_search", arguments={"query": "market"}))

    assert result.success is True
