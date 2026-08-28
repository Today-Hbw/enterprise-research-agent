import pytest

from app.models import ToolCall
from app.tools.mcp import McpInvokeTool, load_mcp_catalog


@pytest.mark.asyncio
async def test_mcp_catalog_discovers_server_approved_schema() -> None:
    catalog = load_mcp_catalog(
        '[{"server":"demo","name":"read","description":"Read","input_schema":{"type":"object"},"permission":"low"}]'
    )
    result = await McpInvokeTool(catalog, 1).execute(
        ToolCall(name="mcp_invoke", arguments={"operation": "list"})
    )

    assert result.success is True
    assert result.data["tools"][0]["server"] == "demo"


def test_mcp_catalog_rejects_high_risk_tools_without_approval_flow() -> None:
    with pytest.raises(ValueError, match="High-risk"):
        load_mcp_catalog('[{"server":"demo","name":"write","permission":"high"}]')
