import json

import httpx
import pytest

from app.models import ToolCall
from app.tools.mcp import McpInvokeTool, load_mcp_catalog


def _catalog(permission: str = "low") -> dict:
    return load_mcp_catalog(
        '[{"server":"demo","name":"read","description":"Read","input_schema":{"type":"object"},"permission":"'
        + permission
        + '","transport_url":"https://mcp.example.com/tools/call"}]'
    )


@pytest.mark.asyncio
async def test_mcp_catalog_discovers_server_approved_schema() -> None:
    result = await McpInvokeTool(_catalog(), 1, {"mcp.example.com"}).execute(
        ToolCall(name="mcp_invoke", arguments={"operation": "list"})
    )

    assert result.success is True
    assert result.data["tools"][0]["server"] == "demo"
    assert result.data["tools"][0]["invocation_enabled"] is True


@pytest.mark.asyncio
async def test_mcp_invokes_only_allowlisted_low_risk_tool() -> None:
    seen: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"items": ["approved result"]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await McpInvokeTool(_catalog(), 1, {"mcp.example.com"}, client).execute(
            ToolCall(
                name="mcp_invoke",
                arguments={"operation": "invoke", "tool": "demo:read", "arguments": {"id": "123"}},
            )
        )
    finally:
        await client.aclose()

    assert result.success is True
    assert result.data["result"] == {"items": ["approved result"]}
    assert seen == {
        "url": "https://mcp.example.com/tools/call",
        "payload": {"tool": "read", "arguments": {"id": "123"}},
    }


@pytest.mark.asyncio
async def test_mcp_rejects_disabled_or_medium_risk_invocation() -> None:
    disabled = await McpInvokeTool(_catalog(), 1, set()).execute(
        ToolCall(name="mcp_invoke", arguments={"operation": "invoke", "tool": "demo:read"})
    )
    medium = await McpInvokeTool(_catalog("medium"), 1, {"mcp.example.com"}).execute(
        ToolCall(name="mcp_invoke", arguments={"operation": "invoke", "tool": "demo:read"})
    )

    assert disabled.success is False
    assert medium.success is False
    assert "low-risk" in medium.summary


def test_mcp_catalog_rejects_high_risk_tools_without_approval_flow() -> None:
    with pytest.raises(ValueError, match="High-risk"):
        load_mcp_catalog('[{"server":"demo","name":"write","permission":"high"}]')


def test_mcp_catalog_rejects_insecure_transport_url() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        load_mcp_catalog(
            '[{"server":"demo","name":"read","transport_url":"http://mcp.example.com"}]'
        )
