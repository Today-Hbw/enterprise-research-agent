from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.models import AccessContext, Source, SourceType, ToolCall, ToolPermission, ToolResult
from app.tools.base import BaseTool


@dataclass(frozen=True)
class McpToolDefinition:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    permission: ToolPermission


def load_mcp_catalog(raw: str) -> dict[str, McpToolDefinition]:
    if not raw.strip():
        return {}
    entries = json.loads(raw)
    if not isinstance(entries, list):
        raise ValueError("MCP_SERVERS_JSON must be a list")
    catalog: dict[str, McpToolDefinition] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("MCP server entry must be an object")
        server, name = entry.get("server"), entry.get("name")
        if not isinstance(server, str) or not isinstance(name, str):
            raise ValueError("MCP tools require server and name")
        permission = ToolPermission(entry.get("permission", "low"))
        if permission == ToolPermission.HIGH:
            raise ValueError("High-risk MCP tools require a future approval workflow")
        definition = McpToolDefinition(
            server,
            name,
            str(entry.get("description", "")),
            entry.get("input_schema", {"type": "object"}),
            permission,
        )
        key = f"{server}:{name}"
        if key in catalog:
            raise ValueError(f"Duplicate MCP tool: {key}")
        catalog[key] = definition
    return catalog


class McpInvokeTool(BaseTool):
    name = "mcp_invoke"
    description = (
        "Discover configured MCP tool schemas; invocation requires a configured transport adapter."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {"enum": ["list", "invoke"]},
            "tool": {"type": "string"},
            "arguments": {"type": "object"},
        },
        "required": ["operation"],
    }
    permission = ToolPermission.MEDIUM
    is_stub = False

    def __init__(self, catalog: dict[str, McpToolDefinition], timeout_seconds: float) -> None:
        super().__init__(timeout_seconds)
        self.catalog = catalog

    async def execute(
        self, call: ToolCall, access_context: AccessContext | None = None
    ) -> ToolResult:
        del access_context
        if call.arguments.get("operation") != "list":
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="MCP invocation is not enabled.",
                error="No server transport adapter is configured",
            )
        tools = [
            {
                "server": item.server,
                "name": item.name,
                "description": item.description,
                "input_schema": item.input_schema,
                "permission": item.permission,
            }
            for item in self.catalog.values()
        ]
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            success=True,
            summary=f"Discovered {len(tools)} configured MCP tool schema(s).",
            data={"tools": tools},
            sources=[
                Source(
                    source_type=SourceType.MCP,
                    title="MCP tool catalog",
                    content_snippet=f"{len(tools)} server-approved MCP tools discovered.",
                )
            ],
        )
