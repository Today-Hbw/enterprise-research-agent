from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from app.models import AccessContext, Source, SourceType, ToolCall, ToolPermission, ToolResult
from app.tools.base import BaseTool


@dataclass(frozen=True)
class McpToolDefinition:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    permission: ToolPermission
    transport_url: str | None = None


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
        transport_url = entry.get("transport_url")
        if transport_url is not None:
            if not isinstance(transport_url, str) or not transport_url.strip():
                raise ValueError("MCP transport_url must be a non-empty string")
            parsed = urlparse(transport_url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                raise ValueError("MCP transport_url must be an HTTPS URL without credentials")
        definition = McpToolDefinition(
            server,
            name,
            str(entry.get("description", "")),
            entry.get("input_schema", {"type": "object"}),
            permission,
            transport_url,
        )
        key = f"{server}:{name}"
        if key in catalog:
            raise ValueError(f"Duplicate MCP tool: {key}")
        catalog[key] = definition
    return catalog


class McpInvokeTool(BaseTool):
    name = "mcp_invoke"
    description = "Discover or invoke server-configured, low-risk MCP tools."
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

    def __init__(
        self,
        catalog: dict[str, McpToolDefinition],
        timeout_seconds: float,
        allowed_hosts: set[str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(timeout_seconds)
        self.catalog = catalog
        self.allowed_hosts = {
            host.strip().lower() for host in allowed_hosts or set() if host.strip()
        }
        self.client = client

    async def execute(
        self, call: ToolCall, access_context: AccessContext | None = None
    ) -> ToolResult:
        del access_context
        operation = call.arguments.get("operation")
        if operation == "list":
            return self._catalog_result(call)
        if operation != "invoke":
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="Unsupported MCP operation.",
                error="operation must be list or invoke",
            )
        return await self._invoke(call)

    def _catalog_result(self, call: ToolCall) -> ToolResult:
        tools = [
            {
                "server": item.server,
                "name": item.name,
                "description": item.description,
                "input_schema": item.input_schema,
                "permission": item.permission,
                "invocation_enabled": self._is_transport_enabled(item),
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

    async def _invoke(self, call: ToolCall) -> ToolResult:
        tool_key = call.arguments.get("tool")
        arguments = call.arguments.get("arguments", {})
        if not isinstance(tool_key, str) or tool_key not in self.catalog:
            return self._failure(call, "MCP tool is not configured.")
        if not isinstance(arguments, dict):
            return self._failure(call, "MCP arguments must be an object.")

        definition = self.catalog[tool_key]
        if definition.permission != ToolPermission.LOW:
            return self._failure(call, "Only low-risk MCP tools may be invoked.")
        if not self._is_transport_enabled(definition):
            return self._failure(call, "MCP transport is not enabled for this tool.")

        try:
            data = await self._post(definition, arguments)
        except (httpx.HTTPError, ValueError) as exc:
            return self._failure(call, "MCP tool request failed.", str(exc))
        serialized = json.dumps(data, ensure_ascii=False, default=str)
        if len(serialized) > 65_536:
            return self._failure(call, "MCP tool response exceeds the 64 KiB limit.")
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            success=True,
            summary=f"Invoked low-risk MCP tool {tool_key}.",
            data={"tool": tool_key, "result": data},
            sources=[
                Source(
                    source_type=SourceType.MCP,
                    title=f"MCP: {tool_key}",
                    content_snippet=serialized[:1_000],
                )
            ],
        )

    async def _post(self, definition: McpToolDefinition, arguments: dict[str, Any]) -> Any:
        payload = {"tool": definition.name, "arguments": arguments}
        if self.client is not None:
            response = await self.client.post(definition.transport_url, json=payload)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(definition.transport_url, json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, (dict, list)):
            raise ValueError("MCP transport response must be a JSON object or array")
        return data

    def _is_transport_enabled(self, definition: McpToolDefinition) -> bool:
        if not definition.transport_url or not self.allowed_hosts:
            return False
        return urlparse(definition.transport_url).hostname in self.allowed_hosts

    def _failure(self, call: ToolCall, summary: str, error: str | None = None) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            success=False,
            summary=summary,
            error=error or summary,
        )
