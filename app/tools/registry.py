import asyncio

from app.models import ToolCall, ToolResult, ToolSpec
from app.tools.base import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                summary="Tool is not registered.",
                error=f"Unknown tool: {call.name}",
            )
        try:
            return await asyncio.wait_for(tool.execute(call), timeout=tool.timeout_seconds)
        except TimeoutError:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                summary="Tool execution timed out.",
                error=f"Tool timed out after {tool.timeout_seconds}s",
            )
        except Exception as exc:  # Tool boundary: normalize implementation failures.
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                summary="Tool execution failed.",
                error=str(exc),
            )
