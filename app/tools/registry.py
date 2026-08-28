import asyncio

from app.models import AccessContext, ToolCall, ToolPermission, ToolResult, ToolSpec
from app.tools.base import BaseTool

_PERMISSION_RANK = {
    ToolPermission.LOW: 0,
    ToolPermission.MEDIUM: 1,
    ToolPermission.HIGH: 2,
}


class ToolRegistry:
    def __init__(self, max_permission: ToolPermission = ToolPermission.HIGH) -> None:
        self._tools: dict[str, BaseTool] = {}
        self.max_permission = max_permission

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        return [
            tool.spec
            for tool in self._tools.values()
            if _PERMISSION_RANK[tool.permission] <= _PERMISSION_RANK[self.max_permission]
        ]

    def is_stub(self, name: str) -> bool:
        tool = self._tools.get(name)
        return True if tool is None else tool.is_stub

    async def execute(
        self, call: ToolCall, access_context: AccessContext | None = None
    ) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                summary="Tool is not registered.",
                error=f"Unknown tool: {call.name}",
            )
        if _PERMISSION_RANK[tool.permission] > _PERMISSION_RANK[self.max_permission]:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                summary="Tool execution is not permitted by server policy.",
                error=(
                    f"Tool permission {tool.permission} exceeds configured maximum "
                    f"{self.max_permission}."
                ),
            )
        try:
            return await asyncio.wait_for(
                tool.execute(call, access_context), timeout=tool.timeout_seconds
            )
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
