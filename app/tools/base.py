from abc import ABC, abstractmethod
from typing import Any

from app.models import AccessContext, ToolCall, ToolPermission, ToolResult, ToolSpec


class BaseTool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]
    permission: ToolPermission = ToolPermission.LOW
    is_stub: bool = True

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            timeout_seconds=self.timeout_seconds,
            permission=self.permission,
            is_stub=self.is_stub,
        )

    @abstractmethod
    async def execute(
        self, call: ToolCall, access_context: AccessContext | None = None
    ) -> ToolResult:
        raise NotImplementedError
