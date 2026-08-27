from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ToolPermission(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SourceType(StrEnum):
    DOCUMENT = "document"
    WEB = "web"
    API = "api"
    SQL = "sql"
    BROWSER = "browser"
    MCP = "mcp"


class Message(BaseModel):
    id: str = Field(default_factory=lambda: new_id("msg"))
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=utc_now)
    run_id: str | None = None


class Source(BaseModel):
    source_id: str = Field(default_factory=lambda: new_id("src"))
    source_type: SourceType
    title: str
    url: str | None = None
    document_id: str | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    content_snippet: str


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    timeout_seconds: float
    permission: ToolPermission
    is_stub: bool = True


class ToolCall(BaseModel):
    call_id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @property
    def signature(self) -> str:
        ordered = sorted((key, repr(value)) for key, value in self.arguments.items())
        return f"{self.name}:{ordered}"


class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    success: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    sources: list[Source] = Field(default_factory=list)
    error: str | None = None


class AgentDecision(BaseModel):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: str | None = None
    decision_summary: str
    input_tokens: int = 0
    output_tokens: int = 0
    provider_response_id: str | None = None


class TraceStep(BaseModel):
    step_id: str = Field(default_factory=lambda: new_id("step"))
    index: int
    kind: str
    summary: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output_summary: str | None = None
    status: str = "completed"
    started_at: datetime = Field(default_factory=utc_now)
    duration_ms: int = 0
    error: str | None = None


class RunMetrics(BaseModel):
    llm_call_count: int = 0
    tool_call_count: int = 0
    duration_ms: int = 0
    token_usage: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float | None = None


class RunRecord(BaseModel):
    run_id: str = Field(default_factory=lambda: new_id("run"))
    conversation_id: str
    user_query: str
    model: str
    status: RunStatus = RunStatus.RUNNING
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    final_answer: str | None = None
    sources: list[Source] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    error: str | None = None


class Conversation(BaseModel):
    conversation_id: str = Field(default_factory=lambda: new_id("conv"))
    title: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    messages: list[Message] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    run: RunRecord


class StreamEvent(BaseModel):
    event: str
    sequence: int
    run_id: str
    data: dict[str, Any] = Field(default_factory=dict)
