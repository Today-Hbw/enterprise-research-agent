from typing import Protocol

from app.knowledge import KnowledgeService
from app.models import AccessContext, Source, SourceType, ToolCall, ToolResult
from app.tools.base import BaseTool


class KnowledgeBaseProvider(Protocol):
    async def list_knowledge_bases(self) -> list[dict[str, object]]: ...


class KnowledgeBaseListTool(BaseTool):
    name = "knowledge_base_list"
    description = (
        "List the RAG Platform knowledge bases authorized for this deployment. "
        "Use a returned id as knowledge_base_id when calling knowledge_search."
    )
    input_schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    is_stub = False

    def __init__(
        self,
        *,
        client: KnowledgeBaseProvider,
        timeout_seconds: float,
    ) -> None:
        super().__init__(timeout_seconds)
        self._client = client

    async def execute(
        self, call: ToolCall, access_context: AccessContext | None = None
    ) -> ToolResult:
        del access_context
        knowledge_bases = await self._client.list_knowledge_bases()
        count = len(knowledge_bases)
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            success=True,
            summary=f"Found {count} authorized knowledge base(s).",
            data={"knowledge_bases": knowledge_bases},
        )


class KnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    description = "Search authorized internal knowledge and return anchored evidence snippets."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "knowledge_base_id": {"type": "string"},
            "metadata_filters": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }
    is_stub = False

    def __init__(
        self,
        *,
        service: KnowledgeService,
        timeout_seconds: float,
        allowed_metadata_keys: set[str] | None = None,
        require_knowledge_base_id: bool = False,
    ) -> None:
        super().__init__(timeout_seconds)
        self._service = service
        self._allowed_metadata_keys = {
            key.strip() for key in allowed_metadata_keys or set() if key.strip()
        }
        properties = dict(type(self).input_schema["properties"])
        if self._allowed_metadata_keys:
            properties["metadata_filters"] = {
                "type": "object",
                "propertyNames": {"enum": sorted(self._allowed_metadata_keys)},
                "additionalProperties": {"type": "string"},
            }
        else:
            properties.pop("metadata_filters", None)
        self.input_schema = {
            **type(self).input_schema,
            "properties": properties,
            "required": (
                ["query", "knowledge_base_id"]
                if require_knowledge_base_id
                else ["query"]
            ),
        }

    async def execute(
        self, call: ToolCall, access_context: AccessContext | None = None
    ) -> ToolResult:
        if access_context is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="Knowledge search requires a server-owned access context.",
                error="Missing access context",
            )
        query = str(call.arguments.get("query", "")).strip()
        if not query:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="Knowledge search query must not be empty.",
                error="Invalid query",
            )
        knowledge_base_id = call.arguments.get("knowledge_base_id")
        if knowledge_base_id is not None:
            knowledge_base_id = str(knowledge_base_id).strip() or None
        raw_metadata_filters = call.arguments.get("metadata_filters", {})
        if not isinstance(raw_metadata_filters, dict) or not all(
            isinstance(key, str) and isinstance(value, str) and value.strip()
            for key, value in raw_metadata_filters.items()
        ):
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="Metadata filters must be non-empty string key/value pairs.",
                error="Invalid metadata filters",
            )
        metadata_filters = {
            key.strip(): value.strip() for key, value in raw_metadata_filters.items()
        }
        if not set(metadata_filters).issubset(self._allowed_metadata_keys):
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="One or more metadata filter keys are not allowed.",
                error="Disallowed metadata filter",
            )
        raw_top_k = call.arguments.get("top_k", 5)
        top_k = raw_top_k if isinstance(raw_top_k, int) and not isinstance(raw_top_k, bool) else 5
        matches = await self._service.search(
            query=query,
            tenant_id=access_context.tenant_id,
            principal_ids=access_context.principal_ids,
            knowledge_base_id=knowledge_base_id,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )
        sources = [
            Source(
                source_type=SourceType.DOCUMENT,
                title=match.title,
                url=match.source_url,
                document_id=match.document_id,
                chunk_id=match.chunk_id,
                knowledge_base_id=match.knowledge_base_id,
                char_start=match.char_start,
                char_end=match.char_end,
                score=match.score,
                content_snippet=match.content,
            )
            for match in matches
        ]
        summary = (
            f"Found {len(matches)} authorized knowledge chunk(s)."
            if matches
            else "No authorized knowledge matched the query."
        )
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            success=True,
            summary=summary,
            data={
                "ranking": self._service.ranking,
                "reranker": self._service.reranker,
                "metadata_filters": metadata_filters,
                "matches": [
                    {
                        "document_id": match.document_id,
                        "chunk_id": match.chunk_id,
                        "knowledge_base_id": match.knowledge_base_id,
                        "score": match.score,
                        "char_start": match.char_start,
                        "char_end": match.char_end,
                    }
                    for match in matches
                ],
            },
            sources=sources,
        )
