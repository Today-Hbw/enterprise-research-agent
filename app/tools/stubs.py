from typing import Any

from app.models import Source, SourceType, ToolCall, ToolPermission, ToolResult
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry


class FixedResultTool(BaseTool):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        summary: str,
        data: dict[str, Any],
        source_type: SourceType,
        source_title: str,
        source_url: str | None,
        timeout_seconds: float,
        permission: ToolPermission = ToolPermission.LOW,
    ) -> None:
        super().__init__(timeout_seconds)
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.permission = permission
        self._summary = summary
        self._data = data
        self._source_type = source_type
        self._source_title = source_title
        self._source_url = source_url

    async def execute(self, call: ToolCall) -> ToolResult:
        source = Source(
            source_type=self._source_type,
            title=self._source_title,
            url=self._source_url,
            document_id=None if self._source_url else "stub-document",
            content_snippet=self._summary,
        )
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            success=True,
            summary=self._summary,
            data={**self._data, "stub": True, "received_input": call.arguments},
            sources=[source],
        )


class SqlStubTool(FixedResultTool):
    async def execute(self, call: ToolCall) -> ToolResult:
        sql = str(call.arguments.get("sql", "")).strip()
        if not sql.lower().startswith("select"):
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="Only SELECT statements are allowed, including in stub mode.",
                error="SQL validator rejected a non-SELECT statement",
            )
        return await super().execute(call)


def build_stub_registry(timeout_seconds: float) -> ToolRegistry:
    registry = ToolRegistry()
    common_query_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    tools: list[BaseTool] = [
        FixedResultTool(
            name="knowledge_search",
            description="Search internal enterprise knowledge (deterministic stub).",
            input_schema=common_query_schema,
            summary="Internal procurement policy requires quarterly supplier review and approval.",
            data={"matches": 2, "knowledge_base": "demo-policy-kb"},
            source_type=SourceType.DOCUMENT,
            source_title="Demo Procurement Policy",
            source_url=None,
            timeout_seconds=timeout_seconds,
        ),
        FixedResultTool(
            name="web_search",
            description="Discover public sources (deterministic stub).",
            input_schema=common_query_schema,
            summary=(
                "Public market signals in the demo dataset indicate sustained "
                "AI infrastructure demand."
            ),
            data={"results": 3},
            source_type=SourceType.WEB,
            source_title="Demo Market Research Source",
            source_url="https://example.com/demo-market-research",
            timeout_seconds=timeout_seconds,
        ),
        FixedResultTool(
            name="http_fetch",
            description="Fetch a known URL or API (deterministic stub; no network request).",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            summary="The demo endpoint returned a static quarterly business summary.",
            data={"status_code": 200, "content_type": "text/plain"},
            source_type=SourceType.API,
            source_title="Demo HTTP Response",
            source_url="https://example.com/demo-quarterly-summary",
            timeout_seconds=timeout_seconds,
        ),
        FixedResultTool(
            name="schema_search",
            description="Find relevant database schema and relationships (deterministic stub).",
            input_schema=common_query_schema,
            summary=(
                "Relevant demo tables: fact_purchase_orders joined to dim_suppliers by supplier_id."
            ),
            data={
                "tables": ["fact_purchase_orders", "dim_suppliers"],
                "relationship": "fact_purchase_orders.supplier_id = dim_suppliers.id",
            },
            source_type=SourceType.SQL,
            source_title="Demo Database Catalog",
            source_url=None,
            timeout_seconds=timeout_seconds,
            permission=ToolPermission.MEDIUM,
        ),
        SqlStubTool(
            name="execute_sql",
            description="Execute validated read-only SQL (deterministic stub; no database access).",
            input_schema={
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
            summary="Demo procurement total is $1.24M; the largest supplier share is 46%.",
            data={"columns": ["total_spend", "largest_supplier_share"], "rows": [[1240000, 0.46]]},
            source_type=SourceType.SQL,
            source_title="Demo Read-only SQL Result",
            source_url=None,
            timeout_seconds=timeout_seconds,
            permission=ToolPermission.MEDIUM,
        ),
        FixedResultTool(
            name="python_execute",
            description="Perform deterministic data analysis (fixed stub; no code execution).",
            input_schema={
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
            summary=(
                "Demo calculation: spend increased 12% quarter over quarter "
                "with moderate concentration risk."
            ),
            data={"qoq_change": 0.12, "risk": "moderate"},
            source_type=SourceType.SQL,
            source_title="Demo Analysis Result",
            source_url=None,
            timeout_seconds=timeout_seconds,
            permission=ToolPermission.MEDIUM,
        ),
        FixedResultTool(
            name="browser",
            description="Interactive web fallback (deterministic stub; no browser launched).",
            input_schema={
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
            summary=(
                "Demo browser flow completed login, date filtering, and CSV export "
                "without real interaction."
            ),
            data={"steps": ["login", "filter_last_30_days", "export_csv"]},
            source_type=SourceType.BROWSER,
            source_title="Demo SaaS Dashboard",
            source_url="https://example.com/demo-saas",
            timeout_seconds=timeout_seconds,
            permission=ToolPermission.HIGH,
        ),
        FixedResultTool(
            name="mcp_invoke",
            description="Discover and invoke an external MCP tool (deterministic stub).",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            summary="Demo MCP server exposed a read-only project metadata tool.",
            data={"server": "demo-mcp", "tools": ["get_project_metadata"]},
            source_type=SourceType.MCP,
            source_title="Demo MCP Tool Result",
            source_url=None,
            timeout_seconds=timeout_seconds,
            permission=ToolPermission.LOW,
        ),
    ]
    for tool in tools:
        registry.register(tool)
    return registry
