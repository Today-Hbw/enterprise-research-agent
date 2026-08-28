from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg import sql
from sqlglot import exp, parse

from app.models import AccessContext, Source, SourceType, ToolCall, ToolPermission, ToolResult
from app.tools.base import BaseTool


class SqlValidationError(ValueError):
    """Raised when SQL violates the read-only query policy."""


@dataclass(frozen=True)
class SchemaTable:
    schema: str
    name: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class SqlAuditEvent:
    statement_hash: str
    row_count: int
    truncated: bool
    tenant_id: str | None


@dataclass
class PostgresBackend:
    dsn: str
    allowed_schemas: frozenset[str]
    query_timeout_ms: int
    max_rows: int
    audit_events: list[SqlAuditEvent] = field(default_factory=list)

    async def search_schema(self, query: str) -> list[SchemaTable]:
        terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9_]+", query)}
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT table_schema, table_name, column_name
                    FROM information_schema.columns
                    WHERE table_schema = ANY(%s)
                    ORDER BY table_schema, table_name, ordinal_position
                    """,
                    [list(self.allowed_schemas)],
                )
                rows = await cursor.fetchall()
        tables: dict[tuple[str, str], list[str]] = {}
        for schema_name, table_name, column_name in rows:
            tables.setdefault((schema_name, table_name), []).append(column_name)
        results = [
            SchemaTable(schema, name, tuple(columns)) for (schema, name), columns in tables.items()
        ]
        if not terms:
            return results
        matched = [
            table
            for table in results
            if terms.intersection(
                {table.schema.lower(), table.name.lower(), *(c.lower() for c in table.columns)}
            )
        ]
        return matched or results

    async def execute_readonly(
        self, statement: str, access_context: AccessContext | None
    ) -> tuple[list[str], list[list[Any]], bool]:
        parsed = validate_readonly_sql(statement, self.allowed_schemas)
        statement = parsed.sql(dialect="postgres")
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            async with connection.transaction():
                await connection.execute("SET TRANSACTION READ ONLY")
                await connection.execute(
                    sql.SQL("SET LOCAL statement_timeout = {} ").format(
                        sql.Literal(self.query_timeout_ms)
                    )
                )
                async with connection.cursor() as cursor:
                    await cursor.execute(statement)
                    columns = [item.name for item in cursor.description or []]
                    rows = [list(row) for row in await cursor.fetchmany(self.max_rows + 1)]
        truncated = len(rows) > self.max_rows
        rows = rows[: self.max_rows]
        self.audit_events.append(
            SqlAuditEvent(
                statement_hash=hashlib.sha256(statement.encode()).hexdigest()[:16],
                row_count=len(rows),
                truncated=truncated,
                tenant_id=access_context.tenant_id if access_context else None,
            )
        )
        return columns, rows, truncated


def validate_readonly_sql(statement: str, allowed_schemas: frozenset[str]) -> exp.Expression:
    if not statement.strip():
        raise SqlValidationError("SQL statement is empty")
    try:
        statements = parse(statement, read="postgres")
    except Exception as exc:
        raise SqlValidationError("SQL parser rejected the statement") from exc
    if len(statements) != 1 or statements[0] is None:
        raise SqlValidationError("Exactly one SQL statement is required")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise SqlValidationError("Only SELECT queries and set operations are allowed")
    forbidden = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Merge,
        exp.Create,
        exp.Drop,
        exp.Alter,
        exp.Command,
    )
    if tree.find(forbidden):
        raise SqlValidationError("SQL contains a forbidden mutation or command")
    if tree.args.get("into") is not None or tree.find(exp.Into):
        raise SqlValidationError("SELECT INTO is not allowed")
    if tree.find(exp.Lock):
        raise SqlValidationError("Locking clauses are not allowed")
    for table in tree.find_all(exp.Table):
        database = table.args.get("db")
        if database is not None and database.name not in allowed_schemas:
            raise SqlValidationError(f"Schema is not approved: {database.name}")
        if database is None and "public" not in allowed_schemas:
            raise SqlValidationError("Unqualified table names require public schema access")
    return tree


class SchemaSearchTool(BaseTool):
    name = "schema_search"
    description = "Search approved PostgreSQL schema metadata."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 1}},
        "required": ["query"],
    }
    permission = ToolPermission.MEDIUM
    is_stub = False

    def __init__(self, backend: PostgresBackend, timeout_seconds: float) -> None:
        super().__init__(timeout_seconds)
        self.backend = backend

    async def execute(
        self, call: ToolCall, access_context: AccessContext | None = None
    ) -> ToolResult:
        del access_context
        query = str(call.arguments.get("query", "")).strip()
        if not query:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="Schema query is required.",
                error="Missing query",
            )
        tables = await self.backend.search_schema(query)
        data = {
            "tables": [
                {"schema": t.schema, "name": t.name, "columns": list(t.columns)} for t in tables
            ]
        }
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            success=True,
            summary=f"Found {len(tables)} approved schema tables.",
            data=data,
            sources=[
                Source(
                    source_type=SourceType.SQL,
                    title="PostgreSQL schema catalog",
                    content_snippet=f"{len(tables)} approved tables retrieved.",
                )
            ],
        )


class ExecuteSqlTool(BaseTool):
    name = "execute_sql"
    description = (
        "Execute one AST-validated read-only SQL query against approved PostgreSQL schemas."
    )
    input_schema = {
        "type": "object",
        "properties": {"sql": {"type": "string", "minLength": 1}},
        "required": ["sql"],
    }
    permission = ToolPermission.MEDIUM
    is_stub = False

    def __init__(self, backend: PostgresBackend, timeout_seconds: float) -> None:
        super().__init__(timeout_seconds)
        self.backend = backend

    async def execute(
        self, call: ToolCall, access_context: AccessContext | None = None
    ) -> ToolResult:
        statement = str(call.arguments.get("sql", ""))
        try:
            columns, rows, truncated = await self.backend.execute_readonly(
                statement, access_context
            )
        except SqlValidationError as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="SQL validator rejected the statement.",
                error=str(exc),
            )
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            success=True,
            summary=f"Query returned {len(rows)} row(s){' (truncated)' if truncated else ''}.",
            data={
                "columns": columns,
                "rows": rows,
                "truncated": truncated,
                "max_rows": self.backend.max_rows,
            },
            sources=[
                Source(
                    source_type=SourceType.SQL,
                    title="Read-only PostgreSQL query result",
                    content_snippet=f"{len(rows)} row(s) returned from approved schema(s).",
                )
            ],
        )
