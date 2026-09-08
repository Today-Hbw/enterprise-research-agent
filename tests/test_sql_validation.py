import json
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from app.models import ToolCall
from app.tools.sql import ExecuteSqlTool, SqlValidationError, validate_readonly_sql


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT id, name FROM public.suppliers",
        "SELECT id FROM public.suppliers UNION SELECT id FROM public.archived_suppliers",
        "WITH recent AS (SELECT id FROM public.suppliers) SELECT * FROM recent",
    ],
)
def test_validator_accepts_read_only_queries(statement: str) -> None:
    result = validate_readonly_sql(statement, frozenset({"public"}))

    assert result is not None


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM public.suppliers",
        "SELECT * INTO public.copy_of_suppliers FROM public.suppliers",
        "SELECT * FROM private.suppliers",
        "SELECT * FROM public.suppliers; SELECT * FROM public.suppliers",
        "SELECT * FROM public.suppliers FOR UPDATE",
    ],
)
def test_validator_rejects_mutations_multi_statement_and_unapproved_schema(statement: str) -> None:
    with pytest.raises(SqlValidationError):
        validate_readonly_sql(statement, frozenset({"public"}))


def test_validator_requires_public_for_unqualified_tables() -> None:
    with pytest.raises(SqlValidationError, match="Unqualified"):
        validate_readonly_sql("SELECT * FROM suppliers", frozenset({"analytics"}))


@pytest.mark.asyncio
async def test_execute_sql_converts_database_values_to_json_primitives() -> None:
    class Backend:
        max_rows = 10

        async def execute_readonly(self, statement, access_context):
            return (
                ["created_at", "due_on", "amount"],
                [
                    [
                        datetime(2026, 9, 8, 12, 30, tzinfo=UTC),
                        date(2026, 9, 9),
                        Decimal("1.25"),
                    ]
                ],
                False,
            )

    result = await ExecuteSqlTool(Backend(), timeout_seconds=1).execute(
        ToolCall(name="execute_sql", arguments={"sql": "SELECT 1"})
    )

    json.dumps(result.data)
    assert result.data["rows"] == [["2026-09-08T12:30:00Z", "2026-09-09", "1.25"]]
