import pytest

from app.tools.sql import SqlValidationError, validate_readonly_sql


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
