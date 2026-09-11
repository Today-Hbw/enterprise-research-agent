import logging

from app.logging_utils import REDACTED, RedactingFormatter, configure_logging, log_json


def test_log_json_redacts_nested_secrets_and_sensitive_sql_columns() -> None:
    rendered = log_json(
        {
            "api_key": "api-key-value",
            "input_tokens": 42,
            "headers": {"Authorization": "Bearer bearer-value"},
            "dsn": "postgresql://reader:database-password@db:5432/app",
            "columns": ["id", "password", "access_token"],
            "rows": [[1, "row-password", "row-token"]],
            "message": "known-secret appeared in output",
        },
        secrets=("known-secret",),
    )

    assert rendered.count(REDACTED) >= 6
    assert "api-key-value" not in rendered
    assert "bearer-value" not in rendered
    assert "database-password" not in rendered
    assert "row-password" not in rendered
    assert "row-token" not in rendered
    assert "known-secret" not in rendered
    assert '"input_tokens":42' in rendered


def test_redacting_formatter_covers_messages_and_exception_text() -> None:
    formatter = RedactingFormatter("%(message)s", secrets=("configured-secret",))
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=(
            "Authorization: Bearer abc123 "
            "postgresql://reader:db-pass@localhost/app configured-secret"
        ),
        args=(),
        exc_info=None,
    )

    rendered = formatter.format(record)

    assert "abc123" not in rendered
    assert "db-pass" not in rendered
    assert "configured-secret" not in rendered
    assert rendered.count(REDACTED) == 3


def test_configured_secrets_are_redacted_before_payload_truncation() -> None:
    try:
        configure_logging(level="INFO", secrets=("configured-runtime-secret",))
        rendered = log_json(
            {"message": f"{'x' * 10}configured-runtime-secret{'y' * 100}"},
            max_chars=40,
        )

        assert "configured-runtime-secret" not in rendered
        assert REDACTED in rendered
        assert "TRUNCATED" in rendered
    finally:
        configure_logging(level="WARNING", secrets=())
