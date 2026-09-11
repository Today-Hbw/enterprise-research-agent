from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, SecretStr

REDACTED = "[REDACTED]"
_configured_secrets: tuple[str, ...] = ()
_SENSITIVE_EXACT_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "credentials",
    "dsn",
    "password",
    "proxy_authorization",
    "secret",
    "set_cookie",
    "token",
}
_SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_cookie",
    "_credentials",
    "_dsn",
    "_password",
    "_secret",
    "_token",
)
_AUTH_PATTERN = re.compile(r"(?i)\b(Bearer|Basic)\s+[^\s,;]+")
_DSN_PATTERN = re.compile(
    r"(?i)(?P<scheme>(?:postgres(?:ql)?|redis|https?)://)"
    r"(?P<username>[^:/@\s]+):(?P<password>[^@\s]+)@"
)
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?P<prefix>\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"cookie|password|secret)\b\s*[=:]\s*[\"']?)"
    r"(?P<value>[^\s,;\"'}&]+)"
)


def is_sensitive_key(value: object) -> bool:
    normalized = str(value).strip().lower().replace("-", "_")
    return normalized in _SENSITIVE_EXACT_KEYS or normalized.endswith(
        _SENSITIVE_KEY_SUFFIXES
    )


def redact_text(value: str, *, secrets: Iterable[str] = ()) -> str:
    redacted = value
    all_secrets = {*_configured_secrets, *secrets}
    for secret in sorted({item for item in all_secrets if len(item) >= 4}, key=len, reverse=True):
        redacted = redacted.replace(secret, REDACTED)
    redacted = _AUTH_PATTERN.sub(lambda match: f"{match.group(1)} {REDACTED}", redacted)
    redacted = _DSN_PATTERN.sub(
        lambda match: (
            f"{match.group('scheme')}{match.group('username')}:{REDACTED}@"
        ),
        redacted,
    )
    return _ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group('prefix')}{REDACTED}", redacted
    )


def sanitize_for_logging(value: Any, *, secrets: Iterable[str] = ()) -> Any:
    secret_values = tuple(secrets)
    if isinstance(value, SecretStr):
        return REDACTED
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        columns = value.get("columns")
        sensitive_column_indexes: set[int] = set()
        if isinstance(columns, Sequence) and not isinstance(columns, (str, bytes)):
            sensitive_column_indexes = {
                index for index, column in enumerate(columns) if is_sensitive_key(column)
            }
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if is_sensitive_key(key_text):
                sanitized[key_text] = REDACTED
            elif key_text == "rows" and sensitive_column_indexes and isinstance(item, Sequence):
                sanitized[key_text] = [
                    [
                        REDACTED
                        if index in sensitive_column_indexes
                        else sanitize_for_logging(cell, secrets=secret_values)
                        for index, cell in enumerate(row)
                    ]
                    if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
                    else sanitize_for_logging(row, secrets=secret_values)
                    for row in item
                ]
            else:
                sanitized[key_text] = sanitize_for_logging(item, secrets=secret_values)
        return sanitized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_for_logging(item, secrets=secret_values) for item in value]
    if isinstance(value, str):
        return redact_text(value, secrets=secret_values)
    return value


def log_json(value: Any, *, secrets: Iterable[str] = (), max_chars: int = 65_536) -> str:
    rendered = json.dumps(
        sanitize_for_logging(value, secrets=secrets),
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    if len(rendered) <= max_chars:
        return rendered
    omitted = len(rendered) - max_chars
    return f"{rendered[:max_chars]}...[TRUNCATED {omitted} chars]"


def settings_secret_values(settings: BaseModel) -> tuple[str, ...]:
    values = []
    for field_name in type(settings).model_fields:
        value = getattr(settings, field_name, None)
        if isinstance(value, SecretStr):
            secret = value.get_secret_value()
            if secret:
                values.append(secret)
    return tuple(values)


class RedactingFormatter(logging.Formatter):
    def __init__(self, *args: Any, secrets: Iterable[str] = (), **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._secrets = tuple(secrets)

    def format(self, record: logging.LogRecord) -> str:
        return redact_text(super().format(record), secrets=self._secrets)


def configure_logging(*, level: str, secrets: Iterable[str] = ()) -> None:
    global _configured_secrets
    _configured_secrets = tuple(secret for secret in secrets if secret)
    handler = logging.StreamHandler()
    handler.setFormatter(
        RedactingFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            secrets=secrets,
        )
    )
    logging.basicConfig(
        level=getattr(logging, level),
        handlers=[handler],
        force=True,
    )
