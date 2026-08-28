import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app.knowledge import KnowledgeDocumentInput
from app.main import app, settings


def test_knowledge_source_url_only_allows_http_or_https() -> None:
    with pytest.raises(ValidationError):
        KnowledgeDocumentInput(
            title="Unsafe source",
            content="Content",
            source_url="javascript:alert(1)",
        )


def test_empty_admin_secret_keeps_ingestion_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "knowledge_admin_token", SecretStr(""))

    with TestClient(app) as client:
        response = client.post(
            "/api/knowledge/documents",
            json={"title": "Policy", "content": "Secret policy content."},
        )

    assert response.status_code == 503
