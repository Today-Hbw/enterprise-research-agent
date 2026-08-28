from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.main import app, settings


def test_admin_ingestion_to_authorized_chat_citation(monkeypatch) -> None:
    admin_token = "test-admin-token"
    tenant_id = f"tenant-{uuid4().hex}"
    marker = f"controlled-policy-{uuid4().hex}"
    monkeypatch.setattr(settings, "knowledge_admin_token", SecretStr(admin_token))
    monkeypatch.setattr(settings, "knowledge_trust_access_headers", True)
    headers = {
        "X-Tenant-Id": tenant_id,
        "X-Principal-Ids": "alice",
        "X-Knowledge-Admin-Token": admin_token,
    }

    with TestClient(app) as client:
        created = client.post(
            "/api/knowledge/documents",
            headers=headers,
            json={
                "title": "Controlled supplier policy",
                "content": f"{marker} requires a quarterly supplier review.",
                "knowledge_base_id": "policy",
                "allowed_principal_ids": ["alice"],
            },
        )
        chat = client.post(
            "/api/chat",
            headers=headers,
            json={"query": marker},
        )
        tools = client.get("/api/tools")

    assert created.status_code == 201
    assert created.json()["chunk_count"] == 1
    assert chat.status_code == 200
    knowledge_sources = [
        source
        for source in chat.json()["run"]["sources"]
        if source["title"] == "Controlled supplier policy"
    ]
    assert len(knowledge_sources) == 1
    assert knowledge_sources[0]["chunk_id"]
    assert marker in knowledge_sources[0]["content_snippet"]
    tool_catalog = {tool["name"]: tool for tool in tools.json()}
    assert tool_catalog["knowledge_search"]["is_stub"] is False
    assert tool_catalog["mcp_invoke"]["is_stub"] is False
    assert tool_catalog["web_search"]["is_stub"] is True
    assert tool_catalog["browser"]["is_stub"] is True


def test_ingestion_rejects_missing_or_invalid_admin_token(monkeypatch) -> None:
    monkeypatch.setattr(settings, "knowledge_admin_token", SecretStr("expected-token"))

    with TestClient(app) as client:
        missing = client.post(
            "/api/knowledge/documents",
            headers={"X-Tenant-Id": "tenant-a"},
            json={"title": "Policy", "content": "Secret policy content."},
        )
        invalid = client.post(
            "/api/knowledge/documents",
            headers={
                "X-Tenant-Id": "tenant-a",
                "X-Knowledge-Admin-Token": "wrong-token",
            },
            json={"title": "Policy", "content": "Secret policy content."},
        )

    assert missing.status_code == 403
    assert invalid.status_code == 403
