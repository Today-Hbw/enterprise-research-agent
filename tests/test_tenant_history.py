from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app, settings


def test_conversation_and_run_history_are_tenant_scoped(monkeypatch) -> None:
    monkeypatch.setattr(settings, "knowledge_trust_access_headers", True)
    tenant_a = f"tenant-a-{uuid4().hex}"
    tenant_b = f"tenant-b-{uuid4().hex}"
    headers_a = {"X-Tenant-Id": tenant_a, "X-Principal-Ids": "alice"}
    headers_b = {"X-Tenant-Id": tenant_b, "X-Principal-Ids": "bob"}

    with TestClient(app) as client:
        created = client.post("/api/chat", headers=headers_a, json={"query": "market research"})
        conversation_id = created.json()["conversation_id"]
        run_id = created.json()["run"]["run_id"]

        own_conversation = client.get(f"/api/conversations/{conversation_id}", headers=headers_a)
        foreign_conversation = client.get(
            f"/api/conversations/{conversation_id}", headers=headers_b
        )
        foreign_run = client.get(f"/api/runs/{run_id}", headers=headers_b)
        foreign_resume = client.post(
            "/api/chat/stream",
            headers=headers_b,
            json={"query": "continue", "conversation_id": conversation_id},
        )
        tenant_b_conversations = client.get("/api/conversations", headers=headers_b)
        tenant_a_dashboard = client.get("/api/runs", headers=headers_a)
        tenant_b_dashboard = client.get("/api/runs", headers=headers_b)

    assert created.status_code == 200
    assert own_conversation.status_code == 200
    assert foreign_conversation.status_code == 404
    assert foreign_run.status_code == 404
    assert foreign_resume.status_code == 404
    assert all(
        conversation["conversation_id"] != conversation_id
        for conversation in tenant_b_conversations.json()
    )
    assert any(run["run_id"] == run_id for run in tenant_a_dashboard.json()["recent_runs"])
    assert all(run["run_id"] != run_id for run in tenant_b_dashboard.json()["recent_runs"])
