from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app, settings


def test_conversation_and_run_history_require_an_original_principal(monkeypatch) -> None:
    monkeypatch.setattr(settings, "knowledge_trust_access_headers", True)
    tenant_id = f"tenant-{uuid4().hex}"
    alice_headers = {"X-Tenant-Id": tenant_id, "X-Principal-Ids": "alice,analyst"}
    bob_headers = {"X-Tenant-Id": tenant_id, "X-Principal-Ids": "bob"}
    analyst_headers = {"X-Tenant-Id": tenant_id, "X-Principal-Ids": "carol,analyst"}

    with TestClient(app) as client:
        created = client.post(
            "/api/chat", headers=alice_headers, json={"query": "private research"}
        )
        conversation_id = created.json()["conversation_id"]
        run_id = created.json()["run"]["run_id"]

        denied_conversation = client.get(
            f"/api/conversations/{conversation_id}", headers=bob_headers
        )
        denied_run = client.get(f"/api/runs/{run_id}", headers=bob_headers)
        group_conversation = client.get(
            f"/api/conversations/{conversation_id}", headers=analyst_headers
        )

    assert denied_conversation.status_code == 404
    assert denied_run.status_code == 404
    assert group_conversation.status_code == 200
