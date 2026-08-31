import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import app

client = TestClient(app)


def test_health_and_tool_catalog() -> None:
    health = client.get("/api/health")
    tools = client.get("/api/tools")

    assert health.status_code == 200
    assert health.json()["demo_mode"] is True
    assert health.json()["knowledge_url_import_enabled"] is False
    assert tools.status_code == 200
    assert len(tools.json()) == 8


def test_chat_api_returns_completed_traceable_run() -> None:
    response = client.post("/api/chat", json={"query": "分析市场和采购数据"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["status"] == "completed"
    assert payload["run"]["trace"]
    assert payload["run"]["sources"]
    assert payload["run"]["plan"]
    assert payload["run"]["plan"][-1]["title"] == "Synthesize the final answer"
    assert all(step["status"] == "completed" for step in payload["run"]["plan"])


def test_stream_endpoint_uses_sse_protocol() -> None:
    with client.stream("POST", "/api/chat/stream", json={"query": "研究公开市场"}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: run_started" in body
    assert "event: plan_created" in body
    assert "event: plan_step_updated" in body
    assert "event: tool_completed" in body
    assert "event: run_completed" in body


def test_unknown_conversation_returns_404() -> None:
    response = client.post(
        "/api/chat/stream", json={"query": "hello", "conversation_id": "conv_missing"}
    )

    assert response.status_code == 404


def test_run_events_can_be_replayed_after_a_sequence() -> None:
    created = client.post("/api/chat", json={"query": "research market"}).json()
    events = client.get(f"/api/runs/{created['run']['run_id']}/events?after_sequence=1")

    assert events.status_code == 200
    assert events.json()[0]["sequence"] == 2
    assert any(event["event"] == "plan_created" for event in events.json())
    assert any(event["event"] == "plan_step_updated" for event in events.json())
    assert events.json()[-1]["event"] == "run_completed"


def test_run_dashboard_aggregates_recent_scoped_runs() -> None:
    created = client.post("/api/chat", json={"query": "dashboard metrics"}).json()["run"]
    response = client.get("/api/runs?limit=20")

    assert response.status_code == 200
    dashboard = response.json()
    assert dashboard["total_runs"] >= 1
    assert dashboard["status_counts"]["completed"] >= 1
    assert dashboard["total_tokens"] >= created["metrics"]["token_usage"]
    assert any(run["run_id"] == created["run_id"] for run in dashboard["recent_runs"])
    summary = next(run for run in dashboard["recent_runs"] if run["run_id"] == created["run_id"])
    assert "trace" not in summary
    assert summary["budget"] == {"token_limit": None, "cost_limit": None}


def test_run_dashboard_rejects_out_of_range_limit() -> None:
    assert client.get("/api/runs?limit=0").status_code == 400
    assert client.get("/api/runs?limit=101").status_code == 400


def test_cost_budget_requires_rates_and_rejects_partial_pricing() -> None:
    with pytest.raises(ValidationError, match="requires configured"):
        Settings(run_cost_budget_usd=1)

    with pytest.raises(ValidationError, match="must be configured together"):
        Settings(llm_input_cost_per_million_tokens=1)
