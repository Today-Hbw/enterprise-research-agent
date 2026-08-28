from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_and_tool_catalog() -> None:
    health = client.get("/api/health")
    tools = client.get("/api/tools")

    assert health.status_code == 200
    assert health.json()["demo_mode"] is True
    assert tools.status_code == 200
    assert len(tools.json()) == 8


def test_chat_api_returns_completed_traceable_run() -> None:
    response = client.post("/api/chat", json={"query": "分析市场和采购数据"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["status"] == "completed"
    assert payload["run"]["trace"]
    assert payload["run"]["sources"]


def test_stream_endpoint_uses_sse_protocol() -> None:
    with client.stream("POST", "/api/chat/stream", json={"query": "研究公开市场"}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: run_started" in body
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
    assert events.json()[-1]["event"] == "run_completed"
