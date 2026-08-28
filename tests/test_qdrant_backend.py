import json

import httpx
import pytest

from app.knowledge import (
    DeterministicEmbedder,
    KnowledgeDocumentInput,
    KnowledgeService,
    QdrantKnowledgeBackend,
)


@pytest.mark.asyncio
async def test_qdrant_creates_collection_and_upserts_anchored_chunks() -> None:
    requests: list[tuple[str, str, dict | None, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body, request.headers.get("api-key")))
        if request.method == "GET":
            return httpx.Response(404, json={"status": "not found"})
        return httpx.Response(200, json={"result": {"status": "ok"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = QdrantKnowledgeBackend(
            base_url="https://qdrant.test",
            collection="knowledge",
            dimensions=64,
            api_key="qdrant-secret",
            http_client=client,
        )
        service = KnowledgeService(
            backend=backend,
            embedder=DeterministicEmbedder(dimensions=64),
            chunk_size=32,
            chunk_overlap=4,
        )
        indexed = await service.index_document(
            tenant_id="tenant-a",
            document=KnowledgeDocumentInput(
                title="Policy",
                content="A" * 70,
                knowledge_base_id="policy",
                allowed_principal_ids=["alice"],
            ),
        )

    assert indexed.chunk_count == 3
    assert [(method, path) for method, path, _, _ in requests] == [
        ("GET", "/collections/knowledge"),
        ("PUT", "/collections/knowledge"),
        ("PUT", "/collections/knowledge/points"),
    ]
    collection_body = requests[1][2]
    assert collection_body == {"vectors": {"size": 64, "distance": "Cosine"}}
    points = requests[2][2]["points"]
    assert len(points) == 3
    assert all(point["payload"]["tenant_id"] == "tenant-a" for point in points)
    assert all(point["payload"]["allowed_principal_ids"] == ["alice"] for point in points)
    assert all(point["payload"]["char_end"] > point["payload"]["char_start"] for point in points)
    assert all(api_key == "qdrant-secret" for _, _, _, api_key in requests)
