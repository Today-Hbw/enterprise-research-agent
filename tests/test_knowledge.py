import json

import httpx
import pytest

from app.knowledge import (
    DeterministicEmbedder,
    InMemoryKnowledgeBackend,
    KnowledgeDocumentInput,
    KnowledgeService,
    QdrantKnowledgeBackend,
)
from app.models import AccessContext, ToolCall
from app.tools.knowledge import KnowledgeSearchTool


def build_service() -> KnowledgeService:
    return KnowledgeService(
        backend=InMemoryKnowledgeBackend(),
        embedder=DeterministicEmbedder(dimensions=64),
        chunk_size=80,
        chunk_overlap=10,
    )


@pytest.mark.asyncio
async def test_knowledge_search_enforces_tenant_and_principal_filters() -> None:
    service = build_service()
    await service.index_document(
        tenant_id="tenant-a",
        document=KnowledgeDocumentInput(
            title="Private procurement policy",
            content="Quarterly supplier review is mandatory for strategic procurement.",
            knowledge_base_id="policy",
            allowed_principal_ids=["alice"],
        ),
    )
    await service.index_document(
        tenant_id="tenant-b",
        document=KnowledgeDocumentInput(
            title="Other tenant policy",
            content="Quarterly supplier review is optional in this unrelated tenant.",
            knowledge_base_id="policy",
            public=True,
        ),
    )
    tool = KnowledgeSearchTool(service=service, timeout_seconds=1)
    call = ToolCall(
        name="knowledge_search",
        arguments={"query": "quarterly supplier review", "knowledge_base_id": "policy"},
    )

    allowed = await tool.execute(call, AccessContext(tenant_id="tenant-a", principal_ids={"alice"}))
    denied = await tool.execute(
        call, AccessContext(tenant_id="tenant-a", principal_ids={"mallory"})
    )

    assert allowed.success is True
    assert [source.title for source in allowed.sources] == ["Private procurement policy"]
    assert denied.success is True
    assert denied.sources == []
    assert denied.data["matches"] == []


@pytest.mark.asyncio
async def test_knowledge_citation_span_round_trips_to_original_content() -> None:
    service = build_service()
    content = "Intro. The approved supplier threshold is 80 percent. Closing note."
    indexed = await service.index_document(
        tenant_id="tenant-a",
        document=KnowledgeDocumentInput(
            title="Supplier controls",
            content=content,
            knowledge_base_id="policy",
            public=True,
        ),
    )
    tool = KnowledgeSearchTool(service=service, timeout_seconds=1)

    result = await tool.execute(
        ToolCall(
            name="knowledge_search",
            arguments={"query": "approved supplier threshold", "top_k": 1},
        ),
        AccessContext(tenant_id="tenant-a", principal_ids=set()),
    )

    assert result.success is True
    source = result.sources[0]
    assert source.document_id == indexed.document_id
    assert source.chunk_id
    assert source.char_start is not None
    assert source.char_end is not None
    assert content[source.char_start : source.char_end] == source.content_snippet


@pytest.mark.asyncio
async def test_qdrant_query_always_contains_server_side_access_filter() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"result": {"status": "green"}})
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"result": {"points": []}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = QdrantKnowledgeBackend(
            base_url="https://qdrant.test",
            collection="knowledge",
            dimensions=64,
            http_client=client,
        )
        await backend.search(
            vector=[0.0] * 64,
            tenant_id="tenant-a",
            principal_ids={"alice", "analyst"},
            knowledge_base_id="policy",
            top_k=3,
        )

    query = requests[-1]
    assert query["limit"] == 3
    must = query["filter"]["must"]
    assert {condition.get("key") for condition in must if "key" in condition} == {
        "tenant_id",
        "knowledge_base_id",
    }
    access_filter = next(condition for condition in must if "should" in condition)
    assert {condition["key"] for condition in access_filter["should"]} == {
        "public",
        "allowed_principal_ids",
    }
