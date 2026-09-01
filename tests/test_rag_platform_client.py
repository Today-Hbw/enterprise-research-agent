"""RagPlatformClient 单元测试。"""

from __future__ import annotations

import json

import httpx
import pytest

from app.knowledge import KnowledgeBackendError, KnowledgeChunk
from app.rag_platform_client import RagPlatformClient


def _chunk(**overrides: object) -> KnowledgeChunk:
    defaults = dict(
        point_id="p1",
        document_id="ext-doc-1",
        chunk_id="c1",
        tenant_id="tenant-a",
        knowledge_base_id="kb-1",
        title="Test Document",
        content="Hello world content.",
        char_start=0,
        char_end=18,
        allowed_principal_ids=frozenset({"user-1"}),
        public=False,
        source_url=None,
        metadata={"category": "test"},
        vector=[0.1] * 64,
    )
    defaults.update(overrides)
    return KnowledgeChunk(**defaults)


def _search_response(results: list[dict]) -> dict:
    return {
        "success": True,
        "query": "hello",
        "results": results,
        "total": len(results),
        "request_id": "req-1",
        "trace_id": "trace-1",
    }


def _source_item(**overrides: object) -> dict:
    defaults = {
        "document_id": "doc_abc123",
        "chunk_id": "chk_def456",
        "title": "Test Doc",
        "content_snippet": "Hello world content.",
        "source_url": None,
        "knowledge_base_id": "kb-1",
        "tenant_id": "tenant-a",
        "score": 0.95,
        "location": {"char_start": 0, "char_end": 18, "chunk_index": 0},
        "metadata": None,
    }
    defaults.update(overrides)
    return defaults


# ── upsert ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_sends_ingest_request():
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(202, json={
            "job_id": "job_1",
            "status": "PENDING",
            "document_id": "doc_abc",
            "knowledge_base_id": "kb-1",
            "tenant_id": "tenant-a",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "error": None,
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(base_url="http://rag.test", api_key="secret", http_client=client)
        await rag.upsert([_chunk()])

    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v1/knowledge-bases/kb-1/documents"
    assert captured["auth"] == "Bearer secret"
    body = captured["body"]
    assert body["external_document_id"] == "ext-doc-1"
    assert body["title"] == "Test Document"
    assert "Hello world" in body["content"]
    assert body["idempotency_key"].startswith("agent_")
    assert body["access_control"]["allowed_principal_ids"] == ["user-1"]


@pytest.mark.asyncio
async def test_upsert_empty_chunks_is_noop():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Should not be called")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(base_url="http://rag.test", http_client=client)
        await rag.upsert([])


# ── search_hybrid ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_hybrid_returns_matches():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["query"] == "hello"
        assert body["top_k"] == 5
        return httpx.Response(200, json=_search_response([_source_item()]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(base_url="http://rag.test", api_key="k", http_client=client)
        matches = await rag.search_hybrid(
            query="hello",
            vector=[0.1] * 64,
            tenant_id="tenant-a",
            principal_ids={"user-1"},
            knowledge_base_id="kb-1",
            top_k=5,
            rrf_k=60,
        )

    assert len(matches) == 1
    m = matches[0]
    assert m.document_id == "doc_abc123"
    assert m.chunk_id == "chk_def456"
    assert m.score == 0.95
    assert m.char_start == 0
    assert m.char_end == 18


@pytest.mark.asyncio
async def test_search_hybrid_with_metadata_filters():
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_search_response([]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(base_url="http://rag.test", api_key="k", http_client=client)
        await rag.search_hybrid(
            query="test",
            vector=[0.1] * 64,
            tenant_id="tenant-a",
            principal_ids=set(),
            knowledge_base_id="kb-1",
            top_k=10,
            rrf_k=60,
            metadata_filters={"category": "policy"},
        )

    filters = captured["body"]["filters"]
    assert filters == [{"key": "category", "equals": "policy"}]


@pytest.mark.asyncio
async def test_search_hybrid_requires_knowledge_base_id():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: None)) as client:
        rag = RagPlatformClient(base_url="http://rag.test", http_client=client)
        with pytest.raises(KnowledgeBackendError, match="knowledge_base_id"):
            await rag.search_hybrid(
                query="test",
                vector=[0.1] * 64,
                tenant_id="t",
                principal_ids=set(),
                knowledge_base_id=None,
                top_k=5,
                rrf_k=60,
            )


@pytest.mark.asyncio
async def test_search_hybrid_error_response():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "success": False,
            "query": "test",
            "results": [],
            "total": 0,
            "error_code": "dependency_unavailable",
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(base_url="http://rag.test", api_key="k", http_client=client)
        with pytest.raises(KnowledgeBackendError, match="dependency_unavailable"):
            await rag.search_hybrid(
                query="test",
                vector=[0.1] * 64,
                tenant_id="t",
                principal_ids=set(),
                knowledge_base_id="kb",
                top_k=5,
                rrf_k=60,
            )


# ── search (vector-only) ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_vector_only_returns_empty():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: None)) as client:
        rag = RagPlatformClient(base_url="http://rag.test", http_client=client)
        result = await rag.search(
            vector=[0.1] * 64,
            tenant_id="t",
            principal_ids=set(),
            knowledge_base_id="kb",
            top_k=5,
        )
    assert result == []


# ── auth / error handling ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unauthorized_raises():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"code": "auth_missing"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(base_url="http://rag.test", http_client=client)
        with pytest.raises(KnowledgeBackendError, match="401"):
            await rag.search_hybrid(
                query="test",
                vector=[0.1] * 64,
                tenant_id="t",
                principal_ids=set(),
                knowledge_base_id="kb",
                top_k=5,
                rrf_k=60,
            )


@pytest.mark.asyncio
async def test_supports_hybrid_search():
    rag = RagPlatformClient(base_url="http://rag.test")
    assert rag.supports_hybrid_search is True


# ── DTO mapping ────────────────────────────────────────────────────────


def test_parse_source_item_minimal():
    item = {
        "document_id": "doc_1",
        "chunk_id": "chk_1",
        "knowledge_base_id": "kb",
        "title": "T",
        "content_snippet": "C",
        "score": 0.5,
        "location": {},
    }
    match = RagPlatformClient._parse_source_item(item)
    assert match.document_id == "doc_1"
    assert match.content == "C"
    assert match.char_start == 0
    assert match.char_end == 0
    assert match.source_url is None
