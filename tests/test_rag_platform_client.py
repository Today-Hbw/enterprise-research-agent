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
        "content": "Hello world content.",
        "source_url": None,
        "knowledge_base_id": "kb-1",
        "tenant_id": "tenant-a",
        "score": 0.95,
        "location": {"char_start": 0, "char_end": 18, "chunk_index": 0},
        "metadata": None,
    }
    defaults.update(overrides)
    return defaults


# ── knowledge-base discovery ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_knowledge_bases_maps_versioned_api_response():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/knowledge-bases"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "knowledge_bases": [
                    {
                        "id": "123456",
                        "name": "Employee Benefits Policy",
                        "source": "yuque",
                        "document_count": 3417,
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(
            base_url="http://rag.test", api_key="secret", http_client=client
        )
        knowledge_bases = await rag.list_knowledge_bases()

    assert knowledge_bases == [
        {
            "id": "123456",
            "name": "Employee Benefits Policy",
            "source": "yuque",
            "document_count": 3417,
        }
    ]


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


@pytest.mark.asyncio
async def test_upsert_preserves_public_access_without_principal_allowlist():
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(202, json={"job_id": "job_1", "status": "PENDING"})

    chunk = _chunk(allowed_principal_ids=frozenset(), public=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(base_url="http://rag.test", api_key="k", http_client=client)
        await rag.upsert([chunk])

    assert captured["body"]["access_control"] == {
        "allowed_principal_ids": [],
        "is_public": True,
    }


@pytest.mark.asyncio
async def test_upsert_reconstructs_overlapping_chunks_without_duplicate_text():
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={
                "job_id": "job_1",
                "status": "PENDING",
                "document_id": "doc_1",
                "knowledge_base_id": "kb-1",
                "tenant_id": "tenant-a",
                "created_at": "2026-09-07T00:00:00Z",
                "updated_at": "2026-09-07T00:00:00Z",
                "error": None,
            },
        )

    chunks = [
        _chunk(content="abcdef", char_start=0, char_end=6),
        _chunk(chunk_id="c2", content="defghi", char_start=3, char_end=9),
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(base_url="http://rag.test", api_key="k", http_client=client)
        await rag.upsert(chunks)

    assert captured["body"]["content"] == "abcdefghi"


# ── lifecycle ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_document_sends_idempotency_header():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/v1/knowledge-bases/kb-1/documents/doc_1"
        assert request.url.query == b""
        assert request.headers["idempotency-key"] == "delete-key-123"
        return httpx.Response(
            200,
            json={
                "document_id": "doc_1",
                "knowledge_base_id": "kb-1",
                "tenant_id": "tenant-a",
                "deleted": True,
                "deleted_at": "2026-09-07T00:00:00Z",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(base_url="http://rag.test", api_key="k", http_client=client)
        result = await rag.delete_document(
            "kb-1", "doc_1", idempotency_key="delete-key-123"
        )

    assert result["deleted"] is True


@pytest.mark.asyncio
async def test_get_ingestion_job_uses_versioned_status_endpoint():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/ingestion-jobs/job_1"
        return httpx.Response(
            200,
            json={
                "job_id": "job_1",
                "status": "READY",
                "document_id": "doc_1",
                "knowledge_base_id": "kb-1",
                "tenant_id": "tenant-a",
                "created_at": "2026-09-07T00:00:00Z",
                "updated_at": "2026-09-07T00:00:01Z",
                "error": None,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(base_url="http://rag.test", api_key="k", http_client=client)
        result = await rag.get_ingestion_job("job_1")

    assert result["status"] == "READY"


@pytest.mark.asyncio
async def test_update_document_sends_idempotency_key_in_json_body():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert request.url.path == "/api/v1/knowledge-bases/kb-1/documents/doc_1"
        assert json.loads(request.content) == {
            "title": "Updated",
            "idempotency_key": "update-key-123",
        }
        return httpx.Response(
            202,
            json={
                "job_id": "job_2",
                "status": "PENDING",
                "document_id": "doc_1",
                "knowledge_base_id": "kb-1",
                "tenant_id": "tenant-a",
                "created_at": "2026-09-07T00:00:00Z",
                "updated_at": "2026-09-07T00:00:00Z",
                "error": None,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(base_url="http://rag.test", api_key="k", http_client=client)
        result = await rag.update_document(
            "kb-1",
            "doc_1",
            updates={"title": "Updated"},
            idempotency_key="update-key-123",
        )

    assert result["job_id"] == "job_2"


# ── search_hybrid ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_hybrid_returns_matches():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["query"] == "hello"
        assert body["top_k"] == 5
        assert "score_threshold" not in body
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
    assert m.content == "Hello world content."


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
async def test_search_hybrid_sends_explicitly_configured_score_threshold():
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_search_response([]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(
            base_url="http://rag.test",
            api_key="k",
            score_threshold=0.6,
            http_client=client,
        )
        await rag.search_hybrid(
            query="employee benefits policy",
            vector=[0.1] * 64,
            tenant_id="tenant-a",
            principal_ids=set(),
            knowledge_base_id="123456",
            top_k=3,
            rrf_k=60,
        )

    assert captured["body"]["score_threshold"] == 0.6


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
async def test_platform_error_preserves_stable_error_code_without_credentials():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "code": "access_denied",
                "message": "Knowledge base access denied",
                "request_id": "req-1",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rag = RagPlatformClient(
            base_url="http://rag.test", api_key="top-secret", http_client=client
        )
        with pytest.raises(KnowledgeBackendError) as exc_info:
            await rag.list_knowledge_bases()

    message = str(exc_info.value)
    assert "403" in message
    assert "access_denied" in message
    assert "Knowledge base access denied" in message
    assert "top-secret" not in message


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
        "content": "C",
        "score": 0.5,
        "location": None,
    }
    match = RagPlatformClient._parse_source_item(item)
    assert match.document_id == "doc_1"
    assert match.content == "C"
    assert match.char_start == 0
    assert match.char_end == 0
    assert match.source_url is None


def test_parse_source_item_prefers_document_url_from_metadata():
    item = _source_item(
        source_url="https://www.yuque.com/hfyi1g/4161555",
        metadata={"document_url": "https://www.yuque.com/hfyi1g/4161555/chengdu"},
    )

    match = RagPlatformClient._parse_source_item(item)

    assert match.source_url == "https://www.yuque.com/hfyi1g/4161555/chengdu"
