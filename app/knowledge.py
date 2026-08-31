from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx
from pydantic import AnyHttpUrl, BaseModel, Field, StringConstraints

PrincipalId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


class KnowledgeBackendError(RuntimeError):
    """Raised when a knowledge backend cannot satisfy its storage contract."""


class KnowledgeDocumentInput(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=2_000_000)
    knowledge_base_id: str = Field(default="default", min_length=1, max_length=128)
    allowed_principal_ids: list[PrincipalId] = Field(default_factory=list, max_length=100)
    public: bool = False
    source_url: AnyHttpUrl | None = None
    metadata: dict[str, str] = Field(default_factory=dict, max_length=20)


class IndexedKnowledgeDocument(BaseModel):
    document_id: str
    tenant_id: str
    knowledge_base_id: str
    chunk_count: int


@dataclass(frozen=True)
class KnowledgeChunk:
    point_id: str
    document_id: str
    chunk_id: str
    tenant_id: str
    knowledge_base_id: str
    title: str
    content: str
    char_start: int
    char_end: int
    allowed_principal_ids: frozenset[str]
    public: bool
    source_url: str | None

    metadata: dict[str, str]
    vector: list[float]

    def payload(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "tenant_id": self.tenant_id,
            "knowledge_base_id": self.knowledge_base_id,
            "title": self.title,
            "content": self.content,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "allowed_principal_ids": sorted(self.allowed_principal_ids),
            "public": self.public,
            "source_url": self.source_url,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class KnowledgeMatch:
    document_id: str
    chunk_id: str
    knowledge_base_id: str
    title: str
    content: str
    char_start: int
    char_end: int
    score: float
    source_url: str | None


class DeterministicEmbedder:
    """Dependency-free feature hashing for offline development and protocol tests."""

    _token_pattern = re.compile(r"[a-z0-9_]+|[\u3400-\u9fff]")

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 16:
            raise ValueError("Embedding dimensions must be at least 16")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._token_pattern.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            return [value / norm for value in vector]
        return vector


class TokenOverlapReranker:
    """Deterministic local reranker for development and offline deployments."""

    _token_pattern = DeterministicEmbedder._token_pattern

    def rerank(self, query: str, matches: list[KnowledgeMatch]) -> list[KnowledgeMatch]:
        query_tokens = set(self._token_pattern.findall(query.lower()))
        if not query_tokens:
            return matches
        reranked = []
        for match in matches:
            title_tokens = set(self._token_pattern.findall(match.title.lower()))
            content_tokens = set(self._token_pattern.findall(match.content.lower()))
            overlap = sum(
                (2 if token in title_tokens else 0) + (1 if token in content_tokens else 0)
                for token in query_tokens
            )
            reranked.append(replace(match, score=overlap + (match.score / 1_000)))
        reranked.sort(key=lambda match: (-match.score, match.document_id, match.chunk_id))
        return reranked


class KnowledgeBackend(ABC):
    @abstractmethod
    async def upsert(self, chunks: list[KnowledgeChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        *,
        vector: list[float],
        tenant_id: str,
        principal_ids: set[str],
        knowledge_base_id: str | None,
        top_k: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[KnowledgeMatch]:
        raise NotImplementedError

    @property
    def supports_hybrid_search(self) -> bool:
        return False

    async def search_hybrid(
        self,
        *,
        query: str,
        vector: list[float],
        tenant_id: str,
        principal_ids: set[str],
        knowledge_base_id: str | None,
        top_k: int,
        rrf_k: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[KnowledgeMatch]:
        raise KnowledgeBackendError("Hybrid search is not supported by this backend")

    async def aclose(self) -> None:
        return None


class InMemoryKnowledgeBackend(KnowledgeBackend):
    _token_pattern = DeterministicEmbedder._token_pattern

    def __init__(self) -> None:
        self._chunks: dict[str, KnowledgeChunk] = {}

    @property
    def supports_hybrid_search(self) -> bool:
        return True

    async def upsert(self, chunks: list[KnowledgeChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.point_id] = chunk

    async def search(
        self,
        *,
        vector: list[float],
        tenant_id: str,
        principal_ids: set[str],
        knowledge_base_id: str | None,
        top_k: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[KnowledgeMatch]:
        candidates = self._authorized_matches(
            vector=vector,
            tenant_id=tenant_id,
            principal_ids=principal_ids,
            knowledge_base_id=knowledge_base_id,
            metadata_filters=metadata_filters,
        )
        candidates.sort(key=lambda match: (-match.score, match.document_id, match.chunk_id))
        return candidates[:top_k]

    async def search_hybrid(
        self,
        *,
        query: str,
        vector: list[float],
        tenant_id: str,
        principal_ids: set[str],
        knowledge_base_id: str | None,
        top_k: int,
        rrf_k: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[KnowledgeMatch]:
        semantic = self._authorized_matches(
            vector=vector,
            tenant_id=tenant_id,
            principal_ids=principal_ids,
            knowledge_base_id=knowledge_base_id,
            metadata_filters=metadata_filters,
        )
        semantic.sort(key=lambda match: (-match.score, match.document_id, match.chunk_id))
        lexical = [(self._lexical_score(query, match), match) for match in semantic]
        lexical = [item for item in lexical if item[0] > 0]
        lexical.sort(key=lambda item: (-item[0], item[1].document_id, item[1].chunk_id))
        semantic_ranks = {match.chunk_id: index for index, match in enumerate(semantic, start=1)}
        lexical_ranks = {match.chunk_id: index for index, (_, match) in enumerate(lexical, start=1)}
        fused = [
            replace(
                match,
                score=(1 / (rrf_k + semantic_ranks[match.chunk_id]))
                + (
                    1 / (rrf_k + lexical_ranks[match.chunk_id])
                    if match.chunk_id in lexical_ranks
                    else 0
                ),
            )
            for match in semantic
        ]
        fused.sort(key=lambda match: (-match.score, match.document_id, match.chunk_id))
        return fused[:top_k]

    def _authorized_matches(
        self,
        *,
        vector: list[float],
        tenant_id: str,
        principal_ids: set[str],
        knowledge_base_id: str | None,
        metadata_filters: dict[str, str] | None,
    ) -> list[KnowledgeMatch]:
        candidates: list[KnowledgeMatch] = []
        for chunk in self._chunks.values():
            if chunk.tenant_id != tenant_id:
                continue
            if knowledge_base_id and chunk.knowledge_base_id != knowledge_base_id:
                continue
            if not chunk.public and not chunk.allowed_principal_ids.intersection(principal_ids):
                continue
            if metadata_filters and any(
                chunk.metadata.get(key) != value for key, value in metadata_filters.items()
            ):
                continue
            score = sum(left * right for left, right in zip(vector, chunk.vector, strict=True))
            candidates.append(
                KnowledgeMatch(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    knowledge_base_id=chunk.knowledge_base_id,
                    title=chunk.title,
                    content=chunk.content,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    score=score,
                    source_url=chunk.source_url,
                )
            )
        return candidates

    def _lexical_score(self, query: str, match: KnowledgeMatch) -> int:
        query_tokens = set(self._token_pattern.findall(query.lower()))
        if not query_tokens:
            return 0
        title_tokens = self._token_pattern.findall(match.title.lower())
        content_tokens = self._token_pattern.findall(match.content.lower())
        return sum(
            (2 * title_tokens.count(token)) + content_tokens.count(token) for token in query_tokens
        )


class QdrantKnowledgeBackend(KnowledgeBackend):
    def __init__(
        self,
        *,
        base_url: str,
        collection: str,
        dimensions: int,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._collection = collection
        self._dimensions = dimensions
        self._headers = {"api-key": api_key} if api_key else {}
        self._client = http_client or httpx.AsyncClient(timeout=10)
        self._owns_client = http_client is None
        self._collection_ready = False

    async def upsert(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks:
            return
        await self._ensure_collection()
        await self._request(
            "PUT",
            f"/collections/{self._collection}/points",
            params={"wait": "true"},
            json={
                "points": [
                    {"id": chunk.point_id, "vector": chunk.vector, "payload": chunk.payload()}
                    for chunk in chunks
                ]
            },
        )

    async def search(
        self,
        *,
        vector: list[float],
        tenant_id: str,
        principal_ids: set[str],
        knowledge_base_id: str | None,
        top_k: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[KnowledgeMatch]:
        await self._ensure_collection()
        must: list[dict[str, Any]] = [
            {"key": "tenant_id", "match": {"value": tenant_id}},
        ]
        if knowledge_base_id:
            must.append({"key": "knowledge_base_id", "match": {"value": knowledge_base_id}})
        for key, value in (metadata_filters or {}).items():
            must.append({"key": f"metadata.{key}", "match": {"value": value}})
        allowed: list[dict[str, Any]] = [
            {"key": "public", "match": {"value": True}},
        ]
        if principal_ids:
            allowed.append(
                {
                    "key": "allowed_principal_ids",
                    "match": {"any": sorted(principal_ids)},
                }
            )
        must.append({"should": allowed})
        body = await self._request(
            "POST",
            f"/collections/{self._collection}/points/query",
            json={
                "query": vector,
                "filter": {"must": must},
                "limit": top_k,
                "with_payload": True,
                "with_vector": False,
            },
        )
        result = body.get("result", {})
        points = result.get("points", []) if isinstance(result, dict) else result
        if not isinstance(points, list):
            raise KnowledgeBackendError("Qdrant query result must contain a points list")
        return [self._parse_match(point) for point in points]

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        response = await self._client.get(
            f"{self._base_url}/collections/{self._collection}", headers=self._headers
        )
        if response.status_code == 404:
            await self._request(
                "PUT",
                f"/collections/{self._collection}",
                json={"vectors": {"size": self._dimensions, "distance": "Cosine"}},
            )
        else:
            try:
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise KnowledgeBackendError(f"Qdrant collection check failed: {exc}") from exc
        self._collection_ready = True

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._client.request(
                method, f"{self._base_url}{path}", headers=self._headers, **kwargs
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise KnowledgeBackendError(f"Qdrant request failed: {exc}") from exc
        body = response.json()
        if not isinstance(body, dict):
            raise KnowledgeBackendError("Qdrant response must be a JSON object")
        return body

    @staticmethod
    def _parse_match(point: Any) -> KnowledgeMatch:
        if not isinstance(point, dict) or not isinstance(point.get("payload"), dict):
            raise KnowledgeBackendError("Qdrant point is missing payload")
        payload = point["payload"]
        try:
            return KnowledgeMatch(
                document_id=str(payload["document_id"]),
                chunk_id=str(payload["chunk_id"]),
                knowledge_base_id=str(payload["knowledge_base_id"]),
                title=str(payload["title"]),
                content=str(payload["content"]),
                char_start=int(payload["char_start"]),
                char_end=int(payload["char_end"]),
                score=float(point.get("score", 0.0)),
                source_url=(str(payload["source_url"]) if payload.get("source_url") else None),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise KnowledgeBackendError("Qdrant point payload is invalid") from exc


class KnowledgeService:
    def __init__(
        self,
        *,
        backend: KnowledgeBackend,
        embedder: DeterministicEmbedder,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        ranking: Literal["semantic", "hybrid"] = "semantic",
        hybrid_rrf_k: int = 60,
        reranker: Literal["none", "token_overlap"] = "none",
        rerank_candidate_k: int = 30,
    ) -> None:
        if chunk_size < 32:
            raise ValueError("Chunk size must be at least 32 characters")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be non-negative and smaller than chunk size")
        if hybrid_rrf_k < 1:
            raise ValueError("Hybrid RRF k must be at least 1")
        self.backend = backend
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ranking = ranking
        self.hybrid_rrf_k = hybrid_rrf_k
        self.reranker = reranker
        self.rerank_candidate_k = rerank_candidate_k
        self._token_overlap_reranker = TokenOverlapReranker()

    async def index_document(
        self,
        *,
        tenant_id: str,
        document: KnowledgeDocumentInput,
        document_id: str | None = None,
    ) -> IndexedKnowledgeDocument:
        document_id = document_id or f"doc_{uuid4().hex}"
        spans = self._chunk_spans(document.content)
        chunks = [
            KnowledgeChunk(
                point_id=str(uuid5(NAMESPACE_URL, f"{document_id}:{index}")),
                document_id=document_id,
                chunk_id=f"{document_id}_chunk_{index}",
                tenant_id=tenant_id,
                knowledge_base_id=document.knowledge_base_id,
                title=document.title,
                content=document.content[start:end],
                char_start=start,
                char_end=end,
                allowed_principal_ids=frozenset(document.allowed_principal_ids),
                public=document.public,
                source_url=str(document.source_url) if document.source_url else None,
                metadata=document.metadata,
                vector=self.embedder.embed(document.content[start:end]),
            )
            for index, (start, end) in enumerate(spans)
        ]
        await self.backend.upsert(chunks)
        return IndexedKnowledgeDocument(
            document_id=document_id,
            tenant_id=tenant_id,
            knowledge_base_id=document.knowledge_base_id,
            chunk_count=len(chunks),
        )

    async def search(
        self,
        *,
        query: str,
        tenant_id: str,
        principal_ids: set[str],
        knowledge_base_id: str | None,
        top_k: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[KnowledgeMatch]:
        vector = self.embedder.embed(query)
        bounded_top_k = max(1, min(top_k, 10))
        candidate_top_k = (
            max(bounded_top_k, self.rerank_candidate_k)
            if self.reranker == "token_overlap"
            else bounded_top_k
        )
        if self.ranking == "hybrid":
            if not self.backend.supports_hybrid_search:
                raise KnowledgeBackendError(
                    "Hybrid ranking is currently supported only by the in-memory backend"
                )
            matches = await self.backend.search_hybrid(
                query=query,
                vector=vector,
                tenant_id=tenant_id,
                principal_ids=principal_ids,
                knowledge_base_id=knowledge_base_id,
                top_k=candidate_top_k,
                rrf_k=self.hybrid_rrf_k,
                metadata_filters=metadata_filters or {},
            )
        else:
            matches = await self.backend.search(
                vector=vector,
                tenant_id=tenant_id,
                principal_ids=principal_ids,
                knowledge_base_id=knowledge_base_id,
                top_k=candidate_top_k,
                metadata_filters=metadata_filters or {},
            )
        if self.reranker == "token_overlap":
            matches = self._token_overlap_reranker.rerank(query, matches)
        return matches[:bounded_top_k]

    async def aclose(self) -> None:
        await self.backend.aclose()

    def _chunk_spans(self, content: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start = 0
        while start < len(content):
            end = min(len(content), start + self.chunk_size)
            if end < len(content):
                boundary = max(
                    content.rfind(marker, start + self.chunk_size // 2, end)
                    for marker in ("\n", ".", "。", "!", "！", "?", "？", " ")
                )
                if boundary > start:
                    end = boundary + 1
            spans.append((start, end))
            if end == len(content):
                break
            start = max(start + 1, end - self.chunk_overlap)
        return spans
