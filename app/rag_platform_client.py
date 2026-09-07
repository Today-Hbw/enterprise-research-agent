"""RAG Platform 适配层。

将 rag-api (/api/v1) 包装为 KnowledgeBackend，使 enterprise-research-agent
能够通过版本化 HTTP API 与 rag-platform 交互，而无需直接依赖 rag-core / Qdrant。

检索走 search_hybrid（需要 query），因为 rag-api 在服务端做 Embedding。
upsert 走异步导入接口。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.knowledge import (
    KnowledgeBackend,
    KnowledgeBackendError,
    KnowledgeChunk,
    KnowledgeMatch,
)

logger = logging.getLogger(__name__)


class RagPlatformClient(KnowledgeBackend):
    """通过 rag-api HTTP API 实现的 KnowledgeBackend。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        score_threshold: float | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if score_threshold is not None and not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._score_threshold = score_threshold
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    @property
    def supports_hybrid_search(self) -> bool:
        """rag-api 在服务端 Embedding，需要 query 字符串。"""
        return True

    @property
    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    # ── knowledge bases ────────────────────────────────────────────────

    async def list_knowledge_bases(self) -> list[dict[str, Any]]:
        """列出当前 Bearer 凭证获授权的知识库。"""
        body = await self._request("GET", "/api/v1/knowledge-bases")
        items = body.get("knowledge_bases")
        if not isinstance(items, list):
            raise KnowledgeBackendError(
                "rag-api knowledge-base response must contain a knowledge_bases list"
            )

        knowledge_bases: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                raise KnowledgeBackendError("rag-api knowledge-base item must be an object")
            knowledge_base_id = item.get("id")
            name = item.get("name")
            source = item.get("source")
            document_count = item.get("document_count")
            if (
                not isinstance(knowledge_base_id, str)
                or not isinstance(name, str)
                or not isinstance(source, str)
                or not isinstance(document_count, int)
                or isinstance(document_count, bool)
                or document_count < 0
            ):
                raise KnowledgeBackendError("rag-api knowledge-base item has invalid fields")
            knowledge_bases.append(
                {
                    "id": knowledge_base_id,
                    "name": name,
                    "source": source,
                    "document_count": document_count,
                }
            )
        return knowledge_bases

    # ── upsert ──────────────────────────────────────────────────────────

    async def upsert(self, chunks: list[KnowledgeChunk]) -> None:
        """通过 rag-api 导入文档。

        KnowledgeChunk 列表属于同一文档（KnowledgeService 保证），
        拼接为一个文档提交给 rag-api。
        """
        if not chunks:
            return

        first = chunks[0]
        full_content = self._reconstruct_content(chunks)

        body: dict[str, Any] = {
            "external_document_id": first.document_id,
            "title": first.title,
            "content": full_content,
            "content_type": "text/plain",
            "idempotency_key": self._make_idempotency_key(first.document_id, full_content),
            "metadata": dict(first.metadata) if first.metadata else None,
        }

        if first.allowed_principal_ids or first.public:
            body["access_control"] = {
                "allowed_principal_ids": sorted(first.allowed_principal_ids),
                "is_public": first.public,
            }
        if first.source_url:
            body["source_url"] = first.source_url

        await self._request(
            "POST",
            f"/api/v1/knowledge-bases/{first.knowledge_base_id}/documents",
            json=body,
        )

    # ── search ──────────────────────────────────────────────────────────

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
        """向量检索（rag-api 不支持直接传向量，退化为空结果）。

        KnowledgeService 在 ranking="semantic" 时会调用 search(vector)。
        rag-api 在服务端做 Embedding，需要 query 字符串，无法适配 vector-only 调用。
        建议使用 ranking="hybrid" 走 search_hybrid。
        """
        logger.warning(
            "RagPlatformClient.search() called with vector only; "
            "rag-api requires a query string. Use ranking='hybrid' in settings."
        )
        return []

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
        """通过 rag-api 执行检索。

        rag-api 在服务端做 Embedding，这里只传 query。
        """
        if not knowledge_base_id:
            raise KnowledgeBackendError("knowledge_base_id is required for rag-api search")

        filters = None
        if metadata_filters:
            filters = [
                {"key": k, "equals": v} for k, v in metadata_filters.items()
            ]

        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "include_content": True,
        }
        if self._score_threshold is not None:
            body["score_threshold"] = self._score_threshold
        if filters:
            body["filters"] = filters

        resp = await self._request(
            "POST",
            f"/api/v1/knowledge-bases/{knowledge_base_id}/search",
            json=body,
        )

        if not resp.get("success", False):
            error_code = resp.get("error_code", "unknown")
            raise KnowledgeBackendError(
                f"rag-api search failed: {error_code}"
            )

        results = resp.get("results", [])
        return [self._parse_source_item(item) for item in results]

    # ── lifecycle ───────────────────────────────────────────────────────

    async def get_document(
        self, knowledge_base_id: str, document_id: str
    ) -> dict[str, Any] | None:
        """查询文档状态（非 KnowledgeBackend 接口，辅助方法）。"""
        try:
            return await self._request(
                "GET",
                f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}",
            )
        except KnowledgeBackendError as e:
            if "404" in str(e):
                return None
            raise

    async def get_ingestion_job(self, job_id: str) -> dict[str, Any] | None:
        """查询异步导入或更新任务状态。"""
        try:
            return await self._request("GET", f"/api/v1/ingestion-jobs/{job_id}")
        except KnowledgeBackendError as exc:
            if "404" in str(exc):
                return None
            raise

    async def update_document(
        self,
        knowledge_base_id: str,
        document_id: str,
        *,
        updates: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """异步覆盖更新 API 管理的文档。"""
        body = dict(updates)
        canonical_updates = json.dumps(
            body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        body["idempotency_key"] = idempotency_key or self._make_idempotency_key(
            document_id, canonical_updates
        )
        return await self._request(
            "PUT",
            f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}",
            json=body,
        )

    async def delete_document(
        self,
        knowledge_base_id: str,
        document_id: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """删除文档（非 KnowledgeBackend 接口，辅助方法）。"""
        key = idempotency_key or self._make_idempotency_key(document_id, "delete")
        return await self._request(
            "DELETE",
            f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}",
            headers={"Idempotency-Key": key},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ── internal ────────────────────────────────────────────────────────

    async def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None
        request_headers = {**self._headers, **kwargs.pop("headers", {})}

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(
                    method, url, headers=request_headers, **kwargs
                )
                if response.status_code >= 500 and attempt < self._max_retries:
                    logger.warning(
                        "rag-api 5xx (attempt %d/%d): HTTP %d",
                        attempt + 1,
                        self._max_retries + 1,
                        response.status_code,
                    )
                    await self._sleep(0.5 * (attempt + 1))
                    continue
                if response.is_error:
                    raise KnowledgeBackendError(self._format_http_error(response))
                body = response.json()
                if not isinstance(body, dict):
                    raise KnowledgeBackendError("Response is not a JSON object")
                return body
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning(
                    "rag-api timeout (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "rag-api error (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
            except KnowledgeBackendError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.exception("rag-api unexpected error")
                raise KnowledgeBackendError(f"rag-api unexpected: {exc}") from exc

        raise KnowledgeBackendError(
            f"rag-api request failed after {self._max_retries + 1} attempts: {last_exc}"
        )

    @staticmethod
    def _format_http_error(response: httpx.Response) -> str:
        code = "unknown"
        message = response.reason_phrase or "Request failed"
        request_id = None
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            if isinstance(payload.get("code"), str):
                code = payload["code"]
            elif isinstance(payload.get("error_code"), str):
                code = payload["error_code"]
            if isinstance(payload.get("message"), str):
                message = payload["message"]
            if isinstance(payload.get("request_id"), str):
                request_id = payload["request_id"]
        suffix = f" (request_id={request_id})" if request_id else ""
        return f"rag-api HTTP {response.status_code} {code}: {message}{suffix}"

    @staticmethod
    async def _sleep(seconds: float) -> None:
        import asyncio
        await asyncio.sleep(seconds)

    @staticmethod
    def _parse_source_item(item: dict[str, Any]) -> KnowledgeMatch:
        location = item.get("location") or {}
        return KnowledgeMatch(
            document_id=str(item["document_id"]),
            chunk_id=str(item["chunk_id"]),
            knowledge_base_id=str(item["knowledge_base_id"]),
            title=str(item.get("title", "")),
            content=str(item.get("content", "") or ""),
            char_start=int(location.get("char_start") or 0),
            char_end=int(location.get("char_end") or 0),
            score=float(item.get("score", 0.0)),
            source_url=str(item["source_url"]) if item.get("source_url") else None,
        )

    @staticmethod
    def _make_idempotency_key(document_id: str, content: str) -> str:
        import hashlib
        digest = hashlib.sha256(f"{document_id}:{content}".encode()).hexdigest()[:16]
        return f"agent_{digest}"

    @staticmethod
    def _reconstruct_content(chunks: list[KnowledgeChunk]) -> str:
        ordered = sorted(chunks, key=lambda chunk: (chunk.char_start, chunk.char_end))
        parts: list[str] = []
        cursor = 0
        for chunk in ordered:
            overlap = max(0, cursor - chunk.char_start)
            if overlap < len(chunk.content):
                parts.append(chunk.content[overlap:])
            cursor = max(cursor, chunk.char_end)
        return "".join(parts)
