"""RAG Platform 适配层。

将 rag-api (/api/v1) 包装为 KnowledgeBackend，使 enterprise-research-agent
能够通过版本化 HTTP API 与 rag-platform 交互，而无需直接依赖 rag-core / Qdrant。

检索走 search_hybrid（需要 query），因为 rag-api 在服务端做 Embedding。
upsert 走异步导入接口。
"""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

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
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
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

    # ── upsert ──────────────────────────────────────────────────────────

    async def upsert(self, chunks: list[KnowledgeChunk]) -> None:
        """通过 rag-api 导入文档。

        KnowledgeChunk 列表属于同一文档（KnowledgeService 保证），
        拼接为一个文档提交给 rag-api。
        """
        if not chunks:
            return

        first = chunks[0]
        full_content = first.content
        for chunk in chunks[1:]:
            full_content += "\n" + chunk.content

        body: dict[str, Any] = {
            "external_document_id": first.document_id,
            "title": first.title,
            "content": full_content,
            "content_type": "text/plain",
            "idempotency_key": self._make_idempotency_key(first.document_id, full_content),
            "metadata": dict(first.metadata) if first.metadata else None,
        }

        if first.allowed_principal_ids:
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
            params={"idempotency_key": key},
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

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(
                    method, url, headers=self._headers, **kwargs
                )
                if response.status_code == 404:
                    raise KnowledgeBackendError(f"404 Not Found: {path}")
                if response.status_code == 401:
                    raise KnowledgeBackendError("401 Unauthorized: check RAG_PLATFORM_API_KEY")
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise KnowledgeBackendError("Response is not a JSON object")
                return body
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning("rag-api timeout (attempt %d/%d): %s", attempt + 1, self._max_retries + 1, exc)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status = exc.response.status_code
                if status >= 500 and attempt < self._max_retries:
                    logger.warning("rag-api 5xx (attempt %d/%d): %s", attempt + 1, self._max_retries + 1, exc)
                    await self._sleep(0.5 * (attempt + 1))
                    continue
                raise KnowledgeBackendError(f"rag-api HTTP {status}: {exc}") from exc
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning("rag-api error (attempt %d/%d): %s", attempt + 1, self._max_retries + 1, exc)
            except KnowledgeBackendError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.exception("rag-api unexpected error")
                raise KnowledgeBackendError(f"rag-api unexpected: {exc}") from exc

        raise KnowledgeBackendError(f"rag-api request failed after {self._max_retries + 1} attempts: {last_exc}")

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
            content=str(item.get("content_snippet", "") or ""),
            char_start=int(location.get("char_start") or 0),
            char_end=int(location.get("char_end") or 0),
            score=float(item.get("score", 0.0)),
            source_url=str(item["source_url"]) if item.get("source_url") else None,
        )

    @staticmethod
    def _make_idempotency_key(document_id: str, content: str) -> str:
        import hashlib
        digest = hashlib.sha256(f"{document_id}:{len(content)}".encode()).hexdigest()[:16]
        return f"agent_{digest}"
