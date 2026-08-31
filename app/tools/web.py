from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from app.document_parser import DocumentParser, DownloadedResource
from app.models import AccessContext, Source, SourceType, ToolCall, ToolResult
from app.tools.base import BaseTool

Resolver = Callable[[str, int], Awaitable[set[str]]]


def _is_safe_citation_url(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme in {"https", "http"}
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
    )


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    description: str
    age: str | None = None


class BraveSearchBackend:
    """Minimal server-side adapter for the Brave Web Search REST API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.search.brave.com/res/v1/web/search",
        timeout_seconds: float = 10,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("BRAVE_SEARCH_API_KEY is required for WEB_SEARCH_BACKEND=brave")
        self._api_key = api_key
        self._base_url = base_url
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = http_client is None

    async def search(self, query: str, *, top_k: int) -> list[WebSearchResult]:
        normalized_query = query.strip()
        if not normalized_query or len(normalized_query) > 400:
            raise ValueError("Web search query must be 1 to 400 characters")
        if len(normalized_query.split()) > 50:
            raise ValueError("Web search query must contain at most 50 words")
        try:
            response = await self._client.get(
                self._base_url,
                headers={"X-Subscription-Token": self._api_key, "Accept": "application/json"},
                params={
                    "q": normalized_query,
                    "count": max(1, min(top_k, 10)),
                    "safesearch": "moderate",
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ValueError(f"Web search request failed: {exc}") from exc

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Web search response must be a JSON object")
        web = payload.get("web", {})
        results = web.get("results", []) if isinstance(web, dict) else []
        if not isinstance(results, list):
            raise ValueError("Web search response results must be a list")
        parsed: list[WebSearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            url = item.get("url")
            description = item.get("description", "")
            if (
                not isinstance(title, str)
                or not isinstance(url, str)
                or not title
                or not _is_safe_citation_url(url)
            ):
                continue
            parsed.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    description=description if isinstance(description, str) else "",
                    age=item.get("age") if isinstance(item.get("age"), str) else None,
                )
            )
            if len(parsed) == top_k:
                break
        return parsed

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


@dataclass(frozen=True)
class FetchedPage:
    final_url: str
    status_code: int
    content_type: str
    title: str
    text: str


async def _resolve_public_addresses(host: str, port: int) -> set[str]:
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return {record[4][0] for record in records}


class SafeHttpFetcher:
    """Allowlisted, size-bounded HTTP(S) fetcher for public evidence only."""

    _REDIRECT_CODES = {301, 302, 303, 307, 308}
    _ALLOWED_CONTENT_TYPES = DocumentParser.SUPPORTED_CONTENT_TYPES

    def __init__(
        self,
        *,
        allowed_hosts: set[str],
        resolver: Resolver = _resolve_public_addresses,
        max_bytes: int = 1_000_000,
        max_redirects: int = 3,
        timeout_seconds: float = 10,
        allow_http: bool = False,
        http_client: httpx.AsyncClient | None = None,
        parser: DocumentParser | None = None,
    ) -> None:
        self._allowed_hosts = {host.strip().lower() for host in allowed_hosts if host.strip()}
        if not self._allowed_hosts:
            raise ValueError("HTTP_ALLOWED_HOSTS is required for HTTP_FETCH_BACKEND=safe")
        self._resolver = resolver
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects
        self._allow_http = allow_http
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = http_client is None
        self._parser = parser or DocumentParser()

    async def fetch(self, url: str) -> FetchedPage:
        resource = await self.download(url)
        parsed = await asyncio.to_thread(self._parser.parse, resource)
        return FetchedPage(
            final_url=resource.final_url,
            status_code=resource.status_code,
            content_type=resource.content_type,
            title=parsed.title,
            text=parsed.content,
        )

    async def download(self, url: str) -> DownloadedResource:
        current_url = url
        for redirect_count in range(self._max_redirects + 1):
            await self._validate_url(current_url)
            try:
                async with self._client.stream(
                    "GET",
                    current_url,
                    follow_redirects=False,
                    headers={"Accept": ",".join(sorted(self._ALLOWED_CONTENT_TYPES))},
                ) as response:
                    if response.status_code in self._REDIRECT_CODES:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("Redirect response is missing Location")
                        if redirect_count == self._max_redirects:
                            raise ValueError("HTTP fetch exceeded redirect limit")
                        current_url = urljoin(str(response.url), location)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if content_type not in self._ALLOWED_CONTENT_TYPES:
                        raise ValueError(
                            f"Unsupported response content type: {content_type or 'missing'}"
                        )
                    content_length = response.headers.get("content-length")
                    if (
                        content_length
                        and content_length.isdigit()
                        and int(content_length) > self._max_bytes
                    ):
                        raise ValueError("HTTP fetch response body exceeds configured limit")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > self._max_bytes:
                            raise ValueError("HTTP fetch response body exceeds configured limit")
                    return DownloadedResource(
                        final_url=str(response.url),
                        status_code=response.status_code,
                        content_type=content_type,
                        body=bytes(body),
                        content_disposition=response.headers.get("content-disposition"),
                    )
            except httpx.HTTPError as exc:
                raise ValueError(f"HTTP fetch request failed: {exc}") from exc
        raise ValueError("HTTP fetch exceeded redirect limit")

    async def _validate_url(self, value: str) -> None:
        parsed = urlsplit(value)
        if parsed.scheme not in ({"https", "http"} if self._allow_http else {"https"}):
            raise ValueError("HTTP fetch only permits HTTPS URLs")
        if parsed.username or parsed.password:
            raise ValueError("HTTP fetch URLs must not contain credentials")
        host = parsed.hostname
        if not host:
            raise ValueError("HTTP fetch URL must contain a host")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise ValueError("HTTP fetch rejects IP literals")
        normalized_host = host.lower()
        if not self._is_allowed_host(normalized_host):
            raise ValueError("HTTP fetch host is not allowlisted")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = await self._resolver(normalized_host, port)
        if not addresses:
            raise ValueError("HTTP fetch host did not resolve")
        for address in addresses:
            try:
                if not ipaddress.ip_address(address).is_global:
                    raise ValueError("HTTP fetch host must resolve only to public IP addresses")
            except ValueError as exc:
                if str(exc).startswith("HTTP fetch"):
                    raise
                raise ValueError("HTTP fetch resolver returned an invalid IP address") from exc

    def _is_allowed_host(self, host: str) -> bool:
        return any(
            host == allowed or (allowed.startswith(".") and host.endswith(allowed))
            for allowed in self._allowed_hosts
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search public web sources through the configured server-side provider."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 400},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }
    is_stub = False

    def __init__(self, *, backend: BraveSearchBackend, timeout_seconds: float) -> None:
        super().__init__(timeout_seconds)
        self._backend = backend

    async def execute(
        self, call: ToolCall, access_context: AccessContext | None = None
    ) -> ToolResult:
        del access_context
        query = str(call.arguments.get("query", "")).strip()
        raw_top_k = call.arguments.get("top_k", 5)
        top_k = raw_top_k if isinstance(raw_top_k, int) and not isinstance(raw_top_k, bool) else 5
        try:
            results = await self._backend.search(query, top_k=max(1, min(top_k, 10)))
        except ValueError as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="Web search could not complete.",
                error=str(exc),
            )
        sources = [
            Source(
                source_type=SourceType.WEB,
                title=result.title,
                url=result.url,
                content_snippet=result.description,
            )
            for result in results
        ]
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            success=True,
            summary=f"Found {len(results)} public web result(s).",
            data={
                "results": [
                    {
                        "title": result.title,
                        "url": result.url,
                        "description": result.description,
                        "age": result.age,
                    }
                    for result in results
                ]
            },
            sources=sources,
        )


class HttpFetchTool(BaseTool):
    name = "http_fetch"
    description = "Fetch an allowlisted public HTTPS URL with bounded parsing."
    input_schema = {
        "type": "object",
        "properties": {"url": {"type": "string", "format": "uri", "maxLength": 2_000}},
        "required": ["url"],
    }
    is_stub = False

    def __init__(self, *, fetcher: SafeHttpFetcher, timeout_seconds: float) -> None:
        super().__init__(timeout_seconds)
        self._fetcher = fetcher

    async def execute(
        self, call: ToolCall, access_context: AccessContext | None = None
    ) -> ToolResult:
        del access_context
        url = str(call.arguments.get("url", "")).strip()
        try:
            page = await self._fetcher.fetch(url)
        except ValueError as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="HTTP fetch was rejected or failed.",
                error=str(exc),
            )
        snippet = page.text[:4_000]
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            success=True,
            summary=f"Fetched {page.content_type} evidence from {page.final_url}.",
            data={
                "status_code": page.status_code,
                "content_type": page.content_type,
                "final_url": page.final_url,
                "truncated": len(page.text) > len(snippet),
            },
            sources=[
                Source(
                    source_type=SourceType.WEB,
                    title=page.title,
                    url=page.final_url,
                    content_snippet=snippet,
                )
            ],
        )
