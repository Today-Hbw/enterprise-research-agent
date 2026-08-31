import inspect

import httpx
import pytest

from app.models import ToolCall
from app.tools.web import BraveSearchBackend, HttpFetchTool, SafeHttpFetcher, WebSearchTool


def _mock_client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler, base_url="https://unit.test")


@pytest.mark.asyncio
async def test_brave_search_uses_server_side_token_and_returns_citations() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["token"] = request.headers.get("X-Subscription-Token")
        observed["query"] = request.url.params.get("q")
        observed["count"] = request.url.params.get("count")
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Primary source",
                            "url": "https://public.example/report",
                            "description": "A public report.",
                            "age": "2026-08-28T00:00:00Z",
                        }
                    ]
                }
            },
        )

    client = _mock_client(httpx.MockTransport(handler))
    backend = BraveSearchBackend(api_key="secret-token", http_client=client)
    tool = WebSearchTool(backend=backend, timeout_seconds=1)

    result = await tool.execute(
        ToolCall(name="web_search", arguments={"query": "market", "top_k": 1})
    )

    assert result.success is True
    assert observed == {"token": "secret-token", "query": "market", "count": "1"}
    assert result.data["results"][0]["url"] == "https://public.example/report"
    assert result.sources[0].title == "Primary source"
    assert result.sources[0].url == "https://public.example/report"
    assert "secret-token" not in str(result.model_dump())
    assert "api_key" not in tool.input_schema["properties"]
    await backend.aclose()


@pytest.mark.asyncio
async def test_safe_fetch_parses_html_and_produces_anchored_source() -> None:
    async def resolver(host: str, port: int) -> set[str]:
        assert (host, port) == ("public.example", 443)
        return {"93.184.216.34"}

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://public.example/report"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=b"<html><head><title>Report</title><style>hidden</style></head>"
            b"<body><h1>Market update</h1><script>ignore()</script>"
            b"<p>Demand rose.</p>"
            b"</body></html>",
        )

    client = _mock_client(httpx.MockTransport(handler))
    fetcher = SafeHttpFetcher(
        allowed_hosts={"public.example"}, resolver=resolver, http_client=client
    )
    tool = HttpFetchTool(fetcher=fetcher, timeout_seconds=1)

    result = await tool.execute(
        ToolCall(name="http_fetch", arguments={"url": "https://public.example/report"})
    )

    assert result.success is True
    assert result.data["status_code"] == 200
    assert result.sources[0].title == "Report"
    assert result.sources[0].url == "https://public.example/report"
    assert "Market update" in result.sources[0].content_snippet
    assert "ignore" not in result.sources[0].content_snippet
    assert "hidden" not in result.sources[0].content_snippet
    await fetcher.aclose()


@pytest.mark.asyncio
async def test_safe_fetch_download_accepts_supported_file_content_types() -> None:
    async def resolver(host: str, port: int) -> set[str]:
        assert (host, port) == ("public.example", 443)
        return {"93.184.216.34"}

    body = b"supplier,amount\nA,42\n"
    client = _mock_client(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={
                    "content-type": "text/csv; charset=utf-8",
                    "content-disposition": 'attachment; filename="purchases.csv"',
                },
                content=body,
            )
        )
    )
    fetcher = SafeHttpFetcher(
        allowed_hosts={"public.example"}, resolver=resolver, http_client=client
    )

    resource = await fetcher.download("https://public.example/purchases.csv")
    page = await fetcher.fetch("https://public.example/purchases.csv")

    assert resource.body == body
    assert resource.content_type == "text/csv"
    assert resource.content_disposition == 'attachment; filename="purchases.csv"'
    assert page.title == "purchases.csv"
    assert page.text == "supplier\tamount\nA\t42"
    await fetcher.aclose()


@pytest.mark.asyncio
async def test_safe_fetch_rejects_private_and_redirected_targets_before_request() -> None:
    calls = 0

    async def resolver(host: str, port: int) -> set[str]:
        if host == "public.example":
            return {"93.184.216.34"}
        return {"127.0.0.1"}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(302, headers={"location": "https://internal.example/secret"})

    client = _mock_client(httpx.MockTransport(handler))
    fetcher = SafeHttpFetcher(
        allowed_hosts={"public.example", "internal.example"}, resolver=resolver, http_client=client
    )

    with pytest.raises(ValueError, match="public IP"):
        await fetcher.fetch("https://public.example/redirect")

    assert calls == 1
    await fetcher.aclose()


@pytest.mark.asyncio
async def test_safe_fetch_rejects_private_literal_and_enforces_response_limit() -> None:
    async def resolver(host: str, port: int) -> set[str]:
        assert (host, port) == ("public.example", 443)
        return {"93.184.216.34"}

    client = _mock_client(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/plain"}, content=b"abcdef"
            )
        )
    )
    fetcher = SafeHttpFetcher(
        allowed_hosts={"public.example"}, resolver=resolver, http_client=client, max_bytes=5
    )

    with pytest.raises(ValueError, match="IP literals"):
        await fetcher.fetch("https://127.0.0.1/metadata")
    with pytest.raises(ValueError, match="response body exceeds"):
        await fetcher.fetch("https://public.example/large")
    await fetcher.aclose()


def test_safe_fetch_uses_async_dns_resolution() -> None:
    async def resolver(host: str, port: int) -> set[str]:
        return {"93.184.216.34"}

    fetcher = SafeHttpFetcher(allowed_hosts={"public.example"}, resolver=resolver)
    assert inspect.iscoroutinefunction(fetcher._resolver)
