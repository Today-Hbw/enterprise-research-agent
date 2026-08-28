import httpx
import pytest

from app.tools.web import BraveSearchBackend


@pytest.mark.asyncio
async def test_brave_search_excludes_non_http_citation_urls() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Unsafe result",
                            "url": "javascript:alert(1)",
                            "description": "Must not become a citation.",
                        },
                        {
                            "title": "Safe result",
                            "url": "https://public.example/report",
                            "description": "Usable evidence.",
                        },
                    ]
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = BraveSearchBackend(api_key="test-token", http_client=client)

    results = await backend.search("market", top_k=1)

    assert [result.url for result in results] == ["https://public.example/report"]
    await backend.aclose()
