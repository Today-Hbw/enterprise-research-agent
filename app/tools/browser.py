from __future__ import annotations

from urllib.parse import urlparse

from playwright.async_api import async_playwright

from app.models import AccessContext, Source, SourceType, ToolCall, ToolPermission, ToolResult
from app.tools.base import BaseTool


class PlaywrightBrowserTool(BaseTool):
    name = "browser"
    description = "Navigate an approved HTTPS page and extract rendered text with Playwright."
    input_schema = {
        "type": "object",
        "properties": {"url": {"type": "string", "format": "uri"}},
        "required": ["url"],
    }
    permission = ToolPermission.HIGH
    is_stub = False

    def __init__(self, allowed_hosts: set[str], timeout_seconds: float) -> None:
        super().__init__(timeout_seconds)
        self.allowed_hosts = {host.lower().strip() for host in allowed_hosts if host.strip()}

    def _validate_url(self, raw_url: str) -> str:
        parsed = urlparse(raw_url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ValueError("Browser requires an HTTPS URL without credentials")
        if not host or not self.allowed_hosts:
            raise ValueError("Browser URL host is not approved")
        if not any(
            host == allowed or host.endswith(f".{allowed.lstrip('.')}")
            for allowed in self.allowed_hosts
        ):
            raise ValueError("Browser URL host is not approved")
        return parsed.geturl()

    async def execute(
        self, call: ToolCall, access_context: AccessContext | None = None
    ) -> ToolResult:
        del access_context
        try:
            url = self._validate_url(str(call.arguments.get("url", "")))
        except ValueError as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="Browser request rejected.",
                error=str(exc),
            )
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                response = await page.goto(
                    url, wait_until="domcontentloaded", timeout=int(self.timeout_seconds * 1000)
                )
                title = await page.title()
                text = (await page.locator("body").inner_text())[:20_000]
            finally:
                await browser.close()
        if response is None or not response.ok:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="Browser navigation failed.",
                error=f"HTTP status {response.status if response else 'unknown'}",
            )
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            success=True,
            summary=f"Rendered {title or url}.",
            data={"url": url, "title": title, "text": text},
            sources=[
                Source(
                    source_type=SourceType.BROWSER,
                    title=title or url,
                    url=url,
                    content_snippet=text[:500],
                )
            ],
        )
