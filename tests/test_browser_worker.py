import pytest

from app.models import ToolCall
from app.tools.browser import PlaywrightBrowserTool


@pytest.mark.asyncio
async def test_browser_rejects_unapproved_or_non_https_urls() -> None:
    tool = PlaywrightBrowserTool({"approved.example"}, timeout_seconds=1)
    result = await tool.execute(
        ToolCall(name="browser", arguments={"url": "http://approved.example"})
    )
    blocked = await tool.execute(
        ToolCall(name="browser", arguments={"url": "https://private.example"})
    )

    assert result.success is False
    assert blocked.success is False
