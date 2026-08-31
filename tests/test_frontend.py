import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
STATIC_DIR = Path(__file__).parents[1] / "app" / "static"


def test_frontend_serves_explicit_inspector_controls_and_local_markdown_renderer() -> None:
    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert 'id="inspector-toggle"' in html
    assert 'aria-controls="inspector"' in html
    assert 'id="inspector-close"' in html
    assert 'id="inspector-backdrop"' in html
    assert 'data-tab="plan"' in html
    assert 'id="plan-pane"' in html
    assert 'id="plan-list"' in html
    assert html.index("/static/markdown.js") < html.index("/static/app.js")

    styles = client.get("/static/styles.css").text
    assert "@media (min-width: 701px) and (max-width: 980px)" in styles
    assert "body.inspector-open .inspector" in styles


def test_frontend_connects_markdown_to_stored_and_streamed_assistant_messages() -> None:
    script = client.get("/static/app.js").text

    assert 'role === "assistant" && !pending' in script
    assert "renderAssistantMarkdown(assistantNode, data.content)" in script
    assert 'window.matchMedia("(min-width: 701px) and (max-width: 980px)")' in script
    assert 'event.key === "Escape"' in script
    assert 'event === "plan_created" || event === "plan_updated"' in script
    assert 'event === "plan_step_updated"' in script
    assert "renderPlan(run.plan)" in script


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_markdown_renderer_formats_supported_syntax_and_escapes_unsafe_content() -> None:
    renderer_path = json.dumps(str(STATIC_DIR / "markdown.js"))
    node_script = f"""
const {{ renderMarkdown }} = require({renderer_path});
const rendered = renderMarkdown(
  '## Result\\n\\n- **safe** item\\n- [docs](https://example.com/a?q=1)\\n\\n' +
  '<img src=x onerror=alert(1)> [bad](javascript:alert(1)) `code`'
);
if (!rendered.includes('<h2>Result</h2>')) process.exit(1);
if (!rendered.includes('<strong>safe</strong>')) process.exit(2);
if (!rendered.includes('href="https://example.com/a?q=1"')) process.exit(3);
if (!rendered.includes('&lt;img src=x onerror=alert(1)&gt;')) process.exit(4);
if (rendered.includes('<img') || rendered.includes('href="javascript:')) process.exit(5);
if (!rendered.includes('<code>code</code>')) process.exit(6);
"""

    subprocess.run(["node", "-e", node_script], check=True)
