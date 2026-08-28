from app.agent.runtime import AgentRuntime
from app.models import Source, SourceType, ToolResult


def test_sources_receive_stable_evidence_anchors_and_sorting() -> None:
    first = Source(
        source_type=SourceType.WEB, title="Z", url="https://example.com/z", content_snippet="z"
    )
    second = Source(
        source_type=SourceType.DOCUMENT,
        title="A",
        document_id="doc-1",
        chunk_id="chunk-1",
        content_snippet="a",
    )
    duplicate = Source(
        source_type=SourceType.WEB, title="Z", url="https://example.com/z", content_snippet="z"
    )
    results = [
        ToolResult(
            call_id="one", tool_name="web_search", success=True, summary="", sources=[first]
        ),
        ToolResult(
            call_id="two",
            tool_name="knowledge_search",
            success=True,
            summary="",
            sources=[second, duplicate],
        ),
    ]

    sources = AgentRuntime._deduplicate_sources(results)

    assert [item.evidence_anchor for item in sources] == ["chunk-1", "https://example.com/z"]
