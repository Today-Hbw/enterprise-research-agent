from __future__ import annotations

from app.knowledge import DeterministicEmbedder, InMemoryKnowledgeBackend, KnowledgeService
from app.models import AccessContext, ToolCall
from app.tools.knowledge import KnowledgeBaseListTool, KnowledgeSearchTool
from app.tools.stubs import build_tool_registry


class FakeRagPlatformClient:
    async def list_knowledge_bases(self) -> list[dict[str, object]]:
        return [
            {
                "id": "123456",
                "name": "Employee Benefits Policy",
                "source": "yuque",
                "document_count": 3417,
            }
        ]


async def test_knowledge_base_list_tool_exposes_authorized_platform_knowledge_bases():
    tool = KnowledgeBaseListTool(client=FakeRagPlatformClient(), timeout_seconds=1)

    result = await tool.execute(
        ToolCall(name="knowledge_base_list", arguments={}),
        AccessContext(tenant_id="local-tenant", principal_ids={"user-1"}),
    )

    assert result.success is True
    assert result.data == {
        "knowledge_bases": [
            {
                "id": "123456",
                "name": "Employee Benefits Policy",
                "source": "yuque",
                "document_count": 3417,
            }
        ]
    }
    assert result.sources == []


def test_registry_adds_discovery_tool_only_when_provided():
    discovery_tool = KnowledgeBaseListTool(
        client=FakeRagPlatformClient(), timeout_seconds=1
    )

    registry = build_tool_registry(
        timeout_seconds=1,
        knowledge_base_tool=discovery_tool,
    )

    assert "knowledge_base_list" in {spec.name for spec in registry.specs()}


def test_rag_platform_search_schema_requires_knowledge_base_id():
    tool = KnowledgeSearchTool(
        service=KnowledgeService(
            backend=InMemoryKnowledgeBackend(),
            embedder=DeterministicEmbedder(),
        ),
        timeout_seconds=1,
        require_knowledge_base_id=True,
    )

    assert tool.spec.input_schema["required"] == ["query", "knowledge_base_id"]
