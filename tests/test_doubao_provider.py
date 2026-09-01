import json

import httpx
import pytest

from app.agent.provider import (
    DoubaoResponsesProvider,
    ProviderConfigurationError,
    build_provider,
)
from app.config import Settings
from app.models import Message, MessageRole, ToolResult
from app.tools.stubs import build_stub_registry


def build_messages() -> list[Message]:
    return [Message(role=MessageRole.USER, content="研究供应商集中度。")]


@pytest.mark.asyncio
async def test_doubao_provider_round_trips_tools_and_parses_ark_output() -> None:
    payloads: list[dict] = []
    responses = iter(
        [
            {
                "id": "resp_doubao_first",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_doubao_1",
                        "name": "knowledge_search",
                        "arguments": '{"query":"供应商集中度"}',
                    }
                ],
                "usage": {"input_tokens": 42, "output_tokens": 8},
            },
            {
                "id": "resp_doubao_final",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "建议按季度复核供应商集中度。",
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 25, "output_tokens": 12},
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://ark.example.test/api/v3/responses"
        assert request.headers["authorization"] == "Bearer ark-secret"
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=next(responses))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = DoubaoResponsesProvider(
            api_key="ark-secret",
            model="doubao-test-model",
            base_url="https://ark.example.test/api/v3/",
            timeout_seconds=5,
            http_client=client,
        )
        tools = build_stub_registry(1).specs()
        first = await provider.decide(
            query="研究供应商集中度。",
            history=build_messages(),
            available_tools=tools,
            prior_results=[],
            step=0,
            run_id="run_doubao_test",
        )
        second = await provider.decide(
            query="研究供应商集中度。",
            history=build_messages(),
            available_tools=tools,
            prior_results=[
                ToolResult(
                    call_id="call_doubao_1",
                    tool_name="knowledge_search",
                    success=True,
                    summary="找到供应商政策。",
                )
            ],
            step=1,
            run_id="run_doubao_test",
        )

    assert provider.name == "doubao-responses:doubao-test-model"
    assert [call.call_id for call in first.tool_calls] == ["call_doubao_1"]
    assert second.final_answer == "建议按季度复核供应商集中度。"
    assert (second.input_tokens, second.output_tokens) == (25, 12)
    assert payloads[0]["store"] is True
    assert payloads[0]["parallel_tool_calls"] is True
    assert "strict" not in payloads[0]["tools"][0]
    assert payloads[1]["previous_response_id"] == "resp_doubao_first"
    assert payloads[1]["input"][0]["type"] == "function_call_output"
    assert payloads[1]["input"][0]["call_id"] == "call_doubao_1"


def test_doubao_mode_uses_dedicated_settings_and_requires_api_key() -> None:
    with pytest.raises(ProviderConfigurationError, match="DOUBAO_API_KEY"):
        build_provider(Settings(llm_provider="doubao"))

    provider = build_provider(
        Settings(
            llm_provider="doubao",
            doubao_api_key="ark-secret",
            doubao_model="doubao-custom-model",
            doubao_base_url="https://ark.example.test/api/v3",
        )
    )
    assert isinstance(provider, DoubaoResponsesProvider)
    assert provider.name == "doubao-responses:doubao-custom-model"
