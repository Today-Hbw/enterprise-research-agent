import json

import httpx
import pytest

from app.agent.provider import (
    OpenAIResponsesProvider,
    ProviderConfigurationError,
    ProviderProtocolError,
    build_provider,
)
from app.agent.runtime import AgentRuntime
from app.config import Settings
from app.models import Message, MessageRole, ToolResult
from app.store import InMemoryStore
from app.tools.stubs import build_stub_registry


def build_messages() -> list[Message]:
    return [Message(role=MessageRole.USER, content="Research supplier concentration.")]


@pytest.mark.asyncio
async def test_openai_provider_round_trips_tool_output_and_token_usage() -> None:
    payloads: list[dict] = []
    responses = iter(
        [
            {
                "id": "resp_first",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "model_call_1",
                        "name": "web_search",
                        "arguments": '{"query":"supplier concentration"}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "model_call_2",
                        "name": "knowledge_search",
                        "arguments": '{"query":"supplier policy"}',
                    },
                ],
                "usage": {"input_tokens": 120, "output_tokens": 30},
            },
            {
                "id": "resp_final",
                "output": [],
                "output_text": "Use a quarterly supplier review and monitor concentration.",
                "usage": {"input_tokens": 80, "output_tokens": 20},
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://mock.openai.test/v1/responses"
        assert request.headers["authorization"] == "Bearer test-secret"
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=next(responses))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIResponsesProvider(
            api_key="test-secret",
            model="test-model",
            base_url="https://mock.openai.test/v1/",
            timeout_seconds=5,
            http_client=client,
        )
        tools = build_stub_registry(1).specs()
        first = await provider.decide(
            query="Research supplier concentration.",
            history=build_messages(),
            available_tools=tools,
            prior_results=[],
            step=0,
            run_id="run_provider_test",
        )
        second = await provider.decide(
            query="Research supplier concentration.",
            history=build_messages(),
            available_tools=tools,
            prior_results=[
                ToolResult(
                    call_id="model_call_1",
                    tool_name="web_search",
                    success=True,
                    summary="Market data found.",
                    data={"results": 3},
                ),
                ToolResult(
                    call_id="model_call_2",
                    tool_name="knowledge_search",
                    success=True,
                    summary="Policy found.",
                ),
            ],
            step=1,
            run_id="run_provider_test",
        )

    assert [call.call_id for call in first.tool_calls] == ["model_call_1", "model_call_2"]
    assert first.input_tokens == 120
    assert first.output_tokens == 30
    assert second.final_answer == "Use a quarterly supplier review and monitor concentration."
    assert (second.input_tokens, second.output_tokens) == (80, 20)
    assert payloads[0]["parallel_tool_calls"] is True
    assert payloads[0]["tools"][0]["type"] == "function"
    assert payloads[0]["tools"][0]["parameters"] == tools[0].input_schema
    assert payloads[1]["previous_response_id"] == "resp_first"
    assert {item["call_id"] for item in payloads[1]["input"]} == {
        "model_call_1",
        "model_call_2",
    }
    assert all(item["type"] == "function_call_output" for item in payloads[1]["input"])


@pytest.mark.asyncio
async def test_openai_provider_rejects_invalid_function_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp_invalid",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_bad",
                        "name": "web_search",
                        "arguments": "not-json",
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIResponsesProvider(
            api_key="test-secret",
            model="test-model",
            base_url="https://mock.openai.test/v1",
            timeout_seconds=5,
            http_client=client,
        )
        with pytest.raises(ProviderProtocolError, match="invalid JSON"):
            await provider.decide(
                query="test",
                history=build_messages(),
                available_tools=build_stub_registry(1).specs(),
                prior_results=[],
                step=0,
                run_id="run_invalid",
            )


def test_openai_mode_requires_an_api_key_but_default_stays_deterministic() -> None:
    assert build_provider(Settings()).is_demo is True

    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        build_provider(Settings(llm_provider="openai"))


@pytest.mark.asyncio
async def test_runtime_records_openai_token_metrics() -> None:
    responses = iter(
        [
            {
                "id": "resp_runtime_1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_runtime_1",
                        "name": "web_search",
                        "arguments": '{"query":"market research"}',
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
            {
                "id": "resp_runtime_2",
                "output": [],
                "output_text": "The stub source supports the demo recommendation.",
                "usage": {"input_tokens": 8, "output_tokens": 3},
            },
        ]
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(responses))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIResponsesProvider(
            api_key="test-secret",
            model="test-model",
            base_url="https://mock.openai.test/v1",
            timeout_seconds=5,
            http_client=client,
        )
        store = InMemoryStore()
        runtime = AgentRuntime(
            settings=Settings(run_timeout_seconds=5, tool_timeout_seconds=1),
            provider=provider,
            registry=build_stub_registry(1),
            store=store,
        )
        events = [
            event async for event in runtime.stream(query="market research", conversation_id=None)
        ]

    run = await store.get_run(events[-1].run_id)
    assert run is not None
    assert run.final_answer == "The stub source supports the demo recommendation."
    assert (run.metrics.input_tokens, run.metrics.output_tokens) == (18, 7)
    assert run.metrics.token_usage == 25
