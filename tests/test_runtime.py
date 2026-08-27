import pytest

from app.agent.provider import DeterministicProvider, LLMProvider
from app.agent.runtime import AgentRuntime
from app.config import Settings
from app.models import AgentDecision, RunStatus, ToolCall
from app.store import InMemoryStore
from app.tools.stubs import build_stub_registry


def build_runtime() -> tuple[AgentRuntime, InMemoryStore]:
    settings = Settings(
        max_steps=8,
        run_timeout_seconds=5,
        tool_timeout_seconds=1,
        max_parallel_tools=4,
    )
    memory = InMemoryStore()
    runtime = AgentRuntime(
        settings=settings,
        provider=DeterministicProvider(),
        registry=build_stub_registry(1),
        store=memory,
    )
    return runtime, memory


@pytest.mark.asyncio
async def test_complex_research_run_completes_with_trace_and_sources() -> None:
    runtime, memory = build_runtime()
    events = [
        event
        async for event in runtime.stream(
            query="调研市场并结合采购数据做分析", conversation_id=None
        )
    ]

    assert events[0].event == "run_started"
    assert events[-1].event == "run_completed"
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))

    run = await memory.get_run(events[-1].run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED
    assert run.metrics.tool_call_count == 5
    assert run.metrics.llm_call_count == 4
    assert {step.tool_name for step in run.trace if step.tool_name} >= {
        "knowledge_search",
        "web_search",
        "schema_search",
        "execute_sql",
        "python_execute",
    }
    assert len(run.sources) == 5
    assert "placeholder" in run.final_answer


@pytest.mark.asyncio
async def test_browser_is_only_selected_for_interactive_intent() -> None:
    runtime, _ = build_runtime()
    plain_events = [
        event async for event in runtime.stream(query="研究公开市场", conversation_id=None)
    ]
    interactive_events = [
        event
        async for event in runtime.stream(query="登录 SaaS 后台并点击导出", conversation_id=None)
    ]

    assert not any(event.data.get("tool_name") == "browser" for event in plain_events)
    assert any(event.data.get("tool_name") == "browser" for event in interactive_events)


class RepeatingProvider(LLMProvider):
    name = "repeating-test-provider"

    async def decide(self, **_kwargs) -> AgentDecision:
        return AgentDecision(
            tool_calls=[ToolCall(name="web_search", arguments={"query": "same"})],
            decision_summary="Repeat the same tool call.",
        )


@pytest.mark.asyncio
async def test_repeated_tool_call_fails_run_instead_of_looping() -> None:
    settings = Settings(max_steps=8, run_timeout_seconds=5, tool_timeout_seconds=1)
    memory = InMemoryStore()
    runtime = AgentRuntime(
        settings=settings,
        provider=RepeatingProvider(),
        registry=build_stub_registry(1),
        store=memory,
    )

    events = [event async for event in runtime.stream(query="repeat", conversation_id=None)]
    run = await memory.get_run(events[-1].run_id)

    assert events[-1].event == "run_failed"
    assert run is not None
    assert run.status == RunStatus.FAILED
    assert "Repeated tool call" in run.error
