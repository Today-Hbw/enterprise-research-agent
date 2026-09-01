import pytest

from app.agent.provider import DeterministicProvider, LLMProvider
from app.agent.runtime import AgentRuntime
from app.config import Settings
from app.models import AgentDecision, PlanStepStatus, RunStatus, ToolCall
from app.store import InMemoryStore
from app.tools.stubs import build_stub_registry


def test_run_timeout_defaults_to_120_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUN_TIMEOUT_SECONDS", raising=False)

    assert Settings(_env_file=None).run_timeout_seconds == 120


def test_run_timeout_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_TIMEOUT_SECONDS", "180")

    assert Settings(_env_file=None).run_timeout_seconds == 180


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
    assert [event.event for event in events].count("plan_created") == 1
    assert [event.event for event in events].count("plan_updated") == 2
    plan_created = next(event for event in events if event.event == "plan_created")
    assert events.index(plan_created) < next(
        i for i, event in enumerate(events) if event.event == "tool_started"
    )
    web_step = next(step for step in plan_created.data["plan"] if step["tool_name"] == "web_search")
    assert [
        event.data["step"]["status"]
        for event in events
        if event.event == "plan_step_updated"
        and event.data["step"]["step_id"] == web_step["step_id"]
    ] == ["running", "completed"]

    run = await memory.get_run(events[-1].run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED
    assert run.metrics.tool_call_count == 5
    assert run.metrics.llm_call_count == 4
    assert [step.index for step in run.plan] == list(range(len(run.plan)))
    assert [step.tool_name for step in run.plan] == [
        "knowledge_search",
        "web_search",
        "schema_search",
        "execute_sql",
        "python_execute",
        None,
    ]
    assert all(step.status == PlanStepStatus.COMPLETED for step in run.plan)
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
    assert events[-1].data["run"]["status"] == "failed"
    assert run is not None
    assert run.status == RunStatus.FAILED
    assert "Repeated tool call" in run.error
    assert [step.status for step in run.plan] == [
        PlanStepStatus.COMPLETED,
        PlanStepStatus.FAILED,
    ]


class TokenHungryProvider(LLMProvider):
    name = "token-hungry-test-provider"

    async def decide(self, **_kwargs) -> AgentDecision:
        return AgentDecision(
            tool_calls=[ToolCall(name="web_search", arguments={"query": "budget"})],
            decision_summary="A provider response consumed the remaining token budget.",
            input_tokens=10,
            output_tokens=5,
        )


class DuplicateCallIdProvider(LLMProvider):
    name = "duplicate-call-id-test-provider"

    async def decide(self, **_kwargs) -> AgentDecision:
        return AgentDecision(
            tool_calls=[
                ToolCall(call_id="call_duplicate", name="knowledge_search"),
                ToolCall(call_id="call_duplicate", name="web_search"),
            ],
            decision_summary="Provider returned conflicting tool call identifiers.",
        )


@pytest.mark.asyncio
async def test_duplicate_tool_call_ids_fail_without_executing_an_ambiguous_plan() -> None:
    memory = InMemoryStore()
    runtime = AgentRuntime(
        settings=Settings(max_steps=2, run_timeout_seconds=5, tool_timeout_seconds=1),
        provider=DuplicateCallIdProvider(),
        registry=build_stub_registry(1),
        store=memory,
    )

    events = [event async for event in runtime.stream(query="duplicate", conversation_id=None)]
    run = await memory.get_run(events[-1].run_id)

    assert run is not None
    assert run.status == RunStatus.FAILED
    assert run.metrics.tool_call_count == 0
    assert "Duplicate tool call ID" in (run.error or "")


@pytest.mark.asyncio
async def test_token_budget_stops_run_before_tools_or_another_llm_call() -> None:
    settings = Settings(
        max_steps=8,
        run_timeout_seconds=5,
        run_token_budget=12,
        tool_timeout_seconds=1,
    )
    memory = InMemoryStore()
    runtime = AgentRuntime(
        settings=settings,
        provider=TokenHungryProvider(),
        registry=build_stub_registry(1),
        store=memory,
    )

    events = [event async for event in runtime.stream(query="budget", conversation_id=None)]
    run = await memory.get_run(events[-1].run_id)

    assert events[-1].event == "run_failed"
    assert events[-1].data["run"]["metrics"]["budget_exhausted"] is True
    assert run is not None
    assert run.status == RunStatus.FAILED
    assert run.metrics.token_usage == 15
    assert run.metrics.tool_call_count == 0
    assert run.metrics.budget_exhausted is True
    assert run.metrics.budget_reason == "Run token budget exhausted: 15/12 tokens"
    assert run.budget.token_limit == 12
    assert run.plan
    assert all(step.status == PlanStepStatus.FAILED for step in run.plan)


@pytest.mark.asyncio
async def test_cost_budget_uses_configured_rates_and_stops_the_run() -> None:
    settings = Settings(
        max_steps=8,
        run_timeout_seconds=5,
        run_cost_budget_usd=0.000019,
        tool_timeout_seconds=1,
        llm_input_cost_per_million_tokens=1,
        llm_output_cost_per_million_tokens=2,
    )
    memory = InMemoryStore()
    runtime = AgentRuntime(
        settings=settings,
        provider=TokenHungryProvider(),
        registry=build_stub_registry(1),
        store=memory,
    )

    events = [event async for event in runtime.stream(query="budget", conversation_id=None)]
    run = await memory.get_run(events[-1].run_id)

    assert events[-1].event == "run_failed"
    assert events[-1].data["run"]["metrics"]["estimated_cost"] == 0.00002
    assert run is not None
    assert run.metrics.estimated_cost == 0.00002
    assert run.metrics.budget_exhausted is True
    assert run.metrics.budget_reason == "Run cost budget exhausted: $0.00002000/$0.00001900"
