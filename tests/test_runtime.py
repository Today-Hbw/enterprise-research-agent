from collections import Counter

import pytest

from app.agent.provider import DeterministicProvider, LLMProvider
from app.agent.runtime import AgentRuntime
from app.config import Settings
from app.models import (
    AgentDecision,
    PlanStepStatus,
    RunStatus,
    Source,
    SourceType,
    ToolCall,
    ToolResult,
)
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
    assert [event.event for event in events].count("plan_updated") == 3
    plan_created = next(event for event in events if event.event == "plan_created")
    assert all(step["tool_name"] is not None for step in plan_created.data["plan"])
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
    synthesis_added = next(
        event
        for event in events
        if event.event == "plan_updated"
        and any(step["tool_name"] is None for step in event.data["plan"])
    )
    assistant_delta = next(event for event in events if event.event == "assistant_delta")
    assert events.index(synthesis_added) < events.index(assistant_delta)


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
    assert [step.status for step in run.plan] == [PlanStepStatus.COMPLETED]


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


class CapturingProvider(LLMProvider):
    name = "capturing-test-provider"

    def __init__(self) -> None:
        self.history = []

    async def decide(self, **kwargs) -> AgentDecision:
        self.history = kwargs["history"]
        return AgentDecision(final_answer="ok", decision_summary="Captured history.")


class DetachedMessageStore(InMemoryStore):
    async def add_message(self, conversation_id, message) -> None:
        async with self._lock:
            conversation = self._conversations[conversation_id].model_copy(deep=True)
            conversation.messages.append(message)
            self._conversations[conversation_id] = conversation


@pytest.mark.asyncio
async def test_runtime_passes_new_user_message_when_store_returns_detached_objects() -> None:
    provider = CapturingProvider()
    runtime = AgentRuntime(
        settings=Settings(max_steps=2, run_timeout_seconds=5, tool_timeout_seconds=1),
        provider=provider,
        registry=build_stub_registry(1),
        store=DetachedMessageStore(),
    )

    events = [event async for event in runtime.stream(query="live query", conversation_id=None)]

    assert events[-1].event == "run_completed"
    assert [message.content for message in provider.history] == ["live query"]


class EvidenceThenSynthesisProvider(LLMProvider):
    name = "evidence-then-synthesis-provider"

    def __init__(self) -> None:
        self.available_tool_names: list[list[str]] = []

    async def decide(self, **kwargs) -> AgentDecision:
        available = [spec.name for spec in kwargs["available_tools"]]
        self.available_tool_names.append(available)
        if not kwargs["prior_results"]:
            return AgentDecision(
                tool_calls=[
                    ToolCall(name="knowledge_search", arguments={"query": "policy"}),
                    ToolCall(name="web_search", arguments={"query": "policy"}),
                ],
                decision_summary="Collect evidence.",
            )
        if not available:
            return AgentDecision(final_answer="Enough evidence.", decision_summary="Synthesize.")
        return AgentDecision(
            tool_calls=[ToolCall(name="schema_search", arguments={"query": "unneeded"})],
            decision_summary="Keep exploring.",
        )


@pytest.mark.asyncio
async def test_live_runtime_forces_synthesis_after_sufficient_evidence() -> None:
    provider = EvidenceThenSynthesisProvider()
    memory = InMemoryStore()
    runtime = AgentRuntime(
        settings=Settings(max_steps=32, run_timeout_seconds=5, tool_timeout_seconds=1),
        provider=provider,
        registry=build_stub_registry(1),
        store=memory,
    )

    events = [event async for event in runtime.stream(query="成都参保资料", conversation_id=None)]
    run = await memory.get_run(events[-1].run_id)

    assert events[-1].event == "run_completed"
    assert run is not None
    assert run.metrics.tool_call_count == 2
    assert provider.available_tool_names[0]
    assert provider.available_tool_names[1] == []


class SynthesisPolicyIgnoringProvider(LLMProvider):
    name = "synthesis-policy-ignoring-provider"

    async def decide(self, **kwargs) -> AgentDecision:
        results = kwargs["prior_results"]
        if not results:
            return AgentDecision(
                tool_calls=[
                    ToolCall(name="web_search", arguments={"query": "policy"}),
                    ToolCall(name="knowledge_search", arguments={"query": "policy"}),
                ],
                decision_summary="Collect evidence.",
            )
        if results[-1].summary.startswith("Tool call blocked"):
            return AgentDecision(final_answer="Synthesized.", decision_summary="Synthesize.")
        return AgentDecision(
            tool_calls=[ToolCall(name="knowledge_search", arguments={"query": "more"})],
            decision_summary="Incorrectly request another tool.",
        )


@pytest.mark.asyncio
async def test_runtime_recovers_when_provider_ignores_forced_synthesis_once() -> None:
    memory = InMemoryStore()
    runtime = AgentRuntime(
        settings=Settings(max_steps=8, run_timeout_seconds=5, tool_timeout_seconds=1),
        provider=SynthesisPolicyIgnoringProvider(),
        registry=build_stub_registry(1),
        store=memory,
    )

    events = [event async for event in runtime.stream(query="成都参保资料", conversation_id=None)]
    run = await memory.get_run(events[-1].run_id)

    assert events[-1].event == "run_completed"
    assert run is not None
    assert run.final_answer == "Synthesized."
    assert run.metrics.tool_call_count == 2


def test_sql_intent_requires_nonempty_sql_before_forcing_synthesis() -> None:
    knowledge_result = ToolResult(
        call_id="knowledge",
        tool_name="knowledge_search",
        success=True,
        summary="Found policy.",
        sources=[
            Source(
                source_type=SourceType.DOCUMENT,
                title=f"Document {index}",
                document_id=f"doc-{index}",
                content_snippet="Evidence",
            )
            for index in range(3)
        ],
    )
    empty_sql_result = ToolResult(
        call_id="sql-empty",
        tool_name="execute_sql",
        success=True,
        summary="No rows.",
        data={"rows": []},
    )
    nonempty_sql_result = empty_sql_result.model_copy(
        update={
            "call_id": "sql-result",
            "data": {"rows": [[1]]},
            "sources": [
                Source(
                    source_type=SourceType.SQL,
                    title="Query result",
                    content_snippet="One row.",
                )
            ],
        }
    )

    assert not AgentRuntime._has_sufficient_evidence(
        "统计成都参保人数", [knowledge_result, empty_sql_result]
    )
    assert AgentRuntime._has_sufficient_evidence(
        "统计成都参保人数", [knowledge_result, nonempty_sql_result]
    )


def test_failed_web_search_and_repeated_empty_sql_are_removed_from_catalog() -> None:
    runtime, _ = build_runtime()
    available = runtime._available_tools(
        tool_call_counts=Counter(),
        tool_failure_counts=Counter({"web_search": 1}),
        empty_sql_results=2,
        force_synthesis=False,
    )
    names = {spec.name for spec in available}

    assert "web_search" not in names
    assert "execute_sql" not in names
