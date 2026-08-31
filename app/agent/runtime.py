from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from time import perf_counter

from app.agent.provider import LLMProvider
from app.config import Settings
from app.models import (
    AccessContext,
    Message,
    MessageRole,
    RunBudget,
    RunRecord,
    RunStatus,
    StreamEvent,
    ToolCall,
    ToolResult,
    TraceStep,
    utc_now,
)
from app.store import InMemoryStore
from app.tools.registry import ToolRegistry


class AgentRuntime:
    def __init__(
        self,
        *,
        settings: Settings,
        provider: LLMProvider,
        registry: ToolRegistry,
        store: InMemoryStore,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.registry = registry
        self.store = store

    async def stream(
        self,
        *,
        query: str,
        conversation_id: str | None,
        access_context: AccessContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        access_context = access_context or AccessContext(
            tenant_id=self.settings.knowledge_default_tenant,
            principal_ids={self.settings.knowledge_default_principal},
        )
        conversation = await self.store.get_or_create_conversation(
            conversation_id,
            query,
            access_context.tenant_id,
            access_context.principal_ids,
        )
        user_message = Message(role=MessageRole.USER, content=query)
        await self.store.add_message(conversation.conversation_id, user_message)

        run = RunRecord(
            conversation_id=conversation.conversation_id,
            tenant_id=access_context.tenant_id,
            principal_ids=access_context.principal_ids,
            user_query=query,
            model=self.provider.name,
            budget=RunBudget(
                token_limit=self.settings.run_token_budget,
                cost_limit=self.settings.run_cost_budget_usd,
            ),
        )
        await self.store.save_run(run)
        sequence = 0

        def event(name: str, **data: object) -> StreamEvent:
            nonlocal sequence
            sequence += 1
            return StreamEvent(event=name, sequence=sequence, run_id=run.run_id, data=data)

        yield event(
            "run_started",
            conversation_id=conversation.conversation_id,
            model=run.model,
            is_demo=self.provider.is_demo,
        )

        try:
            async with asyncio.timeout(self.settings.run_timeout_seconds):
                async for emitted in self._execute(
                    run, conversation.messages, access_context, event
                ):
                    yield emitted
        except TimeoutError:
            run.status = RunStatus.FAILED
            run.error = f"Run timed out after {self.settings.run_timeout_seconds}s"
        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = str(exc)
        finally:
            await self.provider.finish_run(run.run_id)

        run.completed_at = utc_now()
        if run.trace:
            started = run.created_at.timestamp()
            run.metrics.duration_ms = max(0, int((run.completed_at.timestamp() - started) * 1000))
        await self.store.save_run(run)

        if run.status == RunStatus.COMPLETED and run.final_answer:
            await self.store.add_message(
                conversation.conversation_id,
                Message(
                    role=MessageRole.ASSISTANT,
                    content=run.final_answer,
                    run_id=run.run_id,
                ),
            )
            yield event("run_completed", run=run.model_dump(mode="json"))
        else:
            yield event(
                "run_failed",
                error=run.error or "Agent run failed",
                run=run.model_dump(mode="json"),
            )

    async def _execute(
        self,
        run: RunRecord,
        history: list[Message],
        access_context: AccessContext,
        event_factory,
    ) -> AsyncIterator[StreamEvent]:
        prior_results: list[ToolResult] = []
        call_signatures: set[str] = set()

        for step_index in range(self.settings.max_steps):
            self._enforce_budget(run)
            decision_started = perf_counter()
            decision = await self.provider.decide(
                query=run.user_query,
                history=history,
                available_tools=self.registry.specs(),
                prior_results=prior_results,
                step=step_index,
                run_id=run.run_id,
            )
            run.metrics.llm_call_count += 1
            run.metrics.input_tokens += decision.input_tokens
            run.metrics.output_tokens += decision.output_tokens
            run.metrics.token_usage += decision.input_tokens + decision.output_tokens
            self._update_estimated_cost(run)
            run.trace.append(
                TraceStep(
                    index=len(run.trace),
                    kind="agent_decision",
                    summary=decision.decision_summary,
                    duration_ms=int((perf_counter() - decision_started) * 1000),
                )
            )
            yield event_factory(
                "agent_decision", step=step_index + 1, summary=decision.decision_summary
            )

            if decision.final_answer is not None:
                self._mark_exhausted_budget(run)
                run.final_answer = decision.final_answer
                run.status = RunStatus.COMPLETED
                run.sources = self._deduplicate_sources(prior_results)
                yield event_factory("assistant_delta", content=decision.final_answer)
                return

            self._enforce_budget(run)
            fresh_calls = [
                call for call in decision.tool_calls if call.signature not in call_signatures
            ]
            if not fresh_calls:
                raise RuntimeError("Repeated tool call detected; agent stopped to prevent a loop")
            for call in fresh_calls:
                call_signatures.add(call.signature)

            semaphore = asyncio.Semaphore(self.settings.max_parallel_tools)

            for call in fresh_calls:
                yield event_factory(
                    "tool_started",
                    call_id=call.call_id,
                    tool_name=call.name,
                    input=call.arguments,
                )

            executed = await asyncio.gather(
                *(self._execute_tool(call, semaphore, access_context) for call in fresh_calls)
            )
            for call, result, duration_ms in executed:
                prior_results.append(result)
                run.metrics.tool_call_count += 1
                run.trace.append(
                    TraceStep(
                        index=len(run.trace),
                        kind="tool_call",
                        summary=f"{call.name} {'completed' if result.success else 'failed'}.",
                        tool_name=call.name,
                        tool_input=call.arguments,
                        tool_output_summary=result.summary,
                        status="completed" if result.success else "failed",
                        duration_ms=duration_ms,
                        error=result.error,
                    )
                )
                yield event_factory(
                    "tool_completed",
                    call_id=call.call_id,
                    tool_name=call.name,
                    success=result.success,
                    summary=result.summary,
                    sources=[source.model_dump(mode="json") for source in result.sources],
                    duration_ms=duration_ms,
                    is_stub=self.registry.is_stub(call.name),
                )

        raise RuntimeError(f"Agent reached max_steps={self.settings.max_steps}")

    def _update_estimated_cost(self, run: RunRecord) -> None:
        input_rate = self.settings.llm_input_cost_per_million_tokens
        output_rate = self.settings.llm_output_cost_per_million_tokens
        if input_rate is None and output_rate is None:
            run.metrics.estimated_cost = None
            return
        cost = (
            run.metrics.input_tokens * (input_rate or 0)
            + run.metrics.output_tokens * (output_rate or 0)
        ) / 1_000_000
        run.metrics.estimated_cost = round(cost, 8)

    @staticmethod
    def _budget_reason(run: RunRecord) -> str | None:
        if run.budget.token_limit is not None and run.metrics.token_usage >= run.budget.token_limit:
            return (
                f"Run token budget exhausted: {run.metrics.token_usage}/"
                f"{run.budget.token_limit} tokens"
            )
        if (
            run.budget.cost_limit is not None
            and run.metrics.estimated_cost is not None
            and run.metrics.estimated_cost >= run.budget.cost_limit
        ):
            return (
                f"Run cost budget exhausted: ${run.metrics.estimated_cost:.8f}/"
                f"${run.budget.cost_limit:.8f}"
            )
        return None

    def _mark_exhausted_budget(self, run: RunRecord) -> None:
        reason = self._budget_reason(run)
        if reason:
            run.metrics.budget_exhausted = True
            run.metrics.budget_reason = reason

    def _enforce_budget(self, run: RunRecord) -> None:
        self._mark_exhausted_budget(run)
        if run.metrics.budget_reason:
            raise RuntimeError(run.metrics.budget_reason)

    async def _execute_tool(
        self,
        call: ToolCall,
        semaphore: asyncio.Semaphore,
        access_context: AccessContext,
    ) -> tuple[ToolCall, ToolResult, int]:
        started = perf_counter()
        async with semaphore:
            result = await self.registry.execute(call, access_context)
        return call, result, int((perf_counter() - started) * 1000)

    @staticmethod
    def _deduplicate_sources(results: list[ToolResult]):
        unique = {}
        for result in results:
            for source in result.sources:
                key = (
                    source.source_type,
                    source.title,
                    source.url,
                    source.document_id,
                    source.chunk_id,
                )
                anchor = source.url or source.chunk_id or source.document_id or source.source_id
                unique.setdefault(key, source.model_copy(update={"evidence_anchor": anchor}))
        return sorted(
            unique.values(),
            key=lambda source: (
                source.source_type.value,
                source.evidence_anchor or "",
                source.title,
            ),
        )
