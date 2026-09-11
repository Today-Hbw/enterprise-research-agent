from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import AsyncIterator
from time import perf_counter

from app.agent.provider import LLMProvider
from app.config import Settings
from app.logging_utils import log_json
from app.models import (
    AccessContext,
    Message,
    MessageRole,
    PlanStep,
    PlanStepStatus,
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

logger = logging.getLogger(__name__)


class AgentRuntime:
    _TOOL_CALL_LIMITS = {
        "knowledge_base_list": 1,
        "knowledge_search": 4,
        "web_search": 2,
        "http_fetch": 3,
        "schema_search": 2,
        "execute_sql": 4,
        "python_execute": 2,
        "browser": 2,
        "mcp_invoke": 3,
    }
    _TOOL_FAILURE_LIMITS = {
        # Network tools are expensive and retrying the same unavailable backend rarely helps.
        "web_search": 1,
        "http_fetch": 2,
        "browser": 2,
        "mcp_invoke": 2,
        # SQL may be corrected once after the database returns a typed diagnostic.
        "schema_search": 2,
        "execute_sql": 2,
        "knowledge_base_list": 2,
        "knowledge_search": 2,
        "python_execute": 2,
    }
    _PLAN_COPY = {
        "knowledge_search": (
            "Search internal knowledge",
            "Retrieve authorized enterprise documents relevant to the request.",
        ),
        "web_search": (
            "Discover public sources",
            "Search for authoritative public evidence and candidate URLs.",
        ),
        "http_fetch": (
            "Fetch a known source",
            "Read the allowlisted URL through the bounded HTTP fetcher.",
        ),
        "schema_search": (
            "Discover relevant data schema",
            "Find the approved tables, fields, and relationships needed for analysis.",
        ),
        "execute_sql": (
            "Query structured data",
            "Run the validated read-only query against the approved database scope.",
        ),
        "python_execute": (
            "Calculate deterministic metrics",
            "Use the isolated calculation worker for precise derived values.",
        ),
        "browser": (
            "Inspect an interactive page",
            "Use the allowlisted browser worker for explicitly interactive content.",
        ),
        "mcp_invoke": (
            "Read from an external system",
            "Invoke the approved low-risk MCP tool through the server-side catalog.",
        ),
    }

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
        completed_history = list(conversation.messages)
        ignored_failed_messages = 0
        while completed_history and completed_history[-1].role == MessageRole.USER:
            completed_history.pop()
            ignored_failed_messages += 1
        if ignored_failed_messages:
            logger.info(
                "Conversation %s ignored %d trailing user message(s) from failed runs",
                conversation.conversation_id,
                ignored_failed_messages,
            )
        user_message = Message(role=MessageRole.USER, content=query)
        history = [*completed_history, user_message]
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
        logger.info(
            "Run started: run_id=%s, conversation_id=%s, model=%s, input=%s",
            run.run_id,
            run.conversation_id,
            run.model,
            log_json(
                {
                    "query": query,
                    "tenant_id": run.tenant_id,
                    "principal_ids": sorted(run.principal_ids),
                    "history": [message.model_dump(mode="json") for message in history],
                }
            ),
        )
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
                async for emitted in self._execute(run, history, access_context, event):
                    yield emitted
        except TimeoutError:
            run.status = RunStatus.FAILED
            run.error = f"Run timed out after {self.settings.run_timeout_seconds}s"
            logger.error(
                "Run %s timed out after %ss", run.run_id, self.settings.run_timeout_seconds
            )
        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = str(exc)
            logger.exception("Run %s failed: %s", run.run_id, exc)
        finally:
            await self.provider.finish_run(run.run_id)

        if run.status == RunStatus.FAILED:
            for step in self._fail_incomplete_plan(run, run.error or "Agent run failed"):
                yield event("plan_step_updated", step=step.model_dump(mode="json"))

        run.completed_at = utc_now()
        if run.trace:
            started = run.created_at.timestamp()
            run.metrics.duration_ms = max(0, int((run.completed_at.timestamp() - started) * 1000))
        await self.store.save_run(run)

        if run.status == RunStatus.COMPLETED and run.final_answer:
            logger.info(
                "Run %s completed: duration=%dms, tokens=%d, tools=%d, cost=%s",
                run.run_id,
                run.metrics.duration_ms,
                run.metrics.token_usage,
                run.metrics.tool_call_count,
                f"${run.metrics.estimated_cost:.8f}"
                if run.metrics.estimated_cost is not None
                else "N/A",
            )
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
            logger.warning(
                "Run %s failed: error=%s, duration=%dms",
                run.run_id,
                run.error,
                run.metrics.duration_ms,
            )
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
        tool_call_counts: Counter[str] = Counter()
        tool_failure_counts: Counter[str] = Counter()
        empty_sql_results = 0
        blocked_synthesis_rounds = 0

        for step_index in range(self.settings.max_steps):
            self._enforce_budget(run)
            available_tools = self._available_tools(
                tool_call_counts=tool_call_counts,
                tool_failure_counts=tool_failure_counts,
                empty_sql_results=empty_sql_results,
            )
            synthesis_forced = not available_tools
            if synthesis_forced:
                logger.info(
                    "Run %s has no eligible tools remaining; requesting final synthesis at step %d",
                    run.run_id,
                    step_index + 1,
                )
            decision_started = perf_counter()
            logger.info(
                "Run %s LLM input (step=%d): %s",
                run.run_id,
                step_index + 1,
                log_json(
                    {
                        "query": run.user_query,
                        "available_tools": [spec.name for spec in available_tools],
                        "prior_results": [
                            result.model_dump(mode="json") for result in prior_results
                        ],
                        "force_synthesis": synthesis_forced,
                    }
                ),
            )
            decision = await self.provider.decide(
                query=run.user_query,
                history=history,
                available_tools=available_tools,
                prior_results=prior_results,
                step=step_index,
                run_id=run.run_id,
            )
            decision_duration_ms = int((perf_counter() - decision_started) * 1000)
            run.metrics.llm_call_count += 1
            run.metrics.input_tokens += decision.input_tokens
            run.metrics.output_tokens += decision.output_tokens
            run.metrics.token_usage += decision.input_tokens + decision.output_tokens
            self._update_estimated_cost(run)
            logger.info(
                "Run %s LLM decision (step=%d): %d input tokens, %d output tokens, %dms, "
                "output=%s",
                run.run_id,
                step_index + 1,
                decision.input_tokens,
                decision.output_tokens,
                decision_duration_ms,
                log_json(decision.model_dump(mode="json")),
            )
            run.trace.append(
                TraceStep(
                    index=len(run.trace),
                    kind="agent_decision",
                    summary=decision.decision_summary,
                    duration_ms=decision_duration_ms,
                )
            )
            yield event_factory(
                "agent_decision", step=step_index + 1, summary=decision.decision_summary
            )

            if decision.final_answer is not None:
                synthesis = self._ensure_synthesis_step(run)
                synthesis.status = PlanStepStatus.RUNNING
                await self.store.save_run(run)
                yield event_factory(
                    "plan_updated", plan=[step.model_dump(mode="json") for step in run.plan]
                )
                yield event_factory("plan_step_updated", step=synthesis.model_dump(mode="json"))
                synthesis.status = PlanStepStatus.COMPLETED
                synthesis.error = None
                await self.store.save_run(run)
                yield event_factory("plan_step_updated", step=synthesis.model_dump(mode="json"))
                self._mark_exhausted_budget(run)
                run.final_answer = decision.final_answer
                run.status = RunStatus.COMPLETED
                run.sources = self._deduplicate_sources(prior_results)
                logger.info(
                    "Run %s final output: %s",
                    run.run_id,
                    log_json(
                        {
                            "answer": run.final_answer,
                            "sources": [
                                source.model_dump(mode="json") for source in run.sources
                            ],
                        }
                    ),
                )
                yield event_factory("assistant_delta", content=decision.final_answer)
                return

            if synthesis_forced:
                blocked_synthesis_rounds += 1
                logger.warning(
                    "Run %s provider returned %d tool call(s) when no eligible tools remained; "
                    "returning policy-blocked outputs",
                    run.run_id,
                    len(decision.tool_calls),
                )
                if blocked_synthesis_rounds > 1:
                    raise RuntimeError(
                        "Model repeatedly requested tools after all eligible tools were exhausted"
                    )
                blocked_results = [
                    ToolResult(
                        call_id=call.call_id,
                        tool_name=call.name,
                        success=False,
                        summary="Tool call blocked because no eligible tools remain.",
                        error=(
                            "Server safety limits require the model to synthesize the final answer."
                        ),
                    )
                    for call in decision.tool_calls
                ]
                prior_results.extend(blocked_results)
                for call, result in zip(
                    decision.tool_calls, blocked_results, strict=True
                ):
                    logger.info(
                        "Run %s tool %s blocked: call_id=%s, input=%s, output=%s",
                        run.run_id,
                        call.name,
                        call.call_id,
                        log_json(call.arguments),
                        log_json(result.model_dump(mode="json")),
                    )
                    run.trace.append(
                        TraceStep(
                            index=len(run.trace),
                            kind="tool_policy_block",
                            summary=result.summary,
                            tool_name=call.name,
                            tool_input=call.arguments,
                            tool_output_summary=result.summary,
                            status="blocked",
                            error=result.error,
                        )
                    )
                    yield event_factory(
                        "tool_blocked",
                        call_id=call.call_id,
                        tool_name=call.name,
                        input=call.arguments,
                        summary=result.summary,
                        reason=result.error,
                    )
                await self.store.save_run(run)
                continue

            repeated_calls = [
                call for call in decision.tool_calls if call.signature in call_signatures
            ]
            candidate_calls = [
                call for call in decision.tool_calls if call.signature not in call_signatures
            ]
            if not candidate_calls:
                logger.warning(
                    "Run %s: repeated tool call detected at step %d", run.run_id, step_index
                )
                raise RuntimeError("Repeated tool call detected; agent stopped to prevent a loop")

            eligible_names = {spec.name for spec in available_tools}
            accepted_by_tool: Counter[str] = Counter()
            fresh_calls: list[ToolCall] = []
            blocked_calls: list[tuple[ToolCall, ToolResult]] = []
            for call in repeated_calls:
                blocked_calls.append(
                    (
                        call,
                        ToolResult(
                            call_id=call.call_id,
                            tool_name=call.name,
                            success=False,
                            summary="Repeated tool call was not executed.",
                            error="The same tool request already ran during this agent run.",
                        ),
                    )
                )
            for call in candidate_calls:
                call_limit = self._TOOL_CALL_LIMITS.get(call.name, 3)
                remaining_calls = max(0, call_limit - tool_call_counts[call.name])
                if (
                    call.name not in eligible_names
                    or accepted_by_tool[call.name] >= remaining_calls
                ):
                    blocked_calls.append(
                        (
                            call,
                            ToolResult(
                                call_id=call.call_id,
                                tool_name=call.name,
                                success=False,
                                summary="Tool call was not eligible for execution.",
                                error=(
                                    "The tool is unavailable at this step because its call, "
                                    "failure, "
                                    "or result limit has been reached."
                                ),
                            ),
                        )
                    )
                    continue
                accepted_by_tool[call.name] += 1
                fresh_calls.append(call)

            if blocked_calls:
                prior_results.extend(result for _, result in blocked_calls)
                for call, result in blocked_calls:
                    logger.info(
                        "Run %s tool %s blocked: call_id=%s, input=%s, output=%s",
                        run.run_id,
                        call.name,
                        call.call_id,
                        log_json(call.arguments),
                        log_json(result.model_dump(mode="json")),
                    )
                    run.trace.append(
                        TraceStep(
                            index=len(run.trace),
                            kind="tool_policy_block",
                            summary=result.summary,
                            tool_name=call.name,
                            tool_input=call.arguments,
                            tool_output_summary=result.summary,
                            status="blocked",
                            error=result.error,
                        )
                    )
                    yield event_factory(
                        "tool_blocked",
                        call_id=call.call_id,
                        tool_name=call.name,
                        input=call.arguments,
                        summary=result.summary,
                        reason=result.error,
                    )
                await self.store.save_run(run)

            if not fresh_calls:
                continue
            for call in fresh_calls:
                call_signatures.add(call.signature)
            logger.debug(
                "Run %s: %d fresh tool call(s): %s",
                run.run_id,
                len(fresh_calls),
                [f"{c.name}({list(c.arguments.keys())})" for c in fresh_calls],
            )

            plan_event = self._sync_plan(run, fresh_calls)
            await self.store.save_run(run)
            yield event_factory(
                plan_event, plan=[step.model_dump(mode="json") for step in run.plan]
            )

            self._enforce_budget(run)

            semaphore = asyncio.Semaphore(self.settings.max_parallel_tools)

            running_steps = []
            for call in fresh_calls:
                plan_step = self._plan_step_for_call(run, call.call_id)
                plan_step.status = PlanStepStatus.RUNNING
                plan_step.error = None
                running_steps.append(plan_step)
            await self.store.save_run(run)

            for call, plan_step in zip(fresh_calls, running_steps, strict=True):
                logger.info(
                    "Run %s tool %s input: call_id=%s, input=%s",
                    run.run_id,
                    call.name,
                    call.call_id,
                    log_json(call.arguments),
                )
                yield event_factory("plan_step_updated", step=plan_step.model_dump(mode="json"))
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
                tool_call_counts[call.name] += 1
                if result.success:
                    tool_failure_counts[call.name] = 0
                else:
                    tool_failure_counts[call.name] += 1
                if call.name == "execute_sql" and result.success:
                    rows = result.data.get("rows")
                    empty_sql_results = (
                        empty_sql_results + 1 if isinstance(rows, list) and not rows else 0
                    )
                run.metrics.tool_call_count += 1
                logger.info(
                    "Run %s tool %s output: call_id=%s, status=%s, duration_ms=%d, output=%s",
                    run.run_id,
                    call.name,
                    call.call_id,
                    "completed" if result.success else "failed",
                    duration_ms,
                    log_json(result.model_dump(mode="json")),
                )
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
                plan_step = self._plan_step_for_call(run, call.call_id)
                plan_step.status = (
                    PlanStepStatus.COMPLETED if result.success else PlanStepStatus.FAILED
                )
                plan_step.error = result.error
                yield event_factory("plan_step_updated", step=plan_step.model_dump(mode="json"))
            await self.store.save_run(run)

        logger.warning("Run %s reached max_steps=%d", run.run_id, self.settings.max_steps)
        raise RuntimeError(f"Agent reached max_steps={self.settings.max_steps}")

    def _available_tools(
        self,
        *,
        tool_call_counts: Counter[str],
        tool_failure_counts: Counter[str],
        empty_sql_results: int,
    ):
        available = []
        for spec in self.registry.specs():
            call_limit = self._TOOL_CALL_LIMITS.get(spec.name, 3)
            failure_limit = self._TOOL_FAILURE_LIMITS.get(spec.name, 2)
            if tool_call_counts[spec.name] >= call_limit:
                continue
            if tool_failure_counts[spec.name] >= failure_limit:
                continue
            if spec.name == "execute_sql" and empty_sql_results >= 2:
                continue
            available.append(spec)
        return available

    def _sync_plan(self, run: RunRecord, calls: list[ToolCall]) -> str | None:
        created = not run.plan
        known_call_ids = {step.call_id for step in run.plan if step.call_id}
        additions = []
        for call in calls:
            if call.call_id in known_call_ids:
                raise RuntimeError(f"Duplicate tool call ID in run plan: {call.call_id}")
            known_call_ids.add(call.call_id)
            title, description = self._PLAN_COPY.get(
                call.name,
                (
                    f"Run {call.name.replace('_', ' ')}",
                    (
                        "Execute the selected tool within its configured permission "
                        "and timeout limits."
                    ),
                ),
            )
            additions.append(
                PlanStep(
                    index=0,
                    title=title,
                    description=description,
                    tool_name=call.name,
                    call_id=call.call_id,
                )
            )
        if additions:
            run.plan.extend(additions)
        for index, step in enumerate(run.plan):
            step.index = index
        if created:
            return "plan_created"
        return "plan_updated" if additions else None

    @staticmethod
    def _ensure_synthesis_step(run: RunRecord) -> PlanStep:
        synthesis = next((step for step in run.plan if step.tool_name is None), None)
        if synthesis is None:
            synthesis = PlanStep(
                index=len(run.plan),
                title="Synthesize the final answer",
                description="Combine validated evidence into a cited, traceable response.",
            )
            run.plan.append(synthesis)
        return synthesis

    @staticmethod
    def _plan_step_for_call(run: RunRecord, call_id: str) -> PlanStep:
        return next(step for step in run.plan if step.call_id == call_id)

    @staticmethod
    def _fail_incomplete_plan(run: RunRecord, error: str) -> list[PlanStep]:
        changed = []
        for step in run.plan:
            if step.status in {PlanStepStatus.PENDING, PlanStepStatus.RUNNING}:
                step.status = PlanStepStatus.FAILED
                step.error = error
                changed.append(step)
        return changed

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
