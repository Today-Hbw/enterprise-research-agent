from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings
from app.models import AgentDecision, Message, ToolCall, ToolResult, ToolSpec


class ProviderConfigurationError(ValueError):
    """Raised when a configured model provider cannot be initialized safely."""


class ProviderProtocolError(RuntimeError):
    """Raised when a provider response cannot be mapped to the agent contract."""


@dataclass
class _ResponsesRunState:
    previous_response_id: str
    pending_call_ids: set[str]


class LLMProvider(ABC):
    name: str
    is_demo: bool = False

    @abstractmethod
    async def decide(
        self,
        *,
        query: str,
        history: list[Message],
        available_tools: list[ToolSpec],
        prior_results: list[ToolResult],
        step: int,
        run_id: str,
    ) -> AgentDecision:
        raise NotImplementedError

    async def finish_run(self, run_id: str) -> None:
        """Release run-scoped state after a completed, failed, or timed-out run."""

        del run_id

    async def aclose(self) -> None:
        """Close any provider-owned network resources during application shutdown."""

        return None


class DeterministicProvider(LLMProvider):
    """Offline planner used to exercise the runtime without an external LLM."""

    name = "deterministic-demo-provider"
    is_demo = True

    async def decide(
        self,
        *,
        query: str,
        history: list[Message],
        available_tools: list[ToolSpec],
        prior_results: list[ToolResult],
        step: int,
        run_id: str,
    ) -> AgentDecision:
        del history, available_tools, run_id
        normalized = query.lower()
        completed = {result.tool_name for result in prior_results}

        if step == 0:
            calls = [ToolCall(name="knowledge_search", arguments={"query": query})]
            if any(token in normalized for token in ("http", "url", "api", "链接", "网址")):
                calls.append(ToolCall(name="http_fetch", arguments={"url": "https://example.com"}))
            else:
                calls.append(ToolCall(name="web_search", arguments={"query": query}))

            if any(
                token in normalized
                for token in ("采购", "数据", "sql", "database", "revenue", "成本", "分析")
            ):
                calls.append(ToolCall(name="schema_search", arguments={"query": query}))
            if any(
                token in normalized
                for token in ("登录", "点击", "后台", "browser", "dashboard", "导出")
            ):
                calls.append(ToolCall(name="browser", arguments={"task": query}))
            if any(token in normalized for token in ("mcp", "notion", "jira", "github")):
                calls.append(ToolCall(name="mcp_invoke", arguments={"query": query}))
            return AgentDecision(
                tool_calls=calls,
                decision_summary="Selected independent discovery tools for parallel execution.",
            )

        if "schema_search" in completed and "execute_sql" not in completed:
            return AgentDecision(
                tool_calls=[
                    ToolCall(
                        name="execute_sql",
                        arguments={
                            "sql": (
                                "SELECT SUM(amount), MAX(supplier_share) FROM fact_purchase_orders"
                            )
                        },
                    )
                ],
                decision_summary="Schema discovery completed; run a read-only aggregate query.",
            )

        if "execute_sql" in completed and "python_execute" not in completed:
            return AgentDecision(
                tool_calls=[
                    ToolCall(
                        name="python_execute",
                        arguments={"expression": "(1240000 - 1100000) / 1100000"},
                    )
                ],
                decision_summary="Structured data is available; calculate deterministic metrics.",
            )

        successful = [result for result in prior_results if result.success]
        evidence = (
            "\n".join(f"- {result.tool_name}: {result.summary}" for result in successful)
            or "- No tool returned usable evidence."
        )
        citations = " ".join(
            f"[{index}]" for index, result in enumerate(successful, start=1) if result.sources
        )
        answer = (
            "## Demo research result\n\n"
            "This response was produced by the complete agent workflow, but every external tool "
            "currently returns deterministic placeholder data. Do not use these figures for real "
            "business decisions.\n\n"
            f"### Evidence summary\n{evidence}\n\n"
            "### Preliminary recommendation\n"
            "Keep quarterly supplier review controls, validate concentration exposure, and replace "
            "the stub integrations before evaluating live enterprise data. "
            f"{citations}".rstrip()
        )
        return AgentDecision(
            final_answer=answer,
            decision_summary="Sufficient demo evidence collected; synthesize a traceable answer.",
        )


class ResponsesAPIProvider(LLMProvider):
    """Shared adapter for providers implementing the OpenAI Responses protocol."""

    is_demo = False

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float,
        provider_name: str,
        api_key_env: str,
        include_strict_tool_flag: bool,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ProviderConfigurationError(
                f"{api_key_env} is required for LLM_PROVIDER={provider_name}"
            )
        self.name = f"{provider_name}-responses:{model}"
        self._provider_name = provider_name
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._include_strict_tool_flag = include_strict_tool_flag
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = http_client is None
        self._api_key = api_key
        self._runs: dict[str, _ResponsesRunState] = {}

    async def decide(
        self,
        *,
        query: str,
        history: list[Message],
        available_tools: list[ToolSpec],
        prior_results: list[ToolResult],
        step: int,
        run_id: str,
    ) -> AgentDecision:
        del query, step
        tools = [self._serialize_tool(spec) for spec in available_tools]
        state = self._runs.get(run_id)
        payload: dict[str, Any] = {
            "model": self._model,
            "instructions": self._instructions(),
            "tools": tools,
            "parallel_tool_calls": True,
            "store": True,
        }

        if state is None:
            payload["input"] = [
                {"role": message.role.value, "content": message.content} for message in history
            ]
        else:
            outputs = [
                self._serialize_tool_output(result)
                for result in prior_results
                if result.call_id in state.pending_call_ids
            ]
            if not outputs:
                raise ProviderProtocolError(
                    f"{self._provider_name} provider is missing outputs for pending tool calls"
                )
            payload["previous_response_id"] = state.previous_response_id
            payload["input"] = outputs

        response = await self._create_response(payload)
        response_id = self._require_string(response, "id")
        calls = self._parse_tool_calls(response)
        input_tokens, output_tokens = self._usage(response)
        if calls:
            self._runs[run_id] = _ResponsesRunState(
                previous_response_id=response_id,
                pending_call_ids={call.call_id for call in calls},
            )
            return AgentDecision(
                tool_calls=calls,
                decision_summary=(
                    f"{self._provider_name} Responses requested {len(calls)} tool call(s)."
                ),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_response_id=response_id,
            )

        output_text = self._parse_output_text(response)
        if not output_text:
            raise ProviderProtocolError(
                f"{self._provider_name} response contained neither function calls nor output text"
            )
        self._runs.pop(run_id, None)
        return AgentDecision(
            final_answer=output_text,
            decision_summary=f"{self._provider_name} Responses returned a final answer.",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_response_id=response_id,
        )

    async def finish_run(self, run_id: str) -> None:
        self._runs.pop(run_id, None)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(
                f"{self._base_url}/responses",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderProtocolError(
                f"{self._provider_name} Responses request failed: {exc}"
            ) from exc
        body = response.json()
        if not isinstance(body, dict):
            raise ProviderProtocolError(
                f"{self._provider_name} Responses response must be a JSON object"
            )
        if body.get("error"):
            raise ProviderProtocolError(f"{self._provider_name} Responses returned an API error")
        return body

    def _serialize_tool(self, spec: ToolSpec) -> dict[str, Any]:
        tool: dict[str, Any] = {
            "type": "function",
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        }
        if self._include_strict_tool_flag:
            tool["strict"] = False
        return tool

    @staticmethod
    def _serialize_tool_output(result: ToolResult) -> dict[str, str]:
        output = {
            "success": result.success,
            "summary": result.summary,
            "data": result.data,
            "error": result.error,
        }
        return {
            "type": "function_call_output",
            "call_id": result.call_id,
            "output": json.dumps(output, ensure_ascii=False),
        }

    @staticmethod
    def _parse_tool_calls(response: dict[str, Any]) -> list[ToolCall]:
        output = response.get("output", [])
        if not isinstance(output, list):
            raise ProviderProtocolError("OpenAI response output must be a list")
        calls: list[ToolCall] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            call_id = ResponsesAPIProvider._require_string(item, "call_id")
            name = ResponsesAPIProvider._require_string(item, "name")
            arguments_raw = item.get("arguments", "{}")
            if not isinstance(arguments_raw, str):
                raise ProviderProtocolError(f"Tool call {name} arguments must be JSON text")
            try:
                arguments = json.loads(arguments_raw)
            except json.JSONDecodeError as exc:
                raise ProviderProtocolError(f"Tool call {name} has invalid JSON arguments") from exc
            if not isinstance(arguments, dict):
                raise ProviderProtocolError(f"Tool call {name} arguments must be a JSON object")
            calls.append(ToolCall(call_id=call_id, name=name, arguments=arguments))
        return calls

    @staticmethod
    def _parse_output_text(response: dict[str, Any]) -> str:
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        output = response.get("output", [])
        if not isinstance(output, list):
            raise ProviderProtocolError("Responses output must be a list")
        messages: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            parts = [
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ]
            message = "".join(parts).strip()
            if message:
                messages.append(message)
        return "\n".join(messages)

    @staticmethod
    def _usage(response: dict[str, Any]) -> tuple[int, int]:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return 0, 0
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return (
            input_tokens if isinstance(input_tokens, int) and input_tokens >= 0 else 0,
            output_tokens if isinstance(output_tokens, int) and output_tokens >= 0 else 0,
        )

    @staticmethod
    def _require_string(value: dict[str, Any], field: str) -> str:
        result = value.get(field)
        if not isinstance(result, str) or not result:
            raise ProviderProtocolError(f"OpenAI response is missing {field}")
        return result

    @staticmethod
    def _instructions() -> str:
        return (
            "You are a traceable enterprise research agent. Use supplied function tools when "
            "external or enterprise evidence is needed. Prefer low-cost read-only tools. Use the "
            "browser only for explicit interactive tasks. After receiving tool outputs, synthesize "
            "a concise answer and distinguish demo placeholder evidence from live evidence. Never "
            "reveal "
            "hidden chain-of-thought; provide only a brief decision summary through tool usage."
        )


class OpenAIResponsesProvider(ResponsesAPIProvider):
    """OpenAI Responses API adapter for native function calling."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            provider_name="openai",
            api_key_env="OPENAI_API_KEY",
            include_strict_tool_flag=True,
            http_client=http_client,
        )


class DoubaoResponsesProvider(ResponsesAPIProvider):
    """Volcengine Ark Responses API adapter for Doubao models."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            provider_name="doubao",
            api_key_env="DOUBAO_API_KEY",
            include_strict_tool_flag=False,
            http_client=http_client,
        )


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "deterministic":
        return DeterministicProvider()
    if settings.llm_provider == "openai":
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
        return OpenAIResponsesProvider(
            api_key=api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_timeout_seconds,
        )
    api_key = settings.doubao_api_key.get_secret_value() if settings.doubao_api_key else ""
    return DoubaoResponsesProvider(
        api_key=api_key,
        model=settings.doubao_model,
        base_url=settings.doubao_base_url,
        timeout_seconds=settings.doubao_timeout_seconds,
    )
