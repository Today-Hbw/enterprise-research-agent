from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from app.agent.provider import DeterministicProvider
from app.agent.runtime import AgentRuntime
from app.config import Settings
from app.knowledge import (
    DeterministicEmbedder,
    InMemoryKnowledgeBackend,
    KnowledgeDocumentInput,
    KnowledgeService,
)
from app.models import AccessContext, RunStatus, SourceType, ToolPermission
from app.store import InMemoryStore
from app.tools.knowledge import KnowledgeSearchTool
from app.tools.stubs import build_tool_registry


class EvaluationDocument(BaseModel):
    document_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=2_000_000)
    tenant_id: str = Field(default="demo", min_length=1, max_length=128)
    knowledge_base_id: str = Field(default="default", min_length=1, max_length=128)
    allowed_principal_ids: list[str] = Field(default_factory=list, max_length=100)
    public: bool = True
    source_url: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict, max_length=20)


class RetrievalEvaluationCase(BaseModel):
    case_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=10_000)
    relevant_document_ids: set[str] = Field(min_length=1)
    tenant_id: str = Field(default="demo", min_length=1, max_length=128)
    principal_ids: set[str] = Field(default_factory=lambda: {"demo-user"})
    knowledge_base_id: str | None = None
    metadata_filters: dict[str, str] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=1, le=10)


class AgentEvaluationCase(BaseModel):
    case_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=10_000)
    expected_tools: set[str] = Field(default_factory=set)
    forbidden_tools: set[str] = Field(default_factory=set)
    tenant_id: str = Field(default="demo", min_length=1, max_length=128)
    principal_ids: set[str] = Field(default_factory=lambda: {"demo-user"})
    min_source_count: int = Field(default=0, ge=0)
    required_source_types: set[SourceType] = Field(default_factory=set)
    require_citations: bool = False
    required_answer_terms: set[str] = Field(default_factory=set)
    max_llm_calls: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=0)
    max_duration_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_tool_expectations(self) -> Self:
        overlap = self.expected_tools.intersection(self.forbidden_tools)
        if overlap:
            raise ValueError(f"Tools cannot be both expected and forbidden: {sorted(overlap)}")
        return self


class RetrievalEvaluationConfig(BaseModel):
    embedding_dimensions: int = Field(default=64, ge=16, le=4096)
    chunk_size: int = Field(default=800, ge=32, le=20_000)
    chunk_overlap: int = Field(default=120, ge=0, le=10_000)
    ranking: Literal["semantic", "hybrid"] = "hybrid"
    hybrid_rrf_k: int = Field(default=60, ge=1, le=1_000)
    reranker: Literal["none", "token_overlap"] = "token_overlap"
    rerank_candidate_k: int = Field(default=30, ge=1, le=100)

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class AgentEvaluationConfig(BaseModel):
    max_steps: int = Field(default=8, ge=1, le=32)
    run_timeout_seconds: float = Field(default=10, gt=0, le=300)
    tool_timeout_seconds: float = Field(default=2, gt=0, le=60)
    max_parallel_tools: int = Field(default=4, ge=1, le=16)
    tool_max_permission: ToolPermission = ToolPermission.HIGH


class EvaluationThresholds(BaseModel):
    min_retrieval_recall_at_k: float = Field(default=0, ge=0, le=1)
    min_retrieval_mrr: float = Field(default=0, ge=0, le=1)
    min_retrieval_hit_rate: float = Field(default=0, ge=0, le=1)
    min_agent_pass_rate: float = Field(default=0, ge=0, le=1)
    min_agent_completion_rate: float = Field(default=0, ge=0, le=1)
    min_agent_tool_recall: float = Field(default=0, ge=0, le=1)
    min_agent_citation_rate: float = Field(default=0, ge=0, le=1)


class EvaluationDataset(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2_000)
    documents: list[EvaluationDocument] = Field(default_factory=list)
    retrieval_cases: list[RetrievalEvaluationCase] = Field(default_factory=list)
    agent_cases: list[AgentEvaluationCase] = Field(default_factory=list)
    retrieval_config: RetrievalEvaluationConfig = Field(default_factory=RetrievalEvaluationConfig)
    agent_config: AgentEvaluationConfig = Field(default_factory=AgentEvaluationConfig)
    thresholds: EvaluationThresholds = Field(default_factory=EvaluationThresholds)

    @model_validator(mode="after")
    def validate_references_and_case_ids(self) -> Self:
        document_ids = [document.document_id for document in self.documents]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("Evaluation document_id values must be unique")
        known_documents = set(document_ids)
        unknown = {
            document_id
            for case in self.retrieval_cases
            for document_id in case.relevant_document_ids - known_documents
        }
        if unknown:
            raise ValueError(f"Retrieval cases reference unknown documents: {sorted(unknown)}")
        case_ids = [
            *(case.case_id for case in self.retrieval_cases),
            *(case.case_id for case in self.agent_cases),
        ]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Evaluation case_id values must be unique across the dataset")
        return self


class RetrievalCaseResult(BaseModel):
    case_id: str
    query: str
    relevant_document_ids: list[str]
    retrieved_document_ids: list[str]
    recall_at_k: float
    reciprocal_rank: float
    hit: bool
    passed: bool


class RetrievalEvaluationSummary(BaseModel):
    case_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    hit_rate: float
    cases: list[RetrievalCaseResult]


class AgentCaseResult(BaseModel):
    case_id: str
    query: str
    status: RunStatus
    called_tools: list[str]
    missing_tools: list[str]
    forbidden_tools_called: list[str]
    tool_recall: float
    source_count: int
    source_types: list[SourceType]
    citation_present: bool
    llm_call_count: int
    tool_call_count: int
    duration_ms: int
    token_usage: int
    estimated_cost: float | None
    passed: bool
    failure_reasons: list[str]


class AgentEvaluationSummary(BaseModel):
    case_count: int
    pass_rate: float
    completion_rate: float
    tool_recall: float
    citation_rate: float
    average_llm_calls: float
    average_tool_calls: float
    average_duration_ms: float
    total_tokens: int
    total_estimated_cost: float | None
    cases: list[AgentCaseResult]


class EvaluationReport(BaseModel):
    dataset_name: str
    dataset_description: str
    provider: str
    retrieval_config: RetrievalEvaluationConfig
    agent_config: AgentEvaluationConfig
    thresholds: EvaluationThresholds
    retrieval: RetrievalEvaluationSummary
    agent: AgentEvaluationSummary
    passed: bool
    threshold_failures: list[str]


class EvaluationRunner:
    def __init__(self, dataset: EvaluationDataset) -> None:
        self.dataset = dataset

    async def run(self) -> EvaluationReport:
        service = self._build_knowledge_service()
        try:
            document_id_map = await self._index_documents(service)
            retrieval = await self._run_retrieval_cases(service, document_id_map)
            agent = await self._run_agent_cases(service)
        finally:
            await service.aclose()
        threshold_failures = self._threshold_failures(retrieval, agent)
        return EvaluationReport(
            dataset_name=self.dataset.name,
            dataset_description=self.dataset.description,
            provider=DeterministicProvider.name,
            retrieval_config=self.dataset.retrieval_config,
            agent_config=self.dataset.agent_config,
            thresholds=self.dataset.thresholds,
            retrieval=retrieval,
            agent=agent,
            passed=not threshold_failures,
            threshold_failures=threshold_failures,
        )

    def _build_knowledge_service(self) -> KnowledgeService:
        config = self.dataset.retrieval_config
        return KnowledgeService(
            backend=InMemoryKnowledgeBackend(),
            embedder=DeterministicEmbedder(config.embedding_dimensions),
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            ranking=config.ranking,
            hybrid_rrf_k=config.hybrid_rrf_k,
            reranker=config.reranker,
            rerank_candidate_k=config.rerank_candidate_k,
        )

    async def _index_documents(self, service: KnowledgeService) -> dict[str, str]:
        actual_to_dataset: dict[str, str] = {}
        for document in self.dataset.documents:
            indexed = await service.index_document(
                tenant_id=document.tenant_id,
                document=KnowledgeDocumentInput(
                    title=document.title,
                    content=document.content,
                    knowledge_base_id=document.knowledge_base_id,
                    allowed_principal_ids=document.allowed_principal_ids,
                    public=document.public,
                    source_url=document.source_url,
                    metadata=document.metadata,
                ),
            )
            actual_to_dataset[indexed.document_id] = document.document_id
        return actual_to_dataset

    async def _run_retrieval_cases(
        self,
        service: KnowledgeService,
        document_id_map: dict[str, str],
    ) -> RetrievalEvaluationSummary:
        results: list[RetrievalCaseResult] = []
        for case in self.dataset.retrieval_cases:
            matches = await service.search(
                query=case.query,
                tenant_id=case.tenant_id,
                principal_ids=case.principal_ids,
                knowledge_base_id=case.knowledge_base_id,
                top_k=case.top_k,
                metadata_filters=case.metadata_filters,
            )
            retrieved = _unique(
                document_id_map[match.document_id]
                for match in matches
                if match.document_id in document_id_map
            )
            relevant = sorted(case.relevant_document_ids)
            hits = case.relevant_document_ids.intersection(retrieved)
            recall = len(hits) / len(case.relevant_document_ids)
            first_rank = next(
                (
                    index
                    for index, document_id in enumerate(retrieved, start=1)
                    if document_id in case.relevant_document_ids
                ),
                None,
            )
            reciprocal_rank = 1 / first_rank if first_rank else 0
            results.append(
                RetrievalCaseResult(
                    case_id=case.case_id,
                    query=case.query,
                    relevant_document_ids=relevant,
                    retrieved_document_ids=retrieved,
                    recall_at_k=recall,
                    reciprocal_rank=reciprocal_rank,
                    hit=bool(hits),
                    passed=recall == 1,
                )
            )
        return RetrievalEvaluationSummary(
            case_count=len(results),
            recall_at_k=_mean(result.recall_at_k for result in results),
            mean_reciprocal_rank=_mean(result.reciprocal_rank for result in results),
            hit_rate=_mean(1.0 if result.hit else 0.0 for result in results),
            cases=results,
        )

    async def _run_agent_cases(self, service: KnowledgeService) -> AgentEvaluationSummary:
        results: list[AgentCaseResult] = []
        config = self.dataset.agent_config
        for case in self.dataset.agent_cases:
            store = InMemoryStore()
            registry = build_tool_registry(
                config.tool_timeout_seconds,
                knowledge_tool=KnowledgeSearchTool(
                    service=service,
                    timeout_seconds=config.tool_timeout_seconds,
                ),
                max_permission=config.tool_max_permission,
            )
            runtime = AgentRuntime(
                settings=Settings(
                    max_steps=config.max_steps,
                    run_timeout_seconds=config.run_timeout_seconds,
                    run_token_budget=None,
                    run_cost_budget_usd=None,
                    tool_timeout_seconds=config.tool_timeout_seconds,
                    max_parallel_tools=config.max_parallel_tools,
                    tool_max_permission=config.tool_max_permission,
                    llm_provider="deterministic",
                    llm_input_cost_per_million_tokens=None,
                    llm_output_cost_per_million_tokens=None,
                ),
                provider=DeterministicProvider(),
                registry=registry,
                store=store,
            )
            events = [
                event
                async for event in runtime.stream(
                    query=case.query,
                    conversation_id=None,
                    access_context=AccessContext(
                        tenant_id=case.tenant_id,
                        principal_ids=case.principal_ids,
                    ),
                )
            ]
            run = await store.get_run(events[-1].run_id)
            if run is None:
                raise RuntimeError(f"Evaluation run was not persisted: {case.case_id}")
            called_tools = [step.tool_name for step in run.trace if step.tool_name is not None]
            called_tool_set = set(called_tools)
            missing_tools = sorted(case.expected_tools - called_tool_set)
            forbidden_tools_called = sorted(case.forbidden_tools.intersection(called_tool_set))
            tool_recall = (
                len(case.expected_tools.intersection(called_tool_set)) / len(case.expected_tools)
                if case.expected_tools
                else 1
            )
            source_types = sorted(
                {source.source_type for source in run.sources},
                key=lambda source_type: source_type.value,
            )
            citation_present = bool(
                run.sources and run.final_answer and re.search(r"\[\d+\]", run.final_answer)
            )
            failure_reasons = self._agent_failure_reasons(
                case=case,
                run_status=run.status,
                missing_tools=missing_tools,
                forbidden_tools_called=forbidden_tools_called,
                source_count=len(run.sources),
                source_types=set(source_types),
                citation_present=citation_present,
                final_answer=run.final_answer,
                llm_call_count=run.metrics.llm_call_count,
                tool_call_count=run.metrics.tool_call_count,
                duration_ms=run.metrics.duration_ms,
            )
            results.append(
                AgentCaseResult(
                    case_id=case.case_id,
                    query=case.query,
                    status=run.status,
                    called_tools=called_tools,
                    missing_tools=missing_tools,
                    forbidden_tools_called=forbidden_tools_called,
                    tool_recall=tool_recall,
                    source_count=len(run.sources),
                    source_types=source_types,
                    citation_present=citation_present,
                    llm_call_count=run.metrics.llm_call_count,
                    tool_call_count=run.metrics.tool_call_count,
                    duration_ms=run.metrics.duration_ms,
                    token_usage=run.metrics.token_usage,
                    estimated_cost=run.metrics.estimated_cost,
                    passed=not failure_reasons,
                    failure_reasons=failure_reasons,
                )
            )
        citation_cases = [
            result
            for case, result in zip(self.dataset.agent_cases, results, strict=True)
            if case.require_citations
        ]
        estimated_costs = [
            result.estimated_cost for result in results if result.estimated_cost is not None
        ]
        return AgentEvaluationSummary(
            case_count=len(results),
            pass_rate=_mean(1.0 if result.passed else 0.0 for result in results),
            completion_rate=_mean(
                1.0 if result.status == RunStatus.COMPLETED else 0.0 for result in results
            ),
            tool_recall=_mean(result.tool_recall for result in results),
            citation_rate=_mean(
                1.0 if result.citation_present else 0.0 for result in citation_cases
            ),
            average_llm_calls=_mean(float(result.llm_call_count) for result in results),
            average_tool_calls=_mean(float(result.tool_call_count) for result in results),
            average_duration_ms=_mean(float(result.duration_ms) for result in results),
            total_tokens=sum(result.token_usage for result in results),
            total_estimated_cost=(round(sum(estimated_costs), 8) if estimated_costs else None),
            cases=results,
        )

    @staticmethod
    def _agent_failure_reasons(
        *,
        case: AgentEvaluationCase,
        run_status: RunStatus,
        missing_tools: list[str],
        forbidden_tools_called: list[str],
        source_count: int,
        source_types: set[SourceType],
        citation_present: bool,
        final_answer: str | None,
        llm_call_count: int,
        tool_call_count: int,
        duration_ms: int,
    ) -> list[str]:
        reasons: list[str] = []
        if run_status != RunStatus.COMPLETED:
            reasons.append(f"run status was {run_status}")
        if missing_tools:
            reasons.append(f"missing expected tools: {', '.join(missing_tools)}")
        if forbidden_tools_called:
            reasons.append(f"called forbidden tools: {', '.join(forbidden_tools_called)}")
        if source_count < case.min_source_count:
            reasons.append(f"source count {source_count} was below {case.min_source_count}")
        missing_source_types = case.required_source_types - source_types
        if missing_source_types:
            reasons.append(
                "missing source types: "
                + ", ".join(sorted(item.value for item in missing_source_types))
            )
        if case.require_citations and not citation_present:
            reasons.append("final answer did not contain a numbered citation")
        normalized_answer = (final_answer or "").lower()
        missing_terms = sorted(
            term for term in case.required_answer_terms if term.lower() not in normalized_answer
        )
        if missing_terms:
            reasons.append(f"final answer missed required terms: {', '.join(missing_terms)}")
        if case.max_llm_calls is not None and llm_call_count > case.max_llm_calls:
            reasons.append(f"LLM call count {llm_call_count} exceeded {case.max_llm_calls}")
        if case.max_tool_calls is not None and tool_call_count > case.max_tool_calls:
            reasons.append(f"tool call count {tool_call_count} exceeded {case.max_tool_calls}")
        if case.max_duration_ms is not None and duration_ms > case.max_duration_ms:
            reasons.append(f"duration {duration_ms}ms exceeded {case.max_duration_ms}ms")
        return reasons

    def _threshold_failures(
        self,
        retrieval: RetrievalEvaluationSummary,
        agent: AgentEvaluationSummary,
    ) -> list[str]:
        thresholds = self.dataset.thresholds
        checks = [
            (
                "retrieval recall@k",
                retrieval.recall_at_k,
                thresholds.min_retrieval_recall_at_k,
            ),
            (
                "retrieval MRR",
                retrieval.mean_reciprocal_rank,
                thresholds.min_retrieval_mrr,
            ),
            (
                "retrieval hit rate",
                retrieval.hit_rate,
                thresholds.min_retrieval_hit_rate,
            ),
            ("agent pass rate", agent.pass_rate, thresholds.min_agent_pass_rate),
            (
                "agent completion rate",
                agent.completion_rate,
                thresholds.min_agent_completion_rate,
            ),
            ("agent tool recall", agent.tool_recall, thresholds.min_agent_tool_recall),
            (
                "agent citation rate",
                agent.citation_rate,
                thresholds.min_agent_citation_rate,
            ),
        ]
        return [
            f"{name} {actual:.4f} was below threshold {minimum:.4f}"
            for name, actual, minimum in checks
            if actual < minimum
        ]


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    return EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


async def evaluate_dataset(dataset: EvaluationDataset) -> EvaluationReport:
    return await EvaluationRunner(dataset).run()


def _mean(values) -> float:
    items = list(values)
    return fmean(items) if items else 0


def _unique(values) -> list[str]:
    return list(dict.fromkeys(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic Agent and Retrieval evaluations."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to a JSON dataset.")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report.")
    parser.add_argument(
        "--fail-on-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return exit code 1 when configured thresholds are not met.",
    )
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_evaluation_dataset(args.dataset)
    report = asyncio.run(evaluate_dataset(dataset))
    rendered = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if args.fail_on_threshold and not report.passed else 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
