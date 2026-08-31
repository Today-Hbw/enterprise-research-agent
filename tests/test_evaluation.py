import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation import (
    AgentEvaluationCase,
    EvaluationDataset,
    EvaluationDocument,
    EvaluationThresholds,
    RetrievalEvaluationCase,
    cli,
    evaluate_dataset,
    load_evaluation_dataset,
)

DEMO_DATASET = Path(__file__).parents[1] / "evals" / "demo.json"


@pytest.mark.asyncio
async def test_demo_evaluation_dataset_meets_all_thresholds() -> None:
    report = await evaluate_dataset(load_evaluation_dataset(DEMO_DATASET))

    assert report.passed is True
    assert report.threshold_failures == []
    assert report.retrieval.case_count == 2
    assert report.retrieval.recall_at_k == 1
    assert report.retrieval.mean_reciprocal_rank == 1
    assert report.retrieval.hit_rate == 1
    assert report.agent.case_count == 3
    assert report.agent.pass_rate == 1
    assert report.agent.completion_rate == 1
    assert report.agent.tool_recall == 1
    assert report.agent.citation_rate == 1


@pytest.mark.asyncio
async def test_retrieval_miss_fails_thresholds_with_case_level_evidence() -> None:
    dataset = EvaluationDataset(
        name="private-document-miss",
        documents=[
            EvaluationDocument(
                document_id="private-policy",
                title="Private policy",
                content="Quarterly supplier review is mandatory.",
                public=False,
                allowed_principal_ids=["alice"],
            )
        ],
        retrieval_cases=[
            RetrievalEvaluationCase(
                case_id="unauthorized-principal",
                query="quarterly supplier review",
                relevant_document_ids={"private-policy"},
                principal_ids={"bob"},
                top_k=1,
            )
        ],
        thresholds=EvaluationThresholds(
            min_retrieval_recall_at_k=1,
            min_retrieval_mrr=1,
            min_retrieval_hit_rate=1,
        ),
    )

    report = await evaluate_dataset(dataset)

    assert report.passed is False
    assert len(report.threshold_failures) == 3
    result = report.retrieval.cases[0]
    assert result.retrieved_document_ids == []
    assert result.recall_at_k == 0
    assert result.reciprocal_rank == 0
    assert result.hit is False
    assert result.passed is False


@pytest.mark.asyncio
async def test_agent_failure_reports_routing_and_limit_diagnostics() -> None:
    dataset = EvaluationDataset(
        name="agent-routing-failure",
        documents=[
            EvaluationDocument(
                document_id="browser-guide",
                title="Browser guide",
                content="Interactive exports require a browser.",
            )
        ],
        agent_cases=[
            AgentEvaluationCase(
                case_id="wrong-expectations",
                query="登录 SaaS 后台并点击导出",
                expected_tools={"http_fetch"},
                forbidden_tools={"browser"},
                max_tool_calls=1,
                require_citations=True,
            )
        ],
        thresholds=EvaluationThresholds(min_agent_pass_rate=1),
    )

    report = await evaluate_dataset(dataset)

    assert report.passed is False
    result = report.agent.cases[0]
    assert result.status == "completed"
    assert result.missing_tools == ["http_fetch"]
    assert result.forbidden_tools_called == ["browser"]
    assert result.tool_call_count == 3
    assert result.citation_present is True
    assert result.passed is False
    assert result.failure_reasons == [
        "missing expected tools: http_fetch",
        "called forbidden tools: browser",
        "tool call count 3 exceeded 1",
    ]


def test_cli_writes_machine_readable_report_and_returns_success(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    exit_code = cli(
        [
            "--dataset",
            str(DEMO_DATASET),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["retrieval"]["recall_at_k"] == 1
    assert payload["agent"]["tool_recall"] == 1
    assert payload["agent"]["cases"][0]["query"] == "调研市场并结合采购数据做分析"


def test_cli_threshold_exit_can_be_disabled_for_exploration(tmp_path: Path) -> None:
    dataset = EvaluationDataset(
        name="threshold-failure",
        documents=[
            EvaluationDocument(
                document_id="private",
                title="Private",
                content="Restricted evidence.",
                public=False,
                allowed_principal_ids=["alice"],
            )
        ],
        retrieval_cases=[
            RetrievalEvaluationCase(
                case_id="miss",
                query="restricted evidence",
                relevant_document_ids={"private"},
                principal_ids={"bob"},
            )
        ],
        thresholds=EvaluationThresholds(min_retrieval_recall_at_k=1),
    )
    dataset_path = tmp_path / "failing.json"
    dataset_path.write_text(dataset.model_dump_json(), encoding="utf-8")

    assert cli(["--dataset", str(dataset_path)]) == 1
    assert (
        cli(
            [
                "--dataset",
                str(dataset_path),
                "--no-fail-on-threshold",
            ]
        )
        == 0
    )


def test_dataset_rejects_unknown_relevant_documents() -> None:
    with pytest.raises(ValidationError, match="unknown documents"):
        EvaluationDataset(
            name="invalid",
            retrieval_cases=[
                RetrievalEvaluationCase(
                    case_id="unknown-reference",
                    query="policy",
                    relevant_document_ids={"missing"},
                )
            ],
        )
