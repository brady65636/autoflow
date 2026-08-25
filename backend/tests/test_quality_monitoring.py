import json

import pytest

from autoflow_scheduling.observability.monitoring_context import MonitoringContext, file_version
from autoflow_scheduling.observability.quality_monitoring import (
    BadCaseStatus,
    BadCaseStore,
    RootCause,
    RuntimeSampleRecorder,
    aggregate_runtime,
    answer_quality,
    check_release_gate,
    detect_retrieval_bad_cases,
    evaluate_alerts,
)


def test_monitoring_context_has_stable_versions(monkeypatch, tmp_path):
    corpus = tmp_path / "manifest.json"
    corpus.write_text("same corpus", encoding="utf8")
    monkeypatch.setenv("AUTOFLOW_ENVIRONMENT", "staging")
    monkeypatch.setenv("AUTOFLOW_CORPUS_MANIFEST", str(corpus))
    monkeypatch.setenv("AUTOFLOW_INDEX_VERSION", "index-2")

    context = MonitoringContext.from_environment().metadata(
        question_type="diagnosis", language="zh"
    )

    assert context["environment"] == "staging"
    assert context["corpus_version"] == file_version(corpus)
    assert context["index_version"] == "index-2"
    assert context["question_type"] == "diagnosis"


def test_runtime_recorder_rejects_unknown_status_and_redacts_extra_fields(tmp_path):
    recorder = RuntimeSampleRecorder(tmp_path / "runtime.jsonl")
    recorder.append({"status": "complete", "total_seconds": 1, "query": "secret"})
    row = json.loads((tmp_path / "runtime.jsonl").read_text(encoding="utf8"))
    assert row["status"] == "complete"
    assert "query" not in row
    with pytest.raises(ValueError):
        recorder.append({"status": "unknown"})


def test_runtime_aggregation_and_baseline_alerts():
    baseline_rows = [
        {"status": "complete", "total_seconds": 1, "candidate_count": 2}
        for _ in range(20)
    ]
    current_rows = [
        {
            "status": "error" if index == 0 else "complete",
            "total_seconds": 3,
            "candidate_count": 0 if index == 0 else 2,
            "fallback": index < 2,
            "stage_seconds": {"dense": 0.2},
        }
        for index in range(20)
    ]
    baseline = aggregate_runtime(baseline_rows)
    current = aggregate_runtime(current_rows)

    assert current["latency"]["p95"] == 3
    assert current["stage_latency"]["dense"]["p50"] == 0.2
    assert {alert["metric"] for alert in evaluate_alerts(current, baseline)} >= {
        "latency.p95",
        "failure_rate",
    }


def test_bad_case_detection_classification_resolution_and_promotion(tmp_path):
    report = {
        "cases": [
            {
                "case_id": "n428",
                "query": "what happens",
                "question_type": "diagnosis",
                "required_knowledge": "failure state",
                "expected_title": "N428",
                "expected_path_contains": ["Actuators"],
                "final_stage": "reranker",
                "ranks": {"rrf": 2, "reranker": 8},
                "candidate_summary": [{"chunk_id": "one"}],
                "reranker_error": None,
            }
        ]
    }
    detected = detect_retrieval_bad_cases(report, trace_id="trace-1", release_version="0.3")
    assert {item.reason for item in detected} == {
        "expected_rank_gt_5",
        "reranker_demoted_expected",
    }

    store = BadCaseStore(tmp_path / "bad.json")
    cases = store.upsert(detected)
    selected = store.classify(cases[0].key, RootCause.RERANKER_ERROR)
    assert selected.status == BadCaseStatus.CONFIRMED.value
    store.resolve(selected.key, "0.3.1")
    golden = tmp_path / "golden.json"
    assert store.append_confirmed_to_golden_set(golden) == 1
    assert json.loads(golden.read_text(encoding="utf8"))[0]["case_id"] == "n428"


def test_bad_case_must_be_classified_before_resolution(tmp_path):
    store = BadCaseStore(tmp_path / "bad.json")
    stored = store.upsert(
        detect_retrieval_bad_cases(
            {
                "cases": [
                    {
                        "case_id": "empty",
                        "final_stage": "rrf",
                        "ranks": {"rrf": None},
                        "candidate_summary": [],
                    }
                ]
            },
            trace_id=None,
            release_version="dev",
        )
    )
    with pytest.raises(ValueError, match="classified"):
        store.resolve(stored[0].key, "next")


def test_answer_citation_metrics_are_claim_based():
    metrics = answer_quality(
        {
            "claims": [
                {"id": "c1", "citation_ids": ["s1"]},
                {"id": "c2", "citation_ids": []},
            ],
            "citations": [{"id": "s1", "supports_claim_ids": ["c1"]}],
            "tokens": 120,
            "cost": 0.01,
        }
    )
    assert metrics == {
        "citation_coverage": 0.5,
        "citation_correctness": 1.0,
        "groundedness": 0.5,
        "unsupported_answer": True,
        "insufficient_evidence_refusal": False,
        "tokens": 120,
        "cost": 0.01,
    }


def test_release_gate_blocks_quality_latency_and_severe_cases():
    baseline = {
        "metrics": {"rrf": {"hit@5": 0.9, "mrr": 0.8}},
        "latency": {"p95": 1.0},
    }
    candidate = {
        "metrics": {"rrf": {"hit@5": 0.8, "mrr": 0.8}},
        "latency": {"p95": 1.3},
        "new_severe_bad_cases": 1,
    }
    result = check_release_gate(candidate, baseline)
    assert not result["passed"]
    assert len(result["failures"]) == 3
