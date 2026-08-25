"""Offline-first quality aggregation, alerts, bad-case tracking and release gates."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Iterable


class RequestStatus(str, Enum):
    COMPLETE = "complete"
    ERROR = "error"
    FALLBACK = "fallback"


class ErrorCode(str, Enum):
    TIMEOUT = "TIMEOUT"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    RERANKER_FAILED = "RERANKER_FAILED"
    GENERATION_FAILED = "GENERATION_FAILED"
    CITATION_FAILED = "CITATION_FAILED"
    INTERNAL = "INTERNAL"


class RootCause(str, Enum):
    CORPUS_MISSING = "CORPUS_MISSING"
    WRONG_METADATA = "WRONG_METADATA"
    PARSING_ERROR = "PARSING_ERROR"
    BAD_SECTION_BOUNDARY = "BAD_SECTION_BOUNDARY"
    BAD_CHUNK = "BAD_CHUNK"
    DENSE_MISS = "DENSE_MISS"
    BM25_TOKENIZATION = "BM25_TOKENIZATION"
    RRF_FUSION = "RRF_FUSION"
    RERANKER_ERROR = "RERANKER_ERROR"
    ANSWER_HALLUCINATION = "ANSWER_HALLUCINATION"
    CITATION_ERROR = "CITATION_ERROR"


class BadCaseStatus(str, Enum):
    OPEN = "OPEN"
    CONFIRMED = "CONFIRMED"
    FIXED = "FIXED"


@dataclass
class BadCase:
    case_id: str
    trace_id: str | None
    failure_stage: str
    reason: str
    introduced_by_version: str
    status: str = BadCaseStatus.OPEN.value
    root_cause: str | None = None
    fixed_by_version: str | None = None
    safety: bool = False
    regression_case: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def key(self) -> str:
        return f"{self.case_id}:{self.failure_stage}:{self.reason}"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RuntimeSampleRecorder:
    """Append bounded operational samples for local aggregation or log shipping."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def append(self, sample: dict[str, Any]) -> None:
        allowed = {
            "timestamp", "status", "error_code", "total_seconds", "stage_seconds",
            "candidate_count", "fallback", "environment", "release_version",
            "pipeline_version", "corpus_version", "index_version", "embedding_model",
            "reranker_mode", "reranker_model", "prompt_version", "question_type",
            "language", "document_type",
        }
        bounded = {key: sample[key] for key in allowed if key in sample}
        if "status" not in bounded:
            raise ValueError("runtime sample requires status")
        bounded["status"] = RequestStatus(bounded["status"]).value
        if bounded.get("error_code"):
            bounded["error_code"] = ErrorCode(bounded["error_code"]).value
        bounded.setdefault("timestamp", datetime.now(UTC).isoformat())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(bounded, ensure_ascii=False) + "\n")


class BadCaseStore:
    """A small durable queue; storage can later be replaced without changing detectors."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def list(self) -> list[BadCase]:
        if not self.path.exists():
            return []
        return [BadCase(**item) for item in json.loads(self.path.read_text(encoding="utf8"))]

    def upsert(self, cases: Iterable[BadCase]) -> list[BadCase]:
        current = {item.key: item for item in self.list()}
        for item in cases:
            if item.key in current:
                existing = current[item.key]
                item.created_at = existing.created_at
                if existing.status != BadCaseStatus.FIXED.value:
                    item.status = existing.status
                    item.root_cause = existing.root_cause
            current[item.key] = item
        result = sorted(current.values(), key=lambda item: item.created_at)
        _atomic_json(self.path, [asdict(item) for item in result])
        return result

    def classify(self, key: str, root_cause: RootCause | str) -> BadCase:
        cause = RootCause(root_cause).value
        cases = self.list()
        selected = next(item for item in cases if item.key == key)
        selected.root_cause = cause
        selected.status = BadCaseStatus.CONFIRMED.value
        selected.updated_at = datetime.now(UTC).isoformat()
        _atomic_json(self.path, [asdict(item) for item in cases])
        return selected

    def resolve(self, key: str, fixed_by_version: str) -> BadCase:
        cases = self.list()
        selected = next(item for item in cases if item.key == key)
        if not selected.root_cause:
            raise ValueError("a bad case must be classified before it can be fixed")
        selected.status = BadCaseStatus.FIXED.value
        selected.fixed_by_version = fixed_by_version
        selected.updated_at = datetime.now(UTC).isoformat()
        _atomic_json(self.path, [asdict(item) for item in cases])
        return selected

    def append_confirmed_to_golden_set(self, golden_path: str | Path) -> int:
        """Promote confirmed/fixed cases, preserving an existing case with the same ID."""
        target = Path(golden_path)
        golden = json.loads(target.read_text(encoding="utf8")) if target.exists() else []
        by_id = {str(item.get("case_id")): item for item in golden}
        added = 0
        for bad_case in self.list():
            if bad_case.status == BadCaseStatus.OPEN.value or not bad_case.regression_case:
                continue
            if bad_case.case_id not in by_id:
                by_id[bad_case.case_id] = bad_case.regression_case
                added += 1
        _atomic_json(target, list(by_id.values()))
        return added


def detect_retrieval_bad_cases(
    report: dict[str, Any], *, trace_id: str | None, release_version: str
) -> list[BadCase]:
    detected: list[BadCase] = []
    for case in report.get("cases", []):
        case_id = str(case.get("case_id", "unknown"))
        ranks = case.get("ranks", {})
        final_stage = str(case.get("final_stage", "reranker"))
        final_rank = ranks.get(final_stage)
        regression = {
            key: case[key]
            for key in (
                "case_id", "query", "question_type", "required_knowledge",
                "expected_title", "expected_path_contains", "expected_sections",
            )
            if key in case
        }

        def add(stage: str, reason: str, safety: bool = False) -> None:
            detected.append(
                BadCase(
                    case_id=case_id,
                    trace_id=trace_id,
                    failure_stage=stage,
                    reason=reason,
                    introduced_by_version=release_version,
                    safety=safety,
                    regression_case=regression,
                )
            )

        if "candidate_summary" in case and not case["candidate_summary"]:
            add("retrieval", "empty_retrieval")
        if final_rank is None or final_rank > 5:
            add(final_stage, "expected_rank_gt_5")
        if case.get("reranker_error"):
            add("reranker", "reranker_fallback")
        rrf_rank, reranker_rank = ranks.get("rrf"), ranks.get("reranker")
        if rrf_rank is not None and (reranker_rank is None or reranker_rank > rrf_rank):
            add("reranker", "reranker_demoted_expected")
        if case.get("safety") and (final_rank is None or final_rank > 1):
            add(final_stage, "safety_case_failed", safety=True)
    return detected


def detect_answer_bad_cases(
    record: dict[str, Any], *, case_id: str, trace_id: str | None, release_version: str
) -> list[BadCase]:
    metrics = answer_quality(record)
    failures: list[tuple[str, str]] = []
    if record.get("user_downvote"):
        failures.append(("answer", "user_downvote"))
    if metrics["unsupported_answer"]:
        failures.append(("answer", "answer_not_grounded"))
    if metrics["citation_correctness"] < 1 and record.get("citations"):
        failures.append(("citation", "citation_does_not_support_claim"))
    return [
        BadCase(
            case_id=case_id,
            trace_id=trace_id,
            failure_stage=stage,
            reason=reason,
            introduced_by_version=release_version,
            safety=bool(record.get("safety")),
            regression_case=record.get("regression_case"),
        )
        for stage, reason in failures
    ]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 6)


def aggregate_runtime(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(samples)
    total = len(rows)
    stage_names = sorted({name for row in rows for name in row.get("stage_seconds", {})})
    denominator = max(total, 1)
    latencies = [float(row["total_seconds"]) for row in rows if "total_seconds" in row]
    return {
        "requests": total,
        "success_rate": sum(row.get("status") == "complete" for row in rows) / denominator,
        "fallback_rate": sum(bool(row.get("fallback")) for row in rows) / denominator,
        "empty_retrieval_rate": sum(
            int(row.get("candidate_count", 0)) == 0 for row in rows
        ) / denominator,
        "timeout_rate": sum(row.get("error_code") == "TIMEOUT" for row in rows) / denominator,
        "latency": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
        },
        "stage_latency": {
            name: {
                "p50": _percentile(
                    [
                        float(row["stage_seconds"][name])
                        for row in rows
                        if name in row.get("stage_seconds", {})
                    ],
                    0.50,
                ),
                "p95": _percentile(
                    [
                        float(row["stage_seconds"][name])
                        for row in rows
                        if name in row.get("stage_seconds", {})
                    ],
                    0.95,
                ),
            }
            for name in stage_names
        },
        "errors": {
            code: sum(row.get("error_code") == code for row in rows)
            for code in sorted({row.get("error_code") for row in rows if row.get("error_code")})
        },
    }


def aggregate_by_dimensions(
    samples: Iterable[dict[str, Any]],
    dimensions: tuple[str, ...] = (
        "release_version", "question_type", "language", "document_type"
    ),
) -> dict[str, dict[str, Any]]:
    rows = list(samples)
    return {
        dimension: {
            str(value): aggregate_runtime(
                row for row in rows if row.get(dimension) == value
            )
            for value in sorted({row.get(dimension) for row in rows if row.get(dimension)})
        }
        for dimension in dimensions
    }


def retrieval_quality(report: dict[str, Any]) -> dict[str, Any]:
    """Add reranker lift/error rates and question-type slices to stage metrics."""
    cases = report.get("cases", [])

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        eligible = [row for row in rows if row.get("ranks", {}).get("rrf") is not None]
        promoted = [
            row for row in eligible
            if row.get("ranks", {}).get("reranker") is not None
            and row["ranks"]["reranker"] < row["ranks"]["rrf"]
        ]
        demoted = [
            row for row in eligible
            if row.get("ranks", {}).get("reranker") is None
            or row["ranks"]["reranker"] > row["ranks"]["rrf"]
        ]
        denominator = max(len(eligible), 1)
        return {
            "eligible_cases": len(eligible),
            "reranker_positive_lift_rate": len(promoted) / denominator,
            "reranker_harm_rate": len(demoted) / denominator,
        }

    question_types = sorted({str(case.get("question_type")) for case in cases})
    return {
        "stages": report.get("metrics", {}),
        "reranker_effect": summarize(cases),
        "by_question_type": {
            kind: summarize([case for case in cases if str(case.get("question_type")) == kind])
            for kind in question_types
        },
    }


def answer_quality(record: dict[str, Any]) -> dict[str, float | int | bool]:
    """Compute deterministic citation metrics from claim/citation verification output."""
    claims = record.get("claims", [])
    citations = {str(item["id"]): item for item in record.get("citations", [])}
    total_claims = len(claims)
    cited_claims = 0
    supported_claims = 0
    citation_links = 0
    correct_links = 0
    for claim in claims:
        claim_id = str(claim.get("id"))
        links = [str(value) for value in claim.get("citation_ids", [])]
        if links:
            cited_claims += 1
        claim_supported = False
        for citation_id in links:
            citation_links += 1
            citation = citations.get(citation_id)
            if citation and claim_id in {str(v) for v in citation.get("supports_claim_ids", [])}:
                correct_links += 1
                claim_supported = True
        supported_claims += claim_supported
    divisor = max(total_claims, 1)
    return {
        "citation_coverage": cited_claims / divisor,
        "citation_correctness": correct_links / max(citation_links, 1),
        "groundedness": supported_claims / divisor,
        "unsupported_answer": bool(total_claims and supported_claims < total_claims),
        "insufficient_evidence_refusal": bool(
            record.get("refused") and not record.get("evidence_available", True)
        ),
        "tokens": int(record.get("tokens", 0)),
        "cost": float(record.get("cost", 0.0)),
    }


@dataclass(frozen=True)
class AlertPolicy:
    latency_multiplier: float = 1.5
    rate_multiplier: float = 2.0
    minimum_requests: int = 20


def evaluate_alerts(
    current: dict[str, Any], baseline: dict[str, Any], *, policy: AlertPolicy = AlertPolicy()
) -> list[dict[str, Any]]:
    if current.get("requests", 0) < policy.minimum_requests:
        return []
    alerts = []
    checks = (
        (
            "latency.p95",
            current.get("latency", {}).get("p95"),
            baseline.get("latency", {}).get("p95"),
            policy.latency_multiplier,
        ),
        (
            "fallback_rate",
            current.get("fallback_rate"),
            baseline.get("fallback_rate"),
            policy.rate_multiplier,
        ),
        (
            "empty_retrieval_rate",
            current.get("empty_retrieval_rate"),
            baseline.get("empty_retrieval_rate"),
            policy.rate_multiplier,
        ),
        (
            "failure_rate",
            1 - current.get("success_rate", 1),
            1 - baseline.get("success_rate", 1),
            policy.rate_multiplier,
        ),
    )
    for metric, value, normal, multiplier in checks:
        threshold = normal * multiplier if normal is not None else None
        if value is not None and threshold is not None and value > threshold:
            alerts.append({"metric": metric, "value": value, "threshold": threshold})
    return alerts


def check_release_gate(
    candidate: dict[str, Any], baseline: dict[str, Any], *, latency_tolerance: float = 1.2
) -> dict[str, Any]:
    """Fail a release on quality regression, latency regression or severe bad cases."""
    failures = []
    candidate_metrics = candidate.get("metrics", {}).get("reranker") or candidate.get(
        "metrics", {}
    ).get("rrf", {})
    baseline_metrics = baseline.get("metrics", {}).get("reranker") or baseline.get(
        "metrics", {}
    ).get("rrf", {})
    for metric in ("hit@5", "mrr"):
        if candidate_metrics.get(metric, 0) < baseline_metrics.get(metric, 0):
            failures.append(f"{metric} regressed")
    candidate_p95 = candidate.get("latency", {}).get("p95")
    baseline_p95 = baseline.get("latency", {}).get("p95")
    if candidate_p95 is not None and baseline_p95 is not None:
        if candidate_p95 > baseline_p95 * latency_tolerance:
            failures.append("p95 latency regressed")
    if candidate.get("new_severe_bad_cases", 0) > 0:
        failures.append("new severe bad cases")
    return {"passed": not failures, "failures": failures}
