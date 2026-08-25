"""CLI for the local half of the monitoring and regression loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .quality_monitoring import (
    BadCaseStatus,
    BadCaseStore,
    RootCause,
    aggregate_by_dimensions,
    aggregate_runtime,
    check_release_gate,
    detect_retrieval_bad_cases,
    evaluate_alerts,
    retrieval_quality,
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def _samples(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoFlow quality monitoring tools")
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan", help="detect and persist bad cases from an eval report")
    scan.add_argument("report", type=Path)
    scan.add_argument("store", type=Path)
    scan.add_argument("--release", required=True)
    scan.add_argument("--trace-id")

    classify = commands.add_parser("classify", help="assign a required root-cause category")
    classify.add_argument("store", type=Path)
    classify.add_argument("key")
    classify.add_argument("root_cause", choices=[item.value for item in RootCause])

    resolve = commands.add_parser("resolve", help="close a classified bad case")
    resolve.add_argument("store", type=Path)
    resolve.add_argument("key")
    resolve.add_argument("--fixed-by", required=True)

    promote = commands.add_parser("promote", help="append confirmed cases to a golden set")
    promote.add_argument("store", type=Path)
    promote.add_argument("golden", type=Path)

    gate = commands.add_parser("gate", help="compare a candidate report with its baseline")
    gate.add_argument("candidate", type=Path)
    gate.add_argument("baseline", type=Path)
    gate.add_argument("--latency-tolerance", type=float, default=1.2)

    dashboard = commands.add_parser("dashboard", help="build dashboard-ready JSON")
    dashboard.add_argument("samples", type=Path, help="runtime JSONL samples")
    dashboard.add_argument("bad_cases", type=Path)
    dashboard.add_argument("--baseline-samples", type=Path)
    dashboard.add_argument("--evaluation-report", type=Path)
    dashboard.add_argument("--output", type=Path)

    args = parser.parse_args()
    exit_code = 0
    if args.command == "scan":
        report = _load(args.report)
        detected = detect_retrieval_bad_cases(
            report, trace_id=args.trace_id, release_version=args.release
        )
        stored = BadCaseStore(args.store).upsert(detected)
        result = {"detected": len(detected), "stored": len(stored)}
    elif args.command == "classify":
        case = BadCaseStore(args.store).classify(args.key, args.root_cause)
        result = {"key": case.key, "status": case.status, "root_cause": case.root_cause}
    elif args.command == "resolve":
        case = BadCaseStore(args.store).resolve(args.key, args.fixed_by)
        result = {"key": case.key, "status": case.status, "fixed_by": case.fixed_by_version}
    elif args.command == "promote":
        result = {"added": BadCaseStore(args.store).append_confirmed_to_golden_set(args.golden)}
    elif args.command == "gate":
        result = check_release_gate(
            _load(args.candidate),
            _load(args.baseline),
            latency_tolerance=args.latency_tolerance,
        )
        exit_code = 0 if result["passed"] else 1
    else:
        samples = _samples(args.samples)
        health = aggregate_runtime(samples)
        cases = BadCaseStore(args.bad_cases).list()
        result = {
            "health": health,
            "health_by_dimension": aggregate_by_dimensions(samples),
            "retrieval_quality": (
                retrieval_quality(_load(args.evaluation_report))
                if args.evaluation_report
                else {}
            ),
            "bad_cases": {
                "open": sum(item.status != BadCaseStatus.FIXED.value for item in cases),
                "fixed": sum(item.status == BadCaseStatus.FIXED.value for item in cases),
                "root_causes": {
                    cause: sum(item.root_cause == cause for item in cases)
                    for cause in sorted({item.root_cause for item in cases if item.root_cause})
                },
                "items": [
                    {
                        "case_id": item.case_id,
                        "trace_id": item.trace_id,
                        "failure_stage": item.failure_stage,
                        "status": item.status,
                    }
                    for item in cases
                ],
            },
            "alerts": (
                evaluate_alerts(health, aggregate_runtime(_samples(args.baseline_samples)))
                if args.baseline_samples
                else []
            ),
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
