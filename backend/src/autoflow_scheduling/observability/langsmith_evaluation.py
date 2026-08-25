"""LangSmith dataset and retrieval-evaluation adapter.

The adapter deliberately accepts a client object so unit tests (and callers that
already own a configured client) never need to construct a network client.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

_LOG = logging.getLogger(__name__)


class LangSmithEvaluationError(RuntimeError):
    """A clear, operation-specific wrapper for SDK or network failures."""


def normalize_evaluation_cases(cases: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert the repository's evaluation-case JSON to LangSmith example IO.

    ``query`` is the model input.  All expected_* fields become reference output;
    case_id and question_type remain metadata, avoiding accidental model input.
    """
    normalized: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or not case.get("query"):
            raise ValueError(f"evaluation case {index} must contain a non-empty query")
        case_id = str(case.get("case_id", f"case_{index + 1}"))
        reference = {key: value for key, value in case.items() if key.startswith("expected_")}
        inputs = {"query": case["query"]}
        if "question_type" in case:
            inputs["question_type"] = case["question_type"]
        normalized.append(
            {
                "inputs": inputs,
                "reference_outputs": reference,
                "metadata": {
                    "case_id": case_id,
                    **({"question_type": case["question_type"]} if "question_type" in case else {}),
                },
            }
        )
    return normalized


def load_evaluation_cases(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list):
        raise ValueError("evaluation_cases JSON must be an array")
    return normalize_evaluation_cases(payload)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def sync_dataset(
    client: Any,
    dataset_name: str,
    cases: Iterable[Mapping[str, Any]],
    *,
    description: str | None = None,
) -> dict[str, Any]:
    """Idempotently create a dataset and upsert examples by metadata.case_id."""
    examples = normalize_evaluation_cases(cases)
    try:
        datasets = list(client.list_datasets(dataset_name=dataset_name, limit=2))
        dataset = (
            datasets[0]
            if datasets
            else client.create_dataset(dataset_name, description=description, data_type="kv")
        )
        dataset_id = _field(dataset, "id")
        existing = {
            str(_field(item, "metadata", {}).get("case_id")): item
            for item in client.list_examples(dataset_id=dataset_id)
            if _field(item, "metadata", {}).get("case_id") is not None
        }
        created = updated = 0
        for example in examples:
            old = existing.get(example["metadata"]["case_id"])
            if old is None:
                client.create_example(
                    inputs=example["inputs"],
                    outputs=example["reference_outputs"],
                    metadata=example["metadata"],
                    dataset_id=dataset_id,
                )
                created += 1
            else:
                client.update_example(
                    _field(old, "id"),
                    inputs=example["inputs"],
                    outputs=example["reference_outputs"],
                    metadata=example["metadata"],
                    dataset_id=dataset_id,
                )
                updated += 1
        return {
            "dataset": dataset,
            "dataset_id": dataset_id,
            "created": created,
            "updated": updated,
            "count": len(examples),
        }
    except Exception as error:
        raise LangSmithEvaluationError(
            f"LangSmith dataset sync failed for {dataset_name!r}: {error}"
        ) from error


def rank_scores(rank: Any) -> dict[str, float]:
    """Return the requested retrieval scores for a 1-based rank (or no hit)."""
    try:
        rank = int(rank)
    except (TypeError, ValueError):
        rank = 0
    return {
        "Hit@1": float(1 <= rank <= 1),
        "Hit@3": float(1 <= rank <= 3),
        "Hit@5": float(1 <= rank <= 5),
        "Hit@10": float(1 <= rank <= 10),
        "MRR": 1.0 / rank if rank > 0 else 0.0,
    }


def report_target(
    report: Mapping[str, Any], *, stage: str = "reranker"
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Create a dataset target that uploads one existing evaluation report."""
    by_query = {str(case.get("query")): case for case in report.get("cases", [])}

    def target(inputs: Mapping[str, Any]) -> dict[str, Any]:
        query = str(inputs.get("query", ""))
        if query not in by_query:
            raise KeyError(f"query is missing from evaluation report: {query!r}")
        case = by_query[query]
        return {
            "case_id": str(case.get("case_id", "")),
            "rank": case.get("ranks", {}).get(stage),
            "final_stage": case.get("final_stage"),
        }

    return target


def report_evaluators(
    report: Mapping[str, Any], *, stage: str = "reranker"
) -> list[Callable[..., dict[str, Any]]]:
    """Build LangSmith evaluators backed by an existing report's case ranks."""
    by_id = {str(case["case_id"]): case for case in report.get("cases", [])}
    by_query = {str(case.get("query")): case for case in report.get("cases", [])}

    def evaluator(
        inputs: Mapping[str, Any],
        outputs: Any,
        reference_outputs: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        case_id = str((reference_outputs or {}).get("case_id", inputs.get("case_id", "")))
        # The target may return case_id; otherwise the normalized query is stable.
        if isinstance(outputs, Mapping):
            case_id = str(outputs.get("case_id", case_id))
        case = by_id.get(case_id) or by_query.get(str(inputs.get("query", "")))
        rank = _field(outputs, "rank") if case is None else case.get("ranks", {}).get(stage)
        values = rank_scores(rank)
        return values

    # Individual evaluators make the experiment UI expose five named scores.
    result: list[Callable[..., dict[str, Any]]] = []
    for key in ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"):

        def metric(*args: Any, _key: str = key, **kwargs: Any) -> dict[str, Any]:
            value = evaluator(*args, **kwargs)[_key]
            return {"key": _key, "score": value}

        metric.__name__ = key.replace("@", "_at_").lower()
        result.append(metric)
    return result


def run_experiment(
    client: Any,
    target: Any,
    dataset_name: str,
    report: Mapping[str, Any],
    *,
    stage: str = "reranker",
    **kwargs: Any,
) -> Any:
    """Run the existing target through LangSmith's Experiment API."""
    try:
        return client.evaluate(
            target, data=dataset_name, evaluators=report_evaluators(report, stage=stage), **kwargs
        )
    except Exception as error:
        raise LangSmithEvaluationError(
            f"LangSmith experiment failed for {dataset_name!r}: {error}"
        ) from error


def _load_target(spec: str) -> Any:
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("target must be written as module:attribute")
    return getattr(importlib.import_module(module), name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync evaluation cases and run a LangSmith experiment"
    )
    parser.add_argument("cases", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("dataset")
    parser.add_argument(
        "--target",
        help="optional live target callable as module:attribute; default uploads the report",
    )
    parser.add_argument("--stage", default="reranker")
    args = parser.parse_args(argv)
    try:
        from langsmith import Client

        cases = json.loads(args.cases.read_text(encoding="utf-8"))
        report = json.loads(args.report.read_text(encoding="utf-8"))
        client = Client()
        result = sync_dataset(client, args.dataset, cases)
        target = (
            _load_target(args.target)
            if args.target
            else report_target(report, stage=args.stage)
        )
        run_experiment(client, target, args.dataset, report, stage=args.stage)
        print(json.dumps({"dataset_id": str(result["dataset_id"]), "count": result["count"]}))
        return 0
    except Exception as error:
        parser.exit(1, f"langsmith evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
