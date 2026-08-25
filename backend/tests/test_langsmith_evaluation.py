import pytest

from autoflow_scheduling.observability.langsmith_evaluation import (
    LangSmithEvaluationError,
    normalize_evaluation_cases,
    rank_scores,
    report_evaluators,
    report_target,
    sync_dataset,
)

CASES = [
    {
        "case_id": "a",
        "query": "where?",
        "question_type": "diagnosis",
        "expected_title": "Valve",
        "expected_path_contains": ["Actuators"],
    }
]


class FakeDataset:
    id = "dataset-1"


class FakeExample:
    def __init__(self, example_id, metadata):
        self.id, self.metadata = example_id, metadata


class FakeClient:
    def __init__(self):
        self.dataset = None
        self.examples = []
        self.calls = []

    def list_datasets(self, **kwargs):
        return [] if self.dataset is None else [self.dataset]

    def create_dataset(self, name, **kwargs):
        self.calls.append(("create_dataset", name, kwargs))
        self.dataset = FakeDataset()
        return self.dataset

    def list_examples(self, **kwargs):
        return iter(self.examples)

    def create_example(self, **kwargs):
        self.calls.append(("create_example", kwargs))
        item = FakeExample(f"example-{len(self.examples)}", kwargs["metadata"])
        self.examples.append(item)
        return item

    def update_example(self, example_id, **kwargs):
        self.calls.append(("update_example", example_id, kwargs))
        for item in self.examples:
            if item.id == example_id:
                item.metadata = kwargs["metadata"]


def test_normalize_cases_uses_inputs_and_reference_outputs():
    result = normalize_evaluation_cases(CASES)
    assert result == [
        {
            "inputs": {"query": "where?", "question_type": "diagnosis"},
            "reference_outputs": {
                "expected_title": "Valve",
                "expected_path_contains": ["Actuators"],
            },
            "metadata": {"case_id": "a", "question_type": "diagnosis"},
        }
    ]


def test_sync_dataset_is_idempotent_and_updates_examples():
    client = FakeClient()
    first = sync_dataset(client, "demo", CASES)
    second = sync_dataset(client, "demo", [{**CASES[0], "expected_title": "Updated"}])
    assert first["created"] == 1 and second["updated"] == 1
    assert [call[0] for call in client.calls] == [
        "create_dataset",
        "create_example",
        "update_example",
    ]


def test_report_target_returns_case_rank():
    target = report_target(
        {"cases": [{"case_id": "a", "query": "where?", "ranks": {"rrf": 4}}]},
        stage="rrf",
    )
    assert target({"query": "where?"}) == {
        "case_id": "a",
        "rank": 4,
        "final_stage": None,
    }


def test_scores_and_named_report_evaluators():
    assert rank_scores(2) == {"Hit@1": 0.0, "Hit@3": 1.0, "Hit@5": 1.0, "Hit@10": 1.0, "MRR": 0.5}
    report = {"cases": [{"case_id": "a", "query": "where?", "ranks": {"reranker": 2}}]}
    values = [e({"query": "where?"}, {}, {}) for e in report_evaluators(report)]
    assert [(value["key"], value["score"]) for value in values] == [
        ("Hit@1", 0.0),
        ("Hit@3", 1.0),
        ("Hit@5", 1.0),
        ("Hit@10", 1.0),
        ("MRR", 0.5),
    ]


def test_sdk_errors_are_explicit():
    class Broken:
        def list_datasets(self, **kwargs):
            raise OSError("offline")

    with pytest.raises(LangSmithEvaluationError, match="dataset sync failed"):
        sync_dataset(Broken(), "demo", CASES)
