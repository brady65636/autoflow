from threading import Barrier

from autoflow_scheduling.knowledge import hybrid_evaluation
from autoflow_scheduling.knowledge.hybrid_evaluation import (
    RerankerMode,
    _candidate_summary,
    _rank_deltas,
)


def test_parallel_hybrid_retrieval_waits_for_both_routes(monkeypatch) -> None:
    rendezvous = Barrier(2, timeout=1)
    calls: list[str] = []

    def dense_order(document_embeddings, query_vector):
        calls.append("dense")
        rendezvous.wait()
        return [0, 1, 2]

    def bm25_order(bm25, query):
        calls.append("bm25")
        rendezvous.wait()
        return [1, 2]

    monkeypatch.setattr(hybrid_evaluation, "_dense_order", dense_order)
    monkeypatch.setattr(hybrid_evaluation, "_bm25_order", bm25_order)

    result = hybrid_evaluation._parallel_hybrid_orders(
        object(),
        object(),
        object(),
        "N428 failure",
        dense_limit=3,
        bm25_limit=3,
    )

    assert set(calls) == {"dense", "bm25"}
    assert result["dense"] == [0, 1, 2]
    assert result["bm25"] == [1, 2]
    assert result["rrf"] == [1, 2, 0]
    assert all(
        result[key] >= 0
        for key in ("dense_seconds", "bm25_seconds", "rrf_seconds", "parallel_seconds")
    )


def test_weighted_rrf_prefers_the_higher_weight_route() -> None:
    assert hybrid_evaluation._rrf([0, 1], [1, 0], 2, dense_weight=2, bm25_weight=1) == [0, 1]
    assert hybrid_evaluation._rrf([0, 1], [1, 0], 2, dense_weight=1, bm25_weight=2) == [1, 0]


def test_business_rule_score_uses_content_compatibility() -> None:
    score, reasons = hybrid_evaluation._business_rule_score(
        {"question_type": "principle"},
        {
            "document_content_type": "technical_training",
            "metadata_confidence": 0.95,
            "quality": {"rag_text_status": "pass"},
        },
    )
    assert score == 0.99
    assert "compatibility=primary" in reasons


def test_candidate_telemetry_is_bounded_and_has_rank_deltas() -> None:
    chunks = [{"chunk_id": f"chunk-{index}", "text": "private"} for index in range(30)]
    summary = _candidate_summary(list(range(30)), chunks, {index: 0.5 for index in range(30)})

    assert len(summary) == 20
    assert summary[0] == {"chunk_id": "chunk-0", "rank": 1, "score": 0.5}
    assert all(set(item) == {"chunk_id", "rank", "score"} for item in summary)
    assert all("text" not in item for item in summary)
    deltas = _rank_deltas([0, 1, 2], [2, 0, 1], chunks)
    assert {item["chunk_id"]: item["rank_delta"] for item in deltas} == {
        "chunk-0": 1,
        "chunk-1": 1,
        "chunk-2": -2,
    }


def test_reranker_modes_are_explicit() -> None:
    assert {mode.value for mode in RerankerMode} == {
        "none",
        "cross_encoder",
        "llm",
    }
