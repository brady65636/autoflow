from __future__ import annotations

from fastapi.testclient import TestClient

from autoflow_scheduling.database import create_session_factory
from autoflow_scheduling.knowledge.hybrid_evaluation import _business_rule_score
from autoflow_scheduling.work_order_api import create_app


def test_retrieval_endpoint_requires_agent_or_advisor_auth(tmp_path) -> None:
    factory = create_session_factory(f"sqlite+pysqlite:///{tmp_path / 'api.db'}")
    client = TestClient(create_app(factory))
    response = client.post(
        "/api/knowledge/retrieve",
        json={"query": "What is the EA211 oil circuit?", "question_type": "principle"},
    )
    # HTTPBearer's default auto-error response is 422 for a missing header.
    assert response.status_code == 422


def test_retrieval_business_rule_score_is_bounded() -> None:
    score, reasons = _business_rule_score(
        {"question_type": "principle"},
        {
            "document_content_type": "technical_training",
            "metadata_confidence": 0.95,
            "quality": {"rag_text_status": "pass"},
        },
    )
    assert 0 <= score <= 1
    assert "compatibility=primary" in reasons
