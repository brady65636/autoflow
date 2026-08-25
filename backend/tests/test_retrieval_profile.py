import pytest

from autoflow_scheduling.knowledge.chunking import chunk_sections
from autoflow_scheduling.knowledge.retrieval_profile import (
    ContentCompatibility,
    DocumentContentType,
    DocumentRetrievalProfile,
    QuestionType,
    SearchProfile,
    build_query_profile_text,
    build_reranker_pair,
    content_type_compatibility,
)


def test_question_type_matches_document_content_type() -> None:
    assert (
        content_type_compatibility(
            QuestionType.PRINCIPLE, DocumentContentType.TECHNICAL_TRAINING
        )
        == ContentCompatibility.PRIMARY
    )
    assert (
        content_type_compatibility(
            QuestionType.COMPETITION, DocumentContentType.EQUIPMENT_LIST
        )
        == ContentCompatibility.NONE
    )


def test_query_profile_uses_free_text_required_knowledge() -> None:
    profile = SearchProfile(
        query="N428 失效后有什么影响？",
        question_type=QuestionType.DIAGNOSIS,
        required_knowledge="N428失效后的系统表现和降级状态",
    )

    text = build_query_profile_text(profile)

    assert "fault diagnosis / 故障诊断" in text
    assert "Required knowledge: N428失效后的系统表现和降级状态" in text
    assert "N428 失效后有什么影响？" in text


def test_chunk_index_text_uses_section_path_and_subheading_directly() -> None:
    section = {
        "document_id": "PDF-002",
        "section_id": "PDF-002:s0001",
        "title": "Valve for oil pressure control N428",
        "path": ["Oil circuit", "Valve for oil pressure control N428"],
        "subheadings": ["Task", "Effects upon failure"],
        "text": (
            "### Valve for oil pressure control N428\n\nDescription.\n\n"
            "##### Task\n\nControls oil pressure.\n\n"
            "##### Effects upon failure\n\nThe pump stays at high pressure."
        ),
    }
    document = DocumentRetrievalProfile(
        document_id="PDF-002",
        content_type=DocumentContentType.TECHNICAL_TRAINING,
        metadata_confidence=0.93,
    )

    chunks = chunk_sections(
        [section],
        chunk_size=1000,
        chunk_overlap=100,
        document_profile=document,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["document_id"] == "PDF-002"
    assert chunk["document_content_type"] == "technical_training"
    assert chunk["metadata_confidence"] == 0.93
    assert "section_capabilities" not in chunk
    assert "OEM technical training / 原厂技术培训" in chunk["index_text"]
    assert "Section: Oil circuit > Valve for oil pressure control N428" in chunk["index_text"]
    assert "Subheadings: Task | Effects upon failure" in chunk["index_text"]
    assert chunk["text"].startswith("### Valve")


def test_reranker_pair_uses_same_query_and_chunk_profiles() -> None:
    search = SearchProfile(
        query="N428失效后会怎样？",
        question_type=QuestionType.DIAGNOSIS,
        required_knowledge="失效后的系统行为",
    )

    query_text, document_text = build_reranker_pair(
        search,
        content_type=DocumentContentType.TECHNICAL_TRAINING,
        section_path=["Oil circuit", "N428"],
        subheadings=["Effects upon failure"],
        chunk_text="The pump remains at the high pressure stage.",
    )

    assert "Required knowledge: 失效后的系统行为" in query_text
    assert "Section: Oil circuit > N428" in document_text
    assert "Subheadings: Effects upon failure" in document_text


def test_search_profile_requires_free_text_knowledge_description() -> None:
    with pytest.raises(ValueError, match="required_knowledge"):
        SearchProfile(
            query="N428失效后会怎样？",
            question_type=QuestionType.DIAGNOSIS,
            required_knowledge=" ",
        )


def test_document_id_is_index_safe() -> None:
    with pytest.raises(ValueError, match="document_id"):
        DocumentRetrievalProfile(
            document_id="../unsafe:id",
            content_type=DocumentContentType.TECHNICAL_TRAINING,
            metadata_confidence=0.9,
        )


def test_document_metadata_confidence_is_validated() -> None:
    with pytest.raises(ValueError, match="metadata_confidence"):
        DocumentRetrievalProfile(
            document_id="PDF-002",
            content_type=DocumentContentType.TECHNICAL_TRAINING,
            metadata_confidence=1.2,
        )
