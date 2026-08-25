import json

import pytest
from sqlalchemy import func, select

from autoflow_scheduling.database import create_session_factory
from autoflow_scheduling.db_models import (
    KnowledgeChunkRow,
    KnowledgeDocumentRow,
    KnowledgeSectionRow,
)
from autoflow_scheduling.knowledge.sqlite_import import (
    KnowledgeHashMismatch,
    KnowledgeImportError,
    import_knowledge_artifacts,
    verify_knowledge_hashes,
)


def _section(text: str = "Oil pressure section") -> dict:
    return {
        "section_id": "PDF-001:s0001",
        "document_id": "PDF-001",
        "title": "Oil pressure",
        "path": ["Engine", "Oil pressure"],
        "level": 2,
        "page_start": 3,
        "page_end": 4,
        "text": text,
        "subheadings": ["Failure"],
        "quality": {"rag_text_status": "pass", "table_quality": None},
        "boundary": {"source": "pdf_toc+markdown", "confidence": 1.0},
    }


def _chunk(chunk_id: str = "PDF-001:s0001:c001", text: str = "High pressure") -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "PDF-001",
        "document_content_type": "technical_training",
        "metadata_confidence": 0.95,
        "section_id": "PDF-001:s0001",
        "section_title": "Oil pressure",
        "section_path": ["Engine", "Oil pressure"],
        "section_level": 2,
        "section_page_start": 3,
        "section_page_end": 4,
        "chunk_index": int(chunk_id.rsplit("c", 1)[1]),
        "subheading": "Failure",
        "subheadings": ["Failure"],
        "text": text,
        "index_text": f"Section: Oil pressure\nContent: {text}",
        "quality": {"rag_text_status": "pass"},
        "boundary": {"source": "pdf_toc+markdown"},
        "parser": "pymupdf4llm+post_processor+sectioning",
        "splitter": "langchain_recursive_character",
        "source_sha256": "a" * 64,
    }


def test_import_persists_documents_sections_and_chunks(tmp_path):
    factory = create_session_factory(f"sqlite+pysqlite:///{tmp_path / 'knowledge.db'}")
    documents = [
        {
            "pdf_id": "PDF-001",
            "filename": "engine.pdf",
            "relative_path": "framework/engine.pdf",
            "content_type": "technical_training",
        }
    ]
    with factory() as session:
        summary = import_knowledge_artifacts(
            session, [_section()], [_chunk()], documents=documents, pipeline_version=4
        )

        assert summary == {
            "documents": 1,
            "imported_documents": 1,
            "skipped_documents": 0,
            "sections": 1,
            "chunks": 1,
        }
        document = session.get(KnowledgeDocumentRow, "PDF-001")
        assert document is not None
        assert document.filename == "engine.pdf"
        assert document.page_count == 4
        assert document.pipeline_version == 4
        assert len(document.source_sha256) == 64
        assert len(document.artifact_sha256) == 64
        assert document.hash_algorithm == "sha256"
        assert document.hash_verified_at is not None
        section = session.get(KnowledgeSectionRow, "PDF-001:s0001")
        assert json.loads(section.path_json) == ["Engine", "Oil pressure"]
        assert len(section.content_sha256) == 64
        chunk = session.get(KnowledgeChunkRow, "PDF-001:s0001:c001")
        assert chunk.section_id == section.id
        assert chunk.index_text.endswith("High pressure")
        assert len(chunk.content_sha256) == 64
        assert verify_knowledge_hashes(session) == {
            "verified_documents": 1,
            "failed_documents": 0,
        }


def test_reimport_replaces_stale_chunks_in_one_document(tmp_path):
    factory = create_session_factory(f"sqlite+pysqlite:///{tmp_path / 'knowledge.db'}")
    with factory() as session:
        import_knowledge_artifacts(
            session,
            [_section("old section")],
            [_chunk(), _chunk("PDF-001:s0001:c002", "stale")],
            pipeline_version=3,
        )
        import_knowledge_artifacts(
            session,
            [_section("new section")],
            [_chunk(text="new chunk")],
            pipeline_version=4,
        )

        assert session.scalar(select(func.count()).select_from(KnowledgeChunkRow)) == 1
        document = session.get(KnowledgeDocumentRow, "PDF-001")
        assert document.chunk_count == 1
        assert document.pipeline_version == 4
        assert session.get(KnowledgeSectionRow, "PDF-001:s0001").text == "new section"
        assert session.get(KnowledgeChunkRow, "PDF-001:s0001:c001").text == "new chunk"
        assert session.get(KnowledgeChunkRow, "PDF-001:s0001:c002") is None


def test_identical_reimport_is_hash_verified_and_skipped(tmp_path):
    factory = create_session_factory(f"sqlite+pysqlite:///{tmp_path / 'knowledge.db'}")
    with factory() as session:
        import_knowledge_artifacts(session, [_section()], [_chunk()], pipeline_version=4)
        imported_at = session.get(KnowledgeDocumentRow, "PDF-001").imported_at
        summary = import_knowledge_artifacts(
            session, [_section()], [_chunk()], pipeline_version=4
        )

        assert summary["imported_documents"] == 0
        assert summary["skipped_documents"] == 1
        assert session.get(KnowledgeDocumentRow, "PDF-001").imported_at == imported_at


def test_tampered_chunk_is_rebuilt_instead_of_false_skip(tmp_path):
    factory = create_session_factory(f"sqlite+pysqlite:///{tmp_path / 'knowledge.db'}")
    with factory() as session:
        import_knowledge_artifacts(session, [_section()], [_chunk()], pipeline_version=4)
        session.get(KnowledgeChunkRow, "PDF-001:s0001:c001").text = "tampered"
        session.commit()
        with pytest.raises(KnowledgeHashMismatch, match="chunk hash mismatch"):
            verify_knowledge_hashes(session)

        summary = import_knowledge_artifacts(
            session, [_section()], [_chunk()], pipeline_version=4
        )
        assert summary["imported_documents"] == 1
        assert summary["skipped_documents"] == 0
        assert session.get(KnowledgeChunkRow, "PDF-001:s0001:c001").text == "High pressure"
        verify_knowledge_hashes(session)


def test_invalid_chunk_reference_does_not_delete_existing_data(tmp_path):
    factory = create_session_factory(f"sqlite+pysqlite:///{tmp_path / 'knowledge.db'}")
    with factory() as session:
        import_knowledge_artifacts(session, [_section()], [_chunk()])
        broken = _chunk()
        broken["section_id"] = "PDF-001:s9999"

        with pytest.raises(KnowledgeImportError, match="unknown section"):
            import_knowledge_artifacts(session, [_section("replacement")], [broken])

        assert session.get(KnowledgeSectionRow, "PDF-001:s0001").text == "Oil pressure section"
        assert session.get(KnowledgeChunkRow, "PDF-001:s0001:c001") is not None
