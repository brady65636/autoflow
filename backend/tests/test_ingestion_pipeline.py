import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from autoflow_scheduling.knowledge import ingestion_pipeline
from autoflow_scheduling.knowledge.retrieval_profile import (
    DocumentContentType,
    DocumentRetrievalProfile,
)


@dataclass
class _QualityReport:
    def to_dict(self) -> dict[str, str]:
        return {"status": "pass"}


@pytest.fixture
def profile() -> DocumentRetrievalProfile:
    return DocumentRetrievalProfile(
        document_id="PDF-001",
        content_type=DocumentContentType.TECHNICAL_TRAINING,
        metadata_confidence=0.95,
    )


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"parse": 0}

    def parse(path: str, **_: object) -> list[dict[str, object]]:
        calls["parse"] += 1
        text = Path(path).read_text(encoding="utf8")
        return [
            {
                "metadata": {"page_number": 1},
                "text": text,
                "parsing": {
                    "status": "complete",
                    "strategy": "pymupdf4llm_markdown",
                    "ocr_used": False,
                },
            }
        ]

    monkeypatch.setattr(ingestion_pipeline, "parse_pdf_pages", parse)
    monkeypatch.setattr(
        ingestion_pipeline,
        "process_chunks",
        lambda pages, pdf_path: (pages, _QualityReport()),
    )
    monkeypatch.setattr(
        ingestion_pipeline,
        "split_sections",
        lambda pages, document_id: [
            {
                "document_id": document_id,
                "section_id": f"{document_id}:s0001",
                "title": "Engine",
                "path": ["Engine"],
                "text": pages[0]["text"],
            }
        ],
    )
    monkeypatch.setattr(
        ingestion_pipeline,
        "chunk_sections",
        lambda sections, document_profile, **_: [
            {
                "chunk_id": f"{sections[0]['section_id']}:c001",
                "document_id": document_profile.document_id,
                "section_id": sections[0]["section_id"],
                "text": sections[0]["text"],
            }
        ],
    )
    return calls


def test_sha256_file_uses_file_content(tmp_path: Path) -> None:
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"abc")

    assert ingestion_pipeline.sha256_file(source) == hashlib.sha256(b"abc").hexdigest()


def test_unchanged_source_and_pipeline_are_skipped(
    tmp_path: Path,
    profile: DocumentRetrievalProfile,
    fake_pipeline: dict[str, int],
) -> None:
    source = tmp_path / "manual.pdf"
    source.write_text("first version", encoding="utf8")
    output = tmp_path / "index"

    first = ingestion_pipeline.ingest_pdf(source, output, profile)
    second = ingestion_pipeline.ingest_pdf(source, output, profile)

    assert first["ingestion_status"] == "complete"
    assert second["ingestion_status"] == "skipped"
    assert second["skip_reason"] == "source_and_pipeline_unchanged"
    assert fake_pipeline["parse"] == 1
    assert json.loads((output / "ingestion_state.json").read_text(encoding="utf8"))[
        "source_sha256"
    ] == hashlib.sha256(b"first version").hexdigest()


def test_changed_source_replaces_chunks_instead_of_appending(
    tmp_path: Path,
    profile: DocumentRetrievalProfile,
    fake_pipeline: dict[str, int],
) -> None:
    source = tmp_path / "manual.pdf"
    output = tmp_path / "index"
    source.write_text("old content", encoding="utf8")
    ingestion_pipeline.ingest_pdf(source, output, profile)

    source.write_text("new content", encoding="utf8")
    report = ingestion_pipeline.ingest_pdf(source, output, profile)
    chunks = json.loads((output / "chunks.json").read_text(encoding="utf8"))

    assert report["ingestion_status"] == "complete"
    assert fake_pipeline["parse"] == 2
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "PDF-001:s0001:c001"
    assert chunks[0]["text"] == "new content"
    assert chunks[0]["source_sha256"] == hashlib.sha256(b"new content").hexdigest()


def test_pipeline_setting_change_and_missing_artifact_force_rebuild(
    tmp_path: Path,
    profile: DocumentRetrievalProfile,
    fake_pipeline: dict[str, int],
) -> None:
    source = tmp_path / "manual.pdf"
    output = tmp_path / "index"
    source.write_text("same content", encoding="utf8")
    ingestion_pipeline.ingest_pdf(source, output, profile)

    ingestion_pipeline.ingest_pdf(source, output, profile, chunk_size=1200)
    (output / "sections.json").unlink()
    ingestion_pipeline.ingest_pdf(source, output, profile, chunk_size=1200)

    assert fake_pipeline["parse"] == 3
    assert (output / "sections.json").is_file()


def test_artifact_validation_rejects_cross_document_and_orphan_ids() -> None:
    sections = [
        {"document_id": "PDF-001", "section_id": "PDF-001:s0001"}
    ]
    wrong_document = [
        {
            "document_id": "PDF-002",
            "section_id": "PDF-001:s0001",
            "chunk_id": "PDF-001:s0001:c001",
        }
    ]
    with pytest.raises(ValueError, match="wrong document_id"):
        ingestion_pipeline._validate_document_artifacts(
            sections, wrong_document, document_id="PDF-001"
        )

    orphan = [
        {
            "document_id": "PDF-001",
            "section_id": "PDF-001:s9999",
            "chunk_id": "PDF-001:s9999:c001",
        }
    ]
    with pytest.raises(ValueError, match="unknown section"):
        ingestion_pipeline._validate_document_artifacts(
            sections, orphan, document_id="PDF-001"
        )
