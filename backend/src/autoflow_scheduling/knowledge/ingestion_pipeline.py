"""End-to-end, incremental PDF ingestion CLI used by retrieval experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from autoflow_scheduling.observability import get_tracer

from .chunking import chunk_sections
from .pdf_parser import parse_pdf_pages
from .post_processor import process_chunks
from .retrieval_profile import DocumentRetrievalProfile
from .sectioning import split_sections

_PIPELINE_VERSION = 4
_STATE_FILENAME = "ingestion_state.json"
_ARTIFACT_FILENAMES = (
    "raw_pages.json",
    "clean_pages.json",
    "quality_report.json",
    "sections.json",
    "chunks.json",
    "ingestion_report.json",
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest without loading the whole PDF in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _pipeline_fingerprint(
    document_profile: DocumentRetrievalProfile,
    *,
    chunk_size: int,
    chunk_overlap: int,
    replacement_ratio_threshold: float,
    ocr_dpi: int,
    ocr_language: str,
) -> str:
    """Identify every setting that can change generated ingestion artifacts."""
    settings = {
        "pipeline_version": _PIPELINE_VERSION,
        "parser": "pymupdf4llm+plain_text_fallback+tesseract_ocr",
        "page_chunks": True,
        "write_images": False,
        "replacement_ratio_threshold": replacement_ratio_threshold,
        "ocr_dpi": ocr_dpi,
        "ocr_language": ocr_language,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "document_profile": {
            "document_id": document_profile.document_id,
            "content_type": document_profile.content_type.value,
            "metadata_confidence": document_profile.metadata_confidence,
        },
    }
    serialized = json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, value: Any) -> None:
    """Replace one JSON artifact atomically so interrupted writes stay invisible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, ensure_ascii=False, indent=2)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _can_skip(
    output_dir: Path,
    *,
    source_sha256: str,
    pipeline_fingerprint: str,
) -> bool:
    state = _read_json(output_dir / _STATE_FILENAME)
    if state is None:
        return False
    return (
        state.get("status") == "complete"
        and state.get("source_sha256") == source_sha256
        and state.get("pipeline_fingerprint") == pipeline_fingerprint
        and all((output_dir / filename).is_file() for filename in _ARTIFACT_FILENAMES)
    )


def _validate_document_artifacts(
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    document_id: str,
) -> None:
    """Reject ambiguous IDs and broken section/chunk references before persist."""
    section_ids = [str(section.get("section_id", "")) for section in sections]
    if len(section_ids) != len(set(section_ids)):
        raise ValueError(f"duplicate section_id in document {document_id!r}")
    for section, section_id in zip(sections, section_ids, strict=True):
        if section.get("document_id") != document_id:
            raise ValueError(f"section {section_id!r} has the wrong document_id")
        if not section_id.startswith(f"{document_id}:s"):
            raise ValueError(f"section {section_id!r} is not namespaced by document_id")

    known_sections = set(section_ids)
    chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(f"duplicate chunk_id in document {document_id!r}")
    for chunk, chunk_id in zip(chunks, chunk_ids, strict=True):
        section_id = str(chunk.get("section_id", ""))
        if chunk.get("document_id") != document_id:
            raise ValueError(f"chunk {chunk_id!r} has the wrong document_id")
        if section_id not in known_sections:
            raise ValueError(f"chunk {chunk_id!r} references unknown section {section_id!r}")
        if not chunk_id.startswith(f"{section_id}:c"):
            raise ValueError(f"chunk {chunk_id!r} is not namespaced by section_id")


def _ingest_pdf(
    pdf_path: Path,
    output_dir: Path,
    document_profile: DocumentRetrievalProfile,
    *,
    chunk_size: int = 1800,
    chunk_overlap: int = 200,
    replacement_ratio_threshold: float = 0.10,
    ocr_dpi: int = 200,
    ocr_language: str = "eng+chi_sim",
    force: bool = False,
) -> dict[str, Any]:
    """Parse one PDF and persist every stage with document-level incrementality.

    An unchanged source and unchanged pipeline configuration are skipped. A new
    or changed source replaces the complete artifact set using atomic writes;
    deterministic chunk IDs therefore never accumulate duplicate records.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    tracer = get_tracer()
    with tracer.stage("hash", metadata={"document_id": document_profile.document_id}) as stage:
        output_dir.mkdir(parents=True, exist_ok=True)
        source_sha256 = sha256_file(pdf_path)
        tracer.update(stage, status="complete", bytes=pdf_path.stat().st_size)
    pipeline_fingerprint = _pipeline_fingerprint(
        document_profile,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        replacement_ratio_threshold=replacement_ratio_threshold,
        ocr_dpi=ocr_dpi,
        ocr_language=ocr_language,
    )

    with tracer.stage("skip_check", metadata={"force": force}) as stage:
        skipped = not force and _can_skip(
            output_dir,
            source_sha256=source_sha256,
            pipeline_fingerprint=pipeline_fingerprint,
        )
        tracer.update(stage, status="skipped" if skipped else "continue")
    if skipped:
        previous_report = _read_json(output_dir / "ingestion_report.json") or {}
        return {
            **previous_report,
            "ingestion_status": "skipped",
            "skip_reason": "source_and_pipeline_unchanged",
        }

    timings: dict[str, float] = {}
    started = time.perf_counter()
    with tracer.stage("parse") as stage:
        raw_pages = parse_pdf_pages(
            pdf_path,
            replacement_ratio_threshold=replacement_ratio_threshold,
            ocr_dpi=ocr_dpi,
            ocr_language=ocr_language,
        )
        timings["parse_seconds"] = time.perf_counter() - started
        tracer.update(
            stage,
            status="complete",
            pages=len(raw_pages),
            seconds=timings["parse_seconds"],
            fallback_pages=sum(
                page.get("parsing", {}).get("strategy") != "pymupdf4llm_markdown"
                for page in raw_pages
            ),
            ocr_pages=sum(
                bool(page.get("parsing", {}).get("ocr_used")) for page in raw_pages
            ),
            failed_pages=sum(
                page.get("parsing", {}).get("status") == "failed" for page in raw_pages
            ),
        )

    started = time.perf_counter()
    with tracer.stage("clean") as stage:
        clean_pages, quality_report = process_chunks(raw_pages, pdf_path=pdf_path)
        timings["clean_seconds"] = time.perf_counter() - started
        tracer.update(
            stage, status="complete", pages=len(clean_pages), seconds=timings["clean_seconds"]
        )

    started = time.perf_counter()
    with tracer.stage("section") as stage:
        sections = split_sections(
            clean_pages, document_id=document_profile.document_id
        )
        timings["section_seconds"] = time.perf_counter() - started
        tracer.update(
            stage, status="complete", sections=len(sections), seconds=timings["section_seconds"]
        )

    started = time.perf_counter()
    with tracer.stage("chunk") as stage:
        chunks = chunk_sections(
            sections,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_profile=document_profile,
        )
        for chunk in chunks:
            chunk["source_sha256"] = source_sha256
        _validate_document_artifacts(
            sections, chunks, document_id=document_profile.document_id
        )
        timings["chunk_seconds"] = time.perf_counter() - started
        tracer.update(
            stage, status="complete", chunks=len(chunks), seconds=timings["chunk_seconds"]
        )

    artifacts = {
        "raw_pages.json": raw_pages,
        "clean_pages.json": clean_pages,
        "quality_report.json": quality_report.to_dict(),
        "sections.json": sections,
        "chunks.json": chunks,
    }
    with tracer.stage("persist") as stage:
        for filename, value in artifacts.items():
            _write_json_atomic(output_dir / filename, value)

        report = {
            "pdf": str(pdf_path),
            "document_id": document_profile.document_id,
            "document_content_type": document_profile.content_type.value,
            "metadata_confidence": document_profile.metadata_confidence,
            "source_sha256": source_sha256,
            "pipeline_fingerprint": pipeline_fingerprint,
            "pipeline_version": _PIPELINE_VERSION,
            "ingestion_status": "complete",
            "pages": len(raw_pages),
            "sections": len(sections),
            "chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "replacement_ratio_threshold": replacement_ratio_threshold,
            "ocr_dpi": ocr_dpi,
            "ocr_language": ocr_language,
            "parser_fallback_pages": sum(
                page.get("parsing", {}).get("strategy") != "pymupdf4llm_markdown"
                for page in raw_pages
            ),
            "ocr_pages": sum(
                bool(page.get("parsing", {}).get("ocr_used")) for page in raw_pages
            ),
            "parser_failed_pages": sum(
                page.get("parsing", {}).get("status") == "failed" for page in raw_pages
            ),
            "timings": {key: round(value, 4) for key, value in timings.items()},
        }
        _write_json_atomic(output_dir / "ingestion_report.json", report)

        # Written last: a complete state must only point at a fully replaced set of artifacts.
        state = {
            "status": "complete",
            "document_id": document_profile.document_id,
            "source_sha256": source_sha256,
            "pipeline_fingerprint": pipeline_fingerprint,
            "pipeline_version": _PIPELINE_VERSION,
            "artifacts": list(_ARTIFACT_FILENAMES),
        }
        _write_json_atomic(output_dir / _STATE_FILENAME, state)
        tracer.update(
            stage,
            status="complete",
            artifacts=len(artifacts) + 2,
            pages=len(raw_pages),
            sections=len(sections),
            chunks=len(chunks),
        )
    return report


def ingest_pdf(*args: Any, **kwargs: Any) -> dict[str, Any]:
    tracer = get_tracer()
    try:
        profile = args[2] if len(args) > 2 else kwargs["document_profile"]
        with tracer.root("ingestion", metadata={"document_id": profile.document_id}) as root:
            report = _ingest_pdf(*args, **kwargs)
            tracer.update(
                root,
                status=report.get("ingestion_status", "complete"),
                pages=report.get("pages", 0),
                sections=report.get("sections", 0),
                chunks=report.get("chunks", 0),
            )
            return report
    except Exception:
        # The adapter must never turn tracing failures into pipeline failures.
        raise
    finally:
        tracer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PDF through the full ingestion pipeline")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("document_profile", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--replacement-ratio-threshold", type=float, default=0.10)
    parser.add_argument("--ocr-dpi", type=int, default=200)
    parser.add_argument("--ocr-language", default="eng+chi_sim")
    parser.add_argument("--force", action="store_true", help="Rebuild even when nothing changed")
    args = parser.parse_args()

    profile = DocumentRetrievalProfile.from_dict(
        json.loads(args.document_profile.read_text(encoding="utf8"))
    )
    report = ingest_pdf(
        args.pdf,
        args.output_dir,
        profile,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        replacement_ratio_threshold=args.replacement_ratio_threshold,
        ocr_dpi=args.ocr_dpi,
        ocr_language=args.ocr_language,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
