"""Batch PDF ingestion coordinator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from autoflow_scheduling.database import create_session_factory

from .ingestion_pipeline import ingest_pdf
from .retrieval_profile import DocumentContentType, DocumentRetrievalProfile
from .sqlite_import import import_knowledge_artifacts

PIPELINE_VERSION = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _profile(entry: dict[str, Any]) -> DocumentRetrievalProfile:
    document_id = str(entry["pdf_id"])
    content_type = (
        DocumentContentType.OFFICIAL_NEWS
        if document_id == "PDF-001"
        else DocumentContentType.TECHNICAL_TRAINING
    )
    return DocumentRetrievalProfile(document_id, content_type, 0.95)


def _worker(
    position: int,
    entry: dict[str, Any],
    corpus_root: str,
    output_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    force: bool,
) -> dict[str, Any]:
    """Top-level picklable process worker; failures stay local to one PDF."""
    document_id = str(entry.get("pdf_id", ""))
    try:
        relative = Path(str(entry["relative_path"]).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("relative_path escapes corpus_root")
        pdf_path = Path(corpus_root) / relative
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)
        actual_sha256 = _sha256(pdf_path)
        manifest_sha256 = str(entry["sha256"])
        if actual_sha256 != manifest_sha256:
            raise ValueError(
                f"sha256 mismatch: manifest={manifest_sha256}, actual={actual_sha256}"
            )
        profile = _profile(entry)
        document_dir = Path(output_dir) / document_id
        report = ingest_pdf(
            pdf_path,
            document_dir,
            profile,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force=force,
        )
        expected_pages = int(entry["page_count"])
        if int(report.get("pages", -1)) != expected_pages:
            raise ValueError(
                f"page count mismatch: manifest={expected_pages}, "
                f"parsed={report.get('pages')}"
            )
        quality = _read_json(document_dir / "quality_report.json")
        return {
            "position": position,
            "pdf_id": document_id,
            "status": "success",
            "actual_sha256": actual_sha256,
            "manifest_sha256": manifest_sha256,
            "report": report,
            "quality": {
                key: quality.get(key)
                for key in (
                    "page_count",
                    "rag_pass_pages",
                    "rag_warning_pages",
                    "quarantine_pages",
                    "parser_fallback_pages",
                    "ocr_pages",
                    "parser_failed_pages",
                )
            },
        }
    except Exception as error:
        return {
            "position": position,
            "pdf_id": document_id,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }


def _validate(
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    expected_document_ids: list[str] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    section_ids = [str(item.get("section_id", "")) for item in sections]
    chunk_ids = [str(item.get("chunk_id", "")) for item in chunks]
    if len(section_ids) != len(set(section_ids)):
        errors.append("duplicate section_id")
    if len(chunk_ids) != len(set(chunk_ids)):
        errors.append("duplicate chunk_id")
    section_set = set(section_ids)
    section_documents = {str(item.get("document_id", "")) for item in sections}
    chunk_documents = {str(item.get("document_id", "")) for item in chunks}
    for document_id in expected_document_ids or []:
        if document_id not in section_documents:
            errors.append(f"document has no sections: {document_id}")
        if document_id not in chunk_documents:
            errors.append(f"document has no chunks: {document_id}")
    for section in sections:
        sid, did = str(section.get("section_id", "")), str(section.get("document_id", ""))
        if not did or not sid.startswith(f"{did}:s"):
            errors.append(f"invalid section namespace: {sid}")
    for chunk in chunks:
        cid = str(chunk.get("chunk_id", ""))
        sid, did = str(chunk.get("section_id", "")), str(chunk.get("document_id", ""))
        if sid not in section_set:
            errors.append(f"unknown section reference: {sid}")
        if not cid.startswith(f"{sid}:c") or not sid.startswith(f"{did}:s"):
            errors.append(f"invalid chunk reference: {cid}")
    replacement_chunks = sum("�" in str(chunk.get("text", "")) for chunk in chunks)
    if replacement_chunks:
        warnings.append(f"replacement characters found in {replacement_chunks} chunks")
    return {
        "status": "pass" if not errors else "fail",
        "documents": len(section_documents | chunk_documents),
        "section_count": len(sections),
        "chunk_count": len(chunks),
        "replacement_character_chunks": replacement_chunks,
        "errors": errors,
        "warnings": warnings,
    }


def run_batch(
    manifest_path: str | Path,
    corpus_root: str | Path,
    output_dir: str | Path,
    offset: int = 0,
    limit: int = 20,
    max_workers: int = 20,
    database_url: str | None = None,
    chunk_size: int = 1800,
    chunk_overlap: int = 200,
    force: bool = False,
) -> dict[str, Any]:
    """Ingest a bounded manifest slice, then atomically publish its batch."""
    if offset < 0 or limit < 0 or max_workers <= 0:
        raise ValueError("offset/limit must be non-negative and max_workers positive")
    manifest_path, corpus_root, output_dir = map(Path, (manifest_path, corpus_root, output_dir))
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, list) or not all(isinstance(item, dict) for item in manifest):
        raise ValueError("manifest must be a JSON array of objects")
    selected = manifest[offset : offset + limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("combined_sections.json", "combined_chunks.json"):
        (output_dir / filename).unlink(missing_ok=True)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    if selected:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(selected))) as pool:
            futures = {
                pool.submit(
                    _worker,
                    index,
                    entry,
                    str(corpus_root),
                    str(output_dir),
                    chunk_size,
                    chunk_overlap,
                    force,
                ): (index, str(entry.get("pdf_id", "")))
                for index, entry in enumerate(selected, offset)
            }
            for future in as_completed(futures):
                position, document_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as error:
                    results.append(
                        {
                            "position": position,
                            "pdf_id": document_id,
                            "status": "failed",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
    results.sort(key=lambda item: item["position"])
    successful = [item for item in results if item["status"] == "success"]
    report: dict[str, Any] = {
        "manifest": str(manifest_path),
        "offset": offset,
        "limit": limit,
        "selected": len(selected),
        "succeeded": len(successful),
        "failed": len(results) - len(successful),
        "max_workers": max_workers,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    if len(successful) == len(selected) and selected:
        sections = sorted(
            [
                section
                for item in successful
                for section in _read_json(output_dir / item["pdf_id"] / "sections.json")
            ],
            key=lambda item: (
                str(item.get("document_id", "")),
                str(item.get("section_id", "")),
            ),
        )
        chunks = sorted(
            [
                chunk
                for item in successful
                for chunk in _read_json(output_dir / item["pdf_id"] / "chunks.json")
            ],
            key=lambda item: (
                str(item.get("document_id", "")),
                str(item.get("chunk_id", "")),
            ),
        )
        expected_ids = [str(entry["pdf_id"]) for entry in selected]
        validation = _validate(sections, chunks, expected_ids)
        result_by_id = {item["pdf_id"]: item for item in successful}
        selection = []
        for entry in selected:
            result = result_by_id[str(entry["pdf_id"])]
            value = dict(entry)
            value.update(
                {
                    "content_type": _profile(entry).content_type.value,
                    "metadata_confidence": 0.95,
                    "manifest_sha256": result["manifest_sha256"],
                    "actual_sha256": result["actual_sha256"],
                    "pipeline_fingerprint": result["report"].get("pipeline_fingerprint"),
                }
            )
            selection.append(value)
        _atomic_json(output_dir / "selection.json", selection)
        _atomic_json(output_dir / "validation.json", validation)
        report["validation"] = validation
        if validation["status"] == "pass":
            _atomic_json(output_dir / "combined_sections.json", sections)
            _atomic_json(output_dir / "combined_chunks.json", chunks)
            if database_url:
                factory = create_session_factory(database_url)
                with factory() as session:
                    report["sqlite_import"] = import_knowledge_artifacts(
                        session,
                        sections,
                        chunks,
                        documents=selection,
                        pipeline_version=PIPELINE_VERSION,
                    )
    _atomic_json(output_dir / "batch_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch PDF ingestion")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--database-url")
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    report = run_batch(
        args.manifest,
        args.corpus_root,
        args.output_dir,
        offset=args.offset,
        limit=args.limit,
        max_workers=args.max_workers,
        database_url=args.database_url,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False))
    passed = (
        report.get("failed") == 0
        and report.get("validation", {}).get("status") == "pass"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
