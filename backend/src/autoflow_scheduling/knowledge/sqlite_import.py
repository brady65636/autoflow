"""Import parsed sections and retrieval chunks into SQLite with SHA256 verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from autoflow_scheduling.database import create_session_factory
from autoflow_scheduling.db_models import (
    KnowledgeChunkRow,
    KnowledgeDocumentRow,
    KnowledgeSectionRow,
)

_HASH_ALGORITHM = "sha256"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class KnowledgeImportError(ValueError):
    """Raised when artifacts have broken IDs, references, or hashes."""


class KnowledgeHashMismatch(KnowledgeImportError):
    """Raised when persisted knowledge no longer matches its stored SHA256."""


def _json_text(value: Any, default: Any) -> str:
    return json.dumps(
        default if value is None else value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _json_value(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf8")).hexdigest()


def _load_array(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KnowledgeImportError(f"cannot read {label} JSON: {path}") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise KnowledgeImportError(f"{label} JSON must be an array of objects")
    return value


def _document_metadata(items: Iterable[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items or []:
        document_id = str(item.get("document_id") or item.get("pdf_id") or "")
        if document_id:
            result[document_id] = item
    return result


def load_document_metadata(path: Path | None) -> list[dict[str, Any]]:
    """Load either a document array or the selection.json object used by corpus runs."""
    if path is None:
        return []
    try:
        value = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KnowledgeImportError(f"cannot read document metadata JSON: {path}") from error
    if isinstance(value, dict):
        value = value.get("selection", value.get("documents"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise KnowledgeImportError("document metadata JSON must contain an object array")
    return value


def _required_id(item: dict[str, Any], key: str, label: str) -> str:
    value = str(item.get(key, "")).strip()
    if not value:
        raise KnowledgeImportError(f"{label} is missing {key}")
    return value


def _consistent(values: Iterable[Any], *, label: str, document_id: str) -> Any:
    present = {value for value in values if value is not None}
    if len(present) > 1:
        raise KnowledgeImportError(f"conflicting {label} for document {document_id}")
    return next(iter(present), None)


def _validated_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if not _SHA256_RE.fullmatch(digest):
        raise KnowledgeImportError(f"{label} must be a lowercase 64-character SHA256")
    return digest


def _section_payload(section: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(section["section_id"]),
        "document_id": str(section["document_id"]),
        "title": str(section.get("title", "")),
        "path": section.get("path") or [],
        "level": section.get("level"),
        "page_start": section.get("page_start"),
        "page_end": section.get("page_end"),
        "text": str(section.get("text", "")),
        "subheadings": section.get("subheadings") or [],
        "quality": section.get("quality") or {},
        "boundary": section.get("boundary") or {},
    }


def _section_row_payload(row: KnowledgeSectionRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "document_id": row.document_id,
        "title": row.title,
        "path": _json_value(row.path_json, []),
        "level": row.level,
        "page_start": row.page_start,
        "page_end": row.page_end,
        "text": row.text,
        "subheadings": _json_value(row.subheadings_json, []),
        "quality": _json_value(row.quality_json, {}),
        "boundary": _json_value(row.boundary_json, {}),
    }


def _chunk_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(chunk["chunk_id"]),
        "document_id": str(chunk["document_id"]),
        "section_id": str(chunk["section_id"]),
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "content_type": str(chunk.get("document_content_type") or "other"),
        "metadata_confidence": (
            float(chunk["metadata_confidence"])
            if chunk.get("metadata_confidence") is not None
            else None
        ),
        "section_title": chunk.get("section_title"),
        "section_path": chunk.get("section_path") or [],
        "section_level": chunk.get("section_level"),
        "section_page_start": chunk.get("section_page_start"),
        "section_page_end": chunk.get("section_page_end"),
        "subheading": chunk.get("subheading"),
        "subheadings": chunk.get("subheadings") or [],
        "text": str(chunk.get("text", "")),
        "index_text": str(chunk.get("index_text") or chunk.get("text", "")),
        "quality": chunk.get("quality") or {},
        "boundary": chunk.get("boundary") or {},
        "parser": chunk.get("parser"),
        "splitter": chunk.get("splitter"),
        "source_sha256": chunk.get("source_sha256"),
    }


def _chunk_row_payload(row: KnowledgeChunkRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "document_id": row.document_id,
        "section_id": row.section_id,
        "chunk_index": row.chunk_index,
        "content_type": row.content_type,
        "metadata_confidence": row.metadata_confidence,
        "section_title": row.section_title,
        "section_path": _json_value(row.section_path_json, []),
        "section_level": row.section_level,
        "section_page_start": row.section_page_start,
        "section_page_end": row.section_page_end,
        "subheading": row.subheading,
        "subheadings": _json_value(row.subheadings_json, []),
        "text": row.text,
        "index_text": row.index_text,
        "quality": _json_value(row.quality_json, {}),
        "boundary": _json_value(row.boundary_json, {}),
        "parser": row.parser,
        "splitter": row.splitter,
        "source_sha256": row.source_sha256,
    }


def _document_payload(
    document: KnowledgeDocumentRow | dict[str, Any],
    section_hashes: Iterable[tuple[str, str]],
    chunk_hashes: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    get = document.get if isinstance(document, dict) else lambda key: getattr(document, key)
    return {
        "id": get("id"),
        "filename": get("filename"),
        "source_path": get("source_path"),
        "source_sha256": get("source_sha256"),
        "content_type": get("content_type"),
        "metadata_confidence": get("metadata_confidence"),
        "pipeline_version": get("pipeline_version"),
        "pipeline_fingerprint": get("pipeline_fingerprint"),
        "page_count": get("page_count"),
        "section_count": get("section_count"),
        "chunk_count": get("chunk_count"),
        "sections": sorted(section_hashes),
        "chunks": sorted(chunk_hashes),
    }


def _verify_document_hashes(session: Session, document_id: str) -> None:
    document = session.get(KnowledgeDocumentRow, document_id)
    if document is None:
        raise KnowledgeHashMismatch(f"missing document: {document_id}")
    if document.hash_algorithm != _HASH_ALGORITHM or not document.artifact_sha256:
        raise KnowledgeHashMismatch(f"document {document_id} has no active SHA256")

    sections = list(
        session.scalars(
            select(KnowledgeSectionRow)
            .where(KnowledgeSectionRow.document_id == document_id)
            .order_by(KnowledgeSectionRow.id)
        )
    )
    chunks = list(
        session.scalars(
            select(KnowledgeChunkRow)
            .where(KnowledgeChunkRow.document_id == document_id)
            .order_by(KnowledgeChunkRow.id)
        )
    )
    section_hashes = []
    for row in sections:
        actual = _sha256(_section_row_payload(row))
        if row.content_sha256 != actual:
            raise KnowledgeHashMismatch(f"section hash mismatch: {row.id}")
        section_hashes.append((row.id, actual))
    chunk_hashes = []
    for row in chunks:
        actual = _sha256(_chunk_row_payload(row))
        if row.content_sha256 != actual:
            raise KnowledgeHashMismatch(f"chunk hash mismatch: {row.id}")
        chunk_hashes.append((row.id, actual))

    if document.section_count != len(sections) or document.chunk_count != len(chunks):
        raise KnowledgeHashMismatch(f"artifact count mismatch: {document_id}")
    actual = _sha256(_document_payload(document, section_hashes, chunk_hashes))
    if document.artifact_sha256 != actual:
        raise KnowledgeHashMismatch(f"document hash mismatch: {document_id}")


def verify_knowledge_hashes(
    session: Session, document_ids: Iterable[str] | None = None
) -> dict[str, int]:
    """Recompute all persisted hashes and fail on the first integrity mismatch."""
    ids = list(document_ids or session.scalars(select(KnowledgeDocumentRow.id).order_by(
        KnowledgeDocumentRow.id
    )))
    for document_id in ids:
        _verify_document_hashes(session, str(document_id))
    return {"verified_documents": len(ids), "failed_documents": 0}


def import_knowledge_artifacts(
    session: Session,
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    documents: Iterable[dict[str, Any]] | None = None,
    pipeline_version: int | None = None,
) -> dict[str, int]:
    """Hash, import, read back, and verify each changed document atomically."""
    section_ids: set[str] = set()
    sections_by_document: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        section_id = _required_id(section, "section_id", "section")
        document_id = _required_id(section, "document_id", f"section {section_id}")
        if section_id in section_ids:
            raise KnowledgeImportError(f"duplicate section_id: {section_id}")
        if not section_id.startswith(f"{document_id}:s"):
            raise KnowledgeImportError(f"section {section_id} is not namespaced by {document_id}")
        section_ids.add(section_id)
        sections_by_document[document_id].append(section)

    chunk_ids: set[str] = set()
    chunks_by_document: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunk_id = _required_id(chunk, "chunk_id", "chunk")
        document_id = _required_id(chunk, "document_id", f"chunk {chunk_id}")
        section_id = _required_id(chunk, "section_id", f"chunk {chunk_id}")
        if chunk_id in chunk_ids:
            raise KnowledgeImportError(f"duplicate chunk_id: {chunk_id}")
        if section_id not in section_ids:
            raise KnowledgeImportError(f"chunk {chunk_id} references unknown section {section_id}")
        if not chunk_id.startswith(f"{section_id}:c"):
            raise KnowledgeImportError(f"chunk {chunk_id} is not namespaced by {section_id}")
        chunk_ids.add(chunk_id)
        chunks_by_document[document_id].append(chunk)

    document_ids = set(sections_by_document) | set(chunks_by_document)
    if not document_ids:
        raise KnowledgeImportError("no documents found in artifacts")
    if set(chunks_by_document) - set(sections_by_document):
        raise KnowledgeImportError("one or more chunk documents have no sections")

    metadata = _document_metadata(documents)
    plans: dict[str, dict[str, Any]] = {}
    for document_id in sorted(document_ids):
        document_sections = sections_by_document[document_id]
        document_chunks = chunks_by_document[document_id]
        info = metadata.get(document_id, {})
        content_type = _consistent(
            (chunk.get("document_content_type") for chunk in document_chunks),
            label="content type",
            document_id=document_id,
        ) or str(info.get("content_type") or "other")
        confidence = _consistent(
            (chunk.get("metadata_confidence") for chunk in document_chunks),
            label="metadata confidence",
            document_id=document_id,
        )
        source_sha256 = _consistent(
            (chunk.get("source_sha256") for chunk in document_chunks),
            label="source SHA256",
            document_id=document_id,
        ) or info.get("actual_sha256") or info.get("manifest_sha256")
        manifest_sha256 = info.get("actual_sha256") or info.get("manifest_sha256")
        source_sha256 = _validated_sha256(
            source_sha256, label=f"source hash for document {document_id}"
        )
        if manifest_sha256:
            manifest_sha256 = _validated_sha256(
                manifest_sha256, label=f"manifest hash for document {document_id}"
            )
        if manifest_sha256 and source_sha256 != manifest_sha256:
            raise KnowledgeImportError(f"source SHA256 mismatch for document {document_id}")
        page_numbers = [
            int(value)
            for section in document_sections
            for value in (section.get("page_start"), section.get("page_end"))
            if value is not None
        ]
        document_values = {
            "id": document_id,
            "filename": info.get("filename"),
            "source_path": info.get("relative_path") or info.get("source_path"),
            "source_sha256": source_sha256,
            "content_type": str(content_type),
            "metadata_confidence": float(confidence) if confidence is not None else None,
            "pipeline_version": pipeline_version,
            "pipeline_fingerprint": info.get("pipeline_fingerprint"),
            "page_count": max(page_numbers, default=0),
            "section_count": len(document_sections),
            "chunk_count": len(document_chunks),
        }
        section_hashes = {
            str(section["section_id"]): _sha256(_section_payload(section))
            for section in document_sections
        }
        chunk_hashes = {
            str(chunk["chunk_id"]): _sha256(_chunk_payload(chunk))
            for chunk in document_chunks
        }
        artifact_sha256 = _sha256(
            _document_payload(document_values, section_hashes.items(), chunk_hashes.items())
        )
        plans[document_id] = {
            "document": document_values,
            "section_hashes": section_hashes,
            "chunk_hashes": chunk_hashes,
            "artifact_sha256": artifact_sha256,
        }

    changed_ids: list[str] = []
    skipped_ids: list[str] = []
    for document_id, plan in plans.items():
        existing = session.get(KnowledgeDocumentRow, document_id)
        if existing is not None and existing.artifact_sha256 == plan["artifact_sha256"]:
            try:
                _verify_document_hashes(session, document_id)
            except KnowledgeHashMismatch:
                changed_ids.append(document_id)
            else:
                skipped_ids.append(document_id)
        else:
            changed_ids.append(document_id)

    now = datetime.now(timezone.utc)
    try:
        if changed_ids:
            session.execute(
                delete(KnowledgeChunkRow).where(KnowledgeChunkRow.document_id.in_(changed_ids))
            )
            session.execute(
                delete(KnowledgeSectionRow).where(
                    KnowledgeSectionRow.document_id.in_(changed_ids)
                )
            )
            session.execute(
                delete(KnowledgeDocumentRow).where(KnowledgeDocumentRow.id.in_(changed_ids))
            )
            for document_id in changed_ids:
                plan = plans[document_id]
                session.add(
                    KnowledgeDocumentRow(
                        **plan["document"],
                        artifact_sha256=plan["artifact_sha256"],
                        hash_algorithm=_HASH_ALGORITHM,
                        hash_verified_at=None,
                        ingestion_status="complete",
                        imported_at=now,
                        updated_at=now,
                    )
                )
            session.flush()

            for section in sections:
                document_id = str(section["document_id"])
                if document_id not in changed_ids:
                    continue
                session.add(
                    KnowledgeSectionRow(
                        id=str(section["section_id"]),
                        document_id=document_id,
                        title=str(section.get("title", "")),
                        path_json=_json_text(section.get("path"), []),
                        level=section.get("level"),
                        page_start=section.get("page_start"),
                        page_end=section.get("page_end"),
                        text=str(section.get("text", "")),
                        subheadings_json=_json_text(section.get("subheadings"), []),
                        quality_json=_json_text(section.get("quality"), {}),
                        boundary_json=_json_text(section.get("boundary"), {}),
                        content_sha256=plans[document_id]["section_hashes"][
                            str(section["section_id"])
                        ],
                    )
                )
            session.flush()

            for chunk in chunks:
                document_id = str(chunk["document_id"])
                if document_id not in changed_ids:
                    continue
                payload = _chunk_payload(chunk)
                session.add(
                    KnowledgeChunkRow(
                        id=payload["id"],
                        document_id=document_id,
                        section_id=payload["section_id"],
                        chunk_index=payload["chunk_index"],
                        content_type=payload["content_type"],
                        metadata_confidence=payload["metadata_confidence"],
                        section_title=payload["section_title"],
                        section_path_json=_json_text(payload["section_path"], []),
                        section_level=payload["section_level"],
                        section_page_start=payload["section_page_start"],
                        section_page_end=payload["section_page_end"],
                        subheading=payload["subheading"],
                        subheadings_json=_json_text(payload["subheadings"], []),
                        text=payload["text"],
                        index_text=payload["index_text"],
                        quality_json=_json_text(payload["quality"], {}),
                        boundary_json=_json_text(payload["boundary"], {}),
                        parser=payload["parser"],
                        splitter=payload["splitter"],
                        source_sha256=payload["source_sha256"],
                        content_sha256=plans[document_id]["chunk_hashes"][payload["id"]],
                    )
                )
            session.flush()
            for document_id in changed_ids:
                _verify_document_hashes(session, document_id)
                session.get(KnowledgeDocumentRow, document_id).hash_verified_at = now
            session.commit()
    except Exception:
        session.rollback()
        raise

    return {
        "documents": len(document_ids),
        "imported_documents": len(changed_ids),
        "skipped_documents": len(skipped_ids),
        "sections": len(sections),
        "chunks": len(chunks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import and SHA256-verify RAG artifacts")
    parser.add_argument("sections", type=Path)
    parser.add_argument("chunks", type=Path)
    parser.add_argument("--documents", type=Path, help="Optional selection/document metadata JSON")
    parser.add_argument("--database-url", help="Defaults to AUTOFLOW_DATABASE_URL/autoflow.db")
    parser.add_argument("--pipeline-version", type=int)
    args = parser.parse_args(argv)

    factory = create_session_factory(args.database_url)
    with factory() as session:
        summary = import_knowledge_artifacts(
            session,
            _load_array(args.sections, "sections"),
            _load_array(args.chunks, "chunks"),
            documents=load_document_metadata(args.documents),
            pipeline_version=args.pipeline_version,
        )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def verify_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify all persisted RAG SHA256 hashes")
    parser.add_argument("--database-url", help="Defaults to AUTOFLOW_DATABASE_URL/autoflow.db")
    parser.add_argument("--document-id", action="append", dest="document_ids")
    args = parser.parse_args(argv)
    factory = create_session_factory(args.database_url)
    with factory() as session:
        summary = verify_knowledge_hashes(session, args.document_ids)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
