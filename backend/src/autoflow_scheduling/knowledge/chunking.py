"""Document chunking for both legacy pages and structure-aware sections."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .retrieval_profile import (
    DocumentContentType,
    DocumentRetrievalProfile,
    build_chunk_profile_text,
)

_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^]]*]\([^)]*\)")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")
_TITLE_KEY_RE = re.compile(r"[^\w]+", re.UNICODE)
_SUBHEADING_SEPARATOR = r"\n(?=#{4,6}[ \t]+)"
_NON_INDEXABLE_SECTION_TITLES = {
    "contents",
    "table of contents",
    "test your knowledge",
}


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Legacy paragraph splitter retained for the page-based baseline."""
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        prefix = current[-overlap:] if overlap and current else ""
        current = f"{prefix}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def _chunk_subheadings(text: str, known_subheadings: list[str]) -> list[str]:
    known = {heading.casefold(): heading for heading in known_subheadings}
    found: list[str] = []
    for _, title in _HEADING_RE.findall(text):
        canonical = known.get(title.strip().casefold())
        if canonical is not None and canonical not in found:
            found.append(canonical)
    return found


def _title_key(value: str) -> str:
    return _TITLE_KEY_RE.sub("", value.casefold())


def _has_meaningful_body(text: str, section: dict[str, Any]) -> bool:
    """Return whether a chunk contains text beyond structural shell content."""
    body = _HEADING_RE.sub("", text)
    body = _MARKDOWN_IMAGE_RE.sub("", body)
    body = _HTML_COMMENT_RE.sub("", body)
    structural_titles = {
        _title_key(str(value))
        for value in [section.get("title"), *(section.get("path") or [])]
        if value
    }
    for line in body.splitlines():
        candidate = line.strip().strip("|*-_`# ")
        if not candidate or _PAGE_NUMBER_RE.fullmatch(candidate):
            continue
        if _title_key(candidate) in structural_titles:
            continue
        if any(character.isalnum() for character in candidate):
            return True
    return False


def chunk_sections(
    sections: list[dict[str, Any]],
    chunk_size: int = 1800,
    chunk_overlap: int = 200,
    document_profile: DocumentRetrievalProfile | None = None,
) -> list[dict[str, Any]]:
    """Recursively split each section, preferring Markdown subheadings.

    A subheading is a preferred separator rather than a hard boundary: a
    section smaller than ``chunk_size`` remains one chunk. Calling the splitter
    once per section guarantees that chunks never cross section boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            _SUBHEADING_SEPARATOR,
            r"\n{2,}",
            "\n",
            "。",
            r"\. ",
            " ",
            "",
        ],
        is_separator_regex=True,
        keep_separator="start",
        strip_whitespace=True,
    )
    chunks: list[dict[str, Any]] = []
    for section in sections:
        section_title = str(section.get("title", "")).strip()
        if section_title.casefold() in _NON_INDEXABLE_SECTION_TITLES:
            continue
        text = str(section.get("text", "")).strip()
        if not text:
            continue
        section_id = str(section.get("section_id", "section"))
        known_subheadings = [str(value) for value in section.get("subheadings", [])]
        content_type = (
            document_profile.content_type
            if document_profile
            else DocumentContentType.OTHER
        )
        document_id = document_profile.document_id if document_profile else None
        if document_profile and section.get("document_id") != document_id:
            raise ValueError(
                f"section {section_id!r} does not belong to document {document_id!r}"
            )
        metadata_confidence = (
            document_profile.metadata_confidence if document_profile else None
        )
        section_chunks = splitter.split_text(text)
        active_subheading: str | None = None
        emitted_index = 0
        for chunk_text in section_chunks:
            subheadings = _chunk_subheadings(chunk_text, known_subheadings)
            if subheadings:
                active_subheading = subheadings[-1]
            if not _has_meaningful_body(chunk_text, section):
                continue
            emitted_index += 1
            section_path = list(section.get("path", []))
            profile_subheadings = subheadings or (
                [active_subheading] if active_subheading else []
            )
            index_text = build_chunk_profile_text(
                content_type=content_type,
                section_path=section_path,
                subheadings=profile_subheadings,
                chunk_text=chunk_text,
            )
            chunks.append(
                {
                    "chunk_id": f"{section_id}:c{emitted_index:03d}",
                    "document_id": document_id,
                    "document_content_type": content_type.value,
                    "metadata_confidence": metadata_confidence,
                    "section_id": section_id,
                    "section_title": section.get("title"),
                    "section_path": section_path,
                    "section_level": section.get("level"),
                    "section_page_start": section.get("page_start"),
                    "section_page_end": section.get("page_end"),
                    "chunk_index": emitted_index,
                    "subheading": active_subheading,
                    "subheadings": subheadings,
                    "text": chunk_text,
                    "index_text": index_text,
                    "quality": section.get("quality", {}),
                    "boundary": section.get("boundary", {}),
                    "parser": "pymupdf4llm+post_processor+sectioning",
                    "splitter": "langchain_recursive_character",
                }
            )
    return chunks


def chunk_pages(
    pages: list[dict[str, Any]],
    table_report: dict[str, Any] | None = None,
    max_chars: int = 1800,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    """Create legacy page chunks without replacing uncertain tables."""
    table_pages = {
        page["page"]: page.get("selected_status")
        for page in (table_report or {}).get("pages", [])
    }
    chunks: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages, 1):
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        metadata = page.get("metadata", {})
        page_table_quality = table_pages.get(page_index)
        for chunk_index, chunk_text in enumerate(_split_text(text, max_chars, overlap), 1):
            headings = _HEADING_RE.findall(chunk_text)
            chunks.append(
                {
                    "chunk_id": f"{metadata.get('title', 'document')}_p{page_index}_c{chunk_index}",
                    "page": page_index,
                    "section": headings[-1][1] if headings else None,
                    "text": chunk_text,
                    "source_type": "raw_text_with_table_fallback",
                    "table_quality": page_table_quality,
                    "table_fallback": "raw_text_preserved"
                    if page_table_quality in {"warning", "failed"}
                    else None,
                    "parser": "pymupdf4llm+post_processor",
                }
            )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk structure-aware PDF sections")
    parser.add_argument("sections", type=Path, help="sectioning sections.json")
    parser.add_argument("output", type=Path, help="output chunks.json")
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--document-profile",
        type=Path,
        help="JSON with document_id, content_type and metadata_confidence",
    )
    args = parser.parse_args()

    sections = json.loads(args.sections.read_text(encoding="utf8"))
    document_profile = None
    if args.document_profile:
        document_profile = DocumentRetrievalProfile.from_dict(
            json.loads(args.document_profile.read_text(encoding="utf8"))
        )
    chunks = chunk_sections(
        sections,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        document_profile=document_profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf8")
    print(
        json.dumps(
            {
                "sections": len(sections),
                "chunks": len(chunks),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
