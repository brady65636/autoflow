"""Structure-aware section splitting for cleaned PDF page chunks.

The splitter keeps page provenance, uses PDF bookmarks as the canonical
hierarchy when available, and treats low-level labels such as "Task" as
anchors inside a section instead of standalone sections.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from .retrieval_profile import validate_document_id

_HEADING_RE = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$")
_PICTURE_BLOCK_RE = re.compile(
    r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TABLE_RE = re.compile(r"(?is)<table\b.*?</table>")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^]]*]\([^)]*\)")
_MARKUP_RE = re.compile(r"<!--.*?-->|<[^>]+>", re.DOTALL)
_SPACE_RE = re.compile(r"\s+")
_TITLE_KEY_RE = re.compile(r"[^\w]+", re.UNICODE)

# These labels carry useful local structure, but usually have no independent
# meaning without their parent component/system title.
_INTERNAL_LABELS = {
    "advantages",
    "application",
    "applications",
    "design",
    "effects upon failure",
    "effects upon signal failure",
    "function",
    "functions",
    "installation position",
    "note",
    "notes",
    "overview",
    "purpose",
    "signal use",
    "special features",
    "task",
    "tasks",
    "technical features",
    "warning",
    "warnings",
}


@dataclass(frozen=True)
class TocEntry:
    level: int
    title: str
    page: int
    path: tuple[str, ...]


@dataclass(frozen=True)
class Heading:
    start: int
    end: int
    markdown_level: int
    title: str


def _title_key(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).casefold()
    return _TITLE_KEY_RE.sub("", normalized)


def _clean_title(title: str) -> str:
    return _SPACE_RE.sub(" ", title).strip().strip("#").strip()


def _page_number(page: dict[str, Any], fallback: int) -> int:
    value = page.get("metadata", {}).get("page_number", fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _collect_toc(pages: Iterable[dict[str, Any]]) -> list[TocEntry]:
    raw_entries: list[tuple[int, str, int]] = []
    seen: set[tuple[int, str, int]] = set()
    for page in pages:
        for item in page.get("toc_items", []) or []:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            try:
                level, title, target_page = int(item[0]), _clean_title(str(item[1])), int(item[2])
            except (TypeError, ValueError):
                continue
            key = (level, _title_key(title), target_page)
            if level < 1 or not title or key in seen:
                continue
            seen.add(key)
            raw_entries.append((level, title, target_page))

    # PyMuPDF page chunks normally expose TOC items in document order. Sorting
    # by target page makes the hierarchy deterministic if a producer does not.
    raw_entries.sort(key=lambda item: (item[2], len(raw_entries)))
    stack: dict[int, str] = {}
    entries: list[TocEntry] = []
    for level, title, target_page in raw_entries:
        for old_level in [value for value in stack if value >= level]:
            del stack[old_level]
        stack[level] = title
        path = tuple(stack[value] for value in sorted(stack) if value <= level)
        entries.append(TocEntry(level, title, target_page, path))
    return entries


def _masked_structure(text: str) -> str:
    """Blank regions whose OCR text must not create section boundaries."""
    chars = list(text)
    for pattern in (_PICTURE_BLOCK_RE, _HTML_TABLE_RE):
        for match in pattern.finditer(text):
            chars[match.start() : match.end()] = [
                "\n" if char == "\n" else " " for char in match.group(0)
            ]
    return "".join(chars)


def _headings(text: str) -> list[Heading]:
    masked = _masked_structure(text)
    return [
        Heading(match.start(), match.end(), len(match.group(1)), _clean_title(match.group(2)))
        for match in _HEADING_RE.finditer(masked)
        if _clean_title(match.group(2))
    ]


def _match_toc(heading: Heading, page: int, toc: list[TocEntry]) -> TocEntry | None:
    heading_key = _title_key(heading.title)
    candidates: list[tuple[float, int, TocEntry]] = []
    for entry in toc:
        distance = abs(page - entry.page)
        if distance > 2:
            continue
        entry_key = _title_key(entry.title)
        similarity = 1.0 if heading_key == entry_key else SequenceMatcher(
            None, heading_key, entry_key
        ).ratio()
        if similarity >= 0.9:
            candidates.append((similarity, -distance, entry))
    return max(candidates, default=(0.0, 0, None), key=lambda item: (item[0], item[1]))[2]


def _internal_label(title: str) -> bool:
    normalized = _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", title).casefold()).strip(" :.-")
    if normalized in _INTERNAL_LABELS:
        return True
    return any(
        normalized.startswith(f"{label} ")
        for label in ("advantages of", "special features of")
    )


def _valid_section_heading(heading: Heading, toc_match: TocEntry | None) -> bool:
    if toc_match is not None:
        return True
    title = heading.title
    if heading.markdown_level > 4 or _internal_label(title):
        return False
    if len(title) < 3 or len(title) > 180:
        return False
    if not any(character.isalpha() for character in title):
        return False
    # Replacement characters are a strong sign that OCR promoted noise to a
    # heading. One replacement in a long otherwise valid title is tolerated.
    if title.count("�") >= 2:
        return False
    return True


def _quality_rank(value: str | None) -> int:
    return {None: 0, "pass": 0, "ok": 0, "warning": 1, "quarantine": 2, "failed": 2}.get(value, 1)


def _section_quality(page_states: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [state.get("rag_text_status") for state in page_states]
    table_qualities = [
        state.get("table_quality")
        for state in page_states
        if state.get("table_quality")
    ]
    return {
        "rag_text_status": max(statuses, key=_quality_rank, default="pass") or "pass",
        "table_quality": max(table_qualities, key=_quality_rank, default=None),
    }


def split_sections(
    pages: list[dict[str, Any]], *, document_id: str
) -> list[dict[str, Any]]:
    """Split pages into sections namespaced by a stable external document ID.

    PDF title metadata is display data and is intentionally excluded from IDs:
    it is frequently empty, duplicated, or corrupted. The caller-owned
    ``document_id`` is therefore required and constrained to index-safe chars.
    """
    document_id = validate_document_id(document_id)
    toc = _collect_toc(pages)
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    heading_stack: dict[int, str] = {}
    active_toc_page = 0

    def append_text(value: str, page_number: int, page_state: dict[str, Any]) -> None:
        nonlocal current
        if not value.strip():
            return
        if current is None:
            current = {
                "title": "Document introduction",
                "path": ["Document introduction"],
                "level": 1,
                "parts": [],
                "pages": [],
                "page_states": [],
                "subheadings": [],
                "boundary": {"source": "implicit", "confidence": 0.5},
            }
        current["parts"].append(value.strip())
        if page_number not in current["pages"]:
            current["pages"].append(page_number)
            current["page_states"].append(page_state)

    def finish_current() -> None:
        nonlocal current
        if current is None:
            return
        text = "\n\n".join(part for part in current.pop("parts") if part).strip()
        page_numbers = current.pop("pages")
        states = current.pop("page_states")
        if text:
            index = len(sections) + 1
            current.update(
                {
                    "document_id": document_id,
                    "section_id": f"{document_id}:s{index:04d}",
                    "page_start": min(page_numbers),
                    "page_end": max(page_numbers),
                    "text": text,
                    "quality": _section_quality(states),
                }
            )
            sections.append(current)
        current = None

    def start_section(heading: Heading, toc_match: TocEntry | None, page_number: int) -> None:
        nonlocal active_toc_page, current, heading_stack
        finish_current()
        if toc_match is not None:
            level = toc_match.level
            path = list(toc_match.path)
            heading_stack = {index: title for index, title in enumerate(path, 1)}
            active_toc_page = toc_match.page
            source, confidence = "pdf_toc+markdown", 1.0
        else:
            level = heading.markdown_level
            # A TOC parent whose target is on a previous page is safer than a
            # stale Markdown stack. Strictly using an earlier page avoids
            # prematurely switching on PDFs whose bookmark points one page
            # before the printed heading.
            parents = [entry for entry in toc if entry.page < page_number and entry.level < level]
            if parents:
                parent = max(parents, key=lambda entry: (entry.page, entry.level))
                if parent.page > active_toc_page:
                    heading_stack = {
                        index: title for index, title in enumerate(parent.path, 1)
                    }
                    active_toc_page = parent.page
            for old_level in [value for value in heading_stack if value >= level]:
                del heading_stack[old_level]
            heading_stack[level] = heading.title
            path = [heading_stack[value] for value in sorted(heading_stack) if value <= level]
            source, confidence = "markdown_heading", 0.8
        current = {
            "title": heading.title,
            "path": path or [heading.title],
            "level": level,
            "parts": [],
            "pages": [],
            "page_states": [],
            "subheadings": [],
            "boundary": {"source": source, "confidence": confidence, "page": page_number},
        }

    def start_plain_toc_section(entry: TocEntry, page_number: int) -> None:
        nonlocal active_toc_page, current, heading_stack
        finish_current()
        heading_stack = {index: title for index, title in enumerate(entry.path, 1)}
        active_toc_page = entry.page
        current = {
            "title": entry.title,
            "path": list(entry.path),
            "level": entry.level,
            "parts": [],
            "pages": [],
            "page_states": [],
            "subheadings": [],
            "boundary": {"source": "pdf_toc+plain_text", "confidence": 0.95, "page": page_number},
        }

    for fallback_page, page in enumerate(pages, 1):
        page_number = _page_number(page, fallback_page)
        text = str(page.get("text", ""))
        page_state = page.get("post_processing", {}) or {}
        if page_state.get("rag_text_status") in {"quarantine", "failed"}:
            finish_current()
            continue
        headings = _headings(text)

        # Some chapter titles are emitted as ordinary top-of-page text rather
        # than Markdown headings. Only promote level-1 TOC titles here; lower
        # levels are too likely to occur naturally in technical prose.
        top_lines = {
            _title_key(line)
            for line in text[:600].splitlines()
            if line.strip() and not line.lstrip().startswith(("!", "<"))
        }
        markdown_keys = {_title_key(heading.title) for heading in headings}
        plain_toc = next(
            (
                entry
                for entry in toc
                if entry.level == 1
                and abs(entry.page - page_number) <= 1
                and _title_key(entry.title) in top_lines
                and _title_key(entry.title) not in markdown_keys
            ),
            None,
        )
        if plain_toc is not None:
            start_plain_toc_section(plain_toc, page_number)

        cursor = 0
        for heading in headings:
            append_text(text[cursor : heading.start], page_number, page_state)
            toc_match = _match_toc(heading, page_number, toc)
            heading_text = text[heading.start : heading.end]
            current_path_keys = {
                _title_key(title) for title in (current or {}).get("path", [])
            }
            late_toc_header = (
                _title_key(heading.title) in current_path_keys
                and any(
                    _title_key(entry.title) == _title_key(heading.title)
                    and entry.page < page_number - 2
                    for entry in toc
                )
            )
            if _valid_section_heading(heading, toc_match) and not late_toc_header:
                start_section(heading, toc_match, page_number)
            elif (
                current is not None
                and not late_toc_header
                and heading.title not in current["subheadings"]
            ):
                current["subheadings"].append(heading.title)
            append_text(heading_text, page_number, page_state)
            cursor = heading.end
        append_text(text[cursor:], page_number, page_state)

    finish_current()
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description="Split cleaned PDF pages into semantic sections")
    parser.add_argument("clean_pages", type=Path, help="post_processor clean_pages.json")
    parser.add_argument("output", type=Path, help="output sections.json")
    parser.add_argument("--document-id", required=True)
    args = parser.parse_args()

    pages = json.loads(args.clean_pages.read_text(encoding="utf8"))
    sections = split_sections(pages, document_id=args.document_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"sections": len(sections), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
