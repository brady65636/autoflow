"""Quality control for PyMuPDF4LLM page chunks.

The parser output is intentionally treated as immutable input.  This module
returns cleaned copies and a quality report; it never overwrites raw chunks.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

_PICTURE_BLOCK_RE = re.compile(
    r"<!--\s*Start of picture text\s*-->(.*?)<!--\s*End of picture text\s*-->",
    re.IGNORECASE | re.DOTALL,
)
_TABLE_RE = re.compile(r"(?ms)(?:^|\n)(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n?)+)")
_HTML_TABLE_RE = re.compile(r"(?is)<table\b.*?</table>")
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class TableQuality:
    table_id: str
    quality: str
    rows: int
    columns: int
    empty_cell_ratio: float
    fragmented_cell_ratio: float
    reason: str | None = None


@dataclass
class PageQualityReport:
    page: int
    text_chars_before: int
    text_chars_after: int
    parser_status: str | None = None
    parser_strategy: str | None = None
    parser_fallback_reason: str | None = None
    replacement_character_ratio: float = 0.0
    ocr_used: bool = False
    ocr_language: str | None = None
    ocr_error: str | None = None
    rag_text_status: str = "pass"
    geometry_watermark_detected: bool = False
    watermark_in_text: bool = False
    image_internal_text_detected: bool = False
    table_detected: bool = False
    table_quality: str | None = None
    watermark_removed: int = 0
    geometry_watermark_removed: int = 0
    image_internal_text_removed: int = 0
    table_count: int = 0
    low_quality_tables: int = 0
    tables: list[TableQuality] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality: str = "ok"


@dataclass
class DocumentQualityReport:
    page_count: int
    normal_pages: int
    warning_pages: int
    failed_pages: int
    rag_pass_pages: int
    rag_warning_pages: int
    quarantine_pages: int
    geometry_watermark_pages: int
    watermark_in_text_pages: int
    watermark_removed: int
    geometry_watermark_removed: int
    image_internal_text_removed: int
    table_count: int
    low_quality_tables: int
    parser_fallback_pages: int
    ocr_pages: int
    parser_failed_pages: int
    warnings: list[str]
    pages: list[PageQualityReport]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _plain(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", _TAG_RE.sub("", value)).strip()


def _line_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _split_picture_lines(block: str) -> list[str]:
    block = block.replace("<br>", "\n").replace("<br/>", "\n")
    return [_plain(line) for line in block.splitlines() if _plain(line)]


def _has_watermark_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "protected by copyright",
            "copyright by volkswagen",
            "volkswagenagdoesnot",
            "does not guarantee or accept any liability",
        )
    )


def _is_obvious_watermark(line: str) -> bool:
    lowered = line.lower()
    return any(
        marker in lowered
        for marker in (
            "protected by copyright",
            "copyright by volkswagen",
            "volkswagen ag does not",
            "volkswagenagdoesnot",
            "copyright",
        )
    )


def _collect_repeated_picture_fragments(chunks: Iterable[dict[str, Any]]) -> set[str]:
    counts: Counter[str] = Counter()
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        for block in _PICTURE_BLOCK_RE.findall(text):
            for line in _split_picture_lines(block):
                key = _line_key(line)
                if key and len(line) <= 24:
                    counts[key] += 1
    # A repeated short fragment across pages is much more likely to be a
    # rotated watermark than a meaningful caption or diagram label.
    return {key for key, count in counts.items() if count >= 3}


def detect_geometry_watermarks(pdf_path: str | Path) -> dict[int, set[str]]:
    """Find high-confidence rotated watermark text per page.

    This is the geometry-based algorithm from ``exp.py``: a page is considered
    to have a watermark candidate only when it contains at least 20 rotated
    text lines.  Returning normalized text keys lets us apply the decision to
    PyMuPDF4LLM's raw page chunk without rewriting the PDF parser.
    """
    import math

    import pymupdf

    result: dict[int, set[str]] = {}
    document = pymupdf.open(str(pdf_path))
    try:
        for page_number, page in enumerate(document, 1):
            rotated: list[str] = []
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    text = "".join(span.get("text", "") for span in line.get("spans", []))
                    text = _plain(text)
                    if not text:
                        continue
                    dx, dy = line.get("dir", (1.0, 0.0))
                    angle = math.degrees(math.atan2(dy, dx))
                    if abs(angle) > 8:
                        key = _line_key(text)
                        if key:
                            rotated.append(key)
            if len(rotated) >= 20:
                result[page_number] = set(rotated)
    finally:
        document.close()
    return result


def _clean_picture_blocks(
    text: str, repeated_fragments: set[str], geometry_fragments: set[str]
) -> tuple[str, int, int, int]:
    watermark_removed = 0
    geometry_removed = 0
    image_noise_removed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal watermark_removed, geometry_removed, image_noise_removed
        lines = _split_picture_lines(match.group(1))
        kept: list[str] = []
        for line in lines:
            key = _line_key(line)
            if key in geometry_fragments:
                watermark_removed += 1
                geometry_removed += 1
            elif _is_obvious_watermark(line):
                watermark_removed += 1
            elif key in repeated_fragments:
                watermark_removed += 1
            elif len(line) <= 2 and not re.search(r"\d", line):
                image_noise_removed += 1
            else:
                kept.append(line)
        if not kept:
            return ""
        return (
            "<!-- Start of picture text -->\n"
            + "\n".join(kept)
            + "\n<!-- End of picture text -->"
        )

    cleaned = _PICTURE_BLOCK_RE.sub(replace, text)
    return cleaned, watermark_removed, geometry_removed, image_noise_removed


def _table_cells(table: str, html: bool) -> list[list[str]]:
    if html:
        rows = re.findall(r"(?is)<tr\b.*?</tr>", table)
        return [
            [_plain(cell) for cell in re.findall(r"(?is)<t[dh]\b.*?</t[dh]>", row)]
            for row in rows
        ]
    rows = []
    for line in table.strip().splitlines():
        if line.lstrip().startswith("|---") or line.lstrip().startswith("| ---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def check_tables(text: str, page: int) -> list[TableQuality]:
    matches = [(match.group(1), False) for match in _TABLE_RE.finditer(text)]
    matches.extend((match.group(0), True) for match in _HTML_TABLE_RE.finditer(text))
    reports: list[TableQuality] = []
    for index, (table, is_html) in enumerate(matches, 1):
        rows = _table_cells(table, is_html)
        nonempty = [cell for row in rows for cell in row if cell]
        columns = max((len(row) for row in rows), default=0)
        empty_ratio = 1 - (len(nonempty) / max(1, sum(len(row) for row in rows)))
        fragmented = sum(1 for cell in nonempty if len(cell) <= 3)
        fragment_ratio = fragmented / max(1, len(nonempty))
        consistent = len({len(row) for row in rows if row}) <= 1
        reasons: list[str] = []
        if columns < 2:
            reasons.append("single_column")
        if empty_ratio > 0.45:
            reasons.append("empty_cells")
        if fragment_ratio > 0.45:
            reasons.append("fragmented_cells")
        if not consistent:
            reasons.append("inconsistent_columns")
        quality = "low" if reasons else "good"
        reports.append(
            TableQuality(
                table_id=f"p{page}_t{index}",
                quality=quality,
                rows=len(rows),
                columns=columns,
                empty_cell_ratio=round(empty_ratio, 3),
                fragmented_cell_ratio=round(fragment_ratio, 3),
                reason=", ".join(reasons) or None,
            )
        )
    return reports


def process_chunks(
    raw_chunks: list[dict[str, Any]],
    pdf_path: str | Path | None = None,
    table_report: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], DocumentQualityReport]:
    """Return cleaned copies and page/document quality metrics."""
    repeated_fragments = _collect_repeated_picture_fragments(raw_chunks)
    geometry_watermarks = detect_geometry_watermarks(pdf_path) if pdf_path else {}
    clean_chunks: list[dict[str, Any]] = []
    page_reports: list[PageQualityReport] = []

    for index, raw_chunk in enumerate(raw_chunks, 1):
        cleaned_chunk = copy.deepcopy(raw_chunk)
        raw_text = str(raw_chunk.get("text", ""))
        cleaned_text, watermark_count, geometry_count, image_noise_count = _clean_picture_blocks(
            raw_text,
            repeated_fragments,
            geometry_watermarks.get(index, set()),
        )
        tables = check_tables(cleaned_text, index)
        table_page = next(
            (
                page
                for page in (table_report or {}).get("pages", [])
                if page.get("page") == index
            ),
            None,
        )
        table_detected = table_page is not None
        table_quality = table_page.get("selected_status") if table_page else None
        parsing = raw_chunk.get("parsing", {}) or {}
        parser_status = parsing.get("status")
        parser_strategy = parsing.get("strategy")
        parser_fallback_reason = parsing.get("fallback_reason")
        replacement_ratio = float(parsing.get("selected_replacement_ratio", 0.0))
        ocr_used = bool(parsing.get("ocr_used"))
        geometry_watermark_detected = index in geometry_watermarks
        watermark_in_text = _has_watermark_text(cleaned_text)
        image_internal_text_detected = image_noise_count > 0
        warnings: list[str] = []
        if parsing and parser_strategy != "pymupdf4llm_markdown":
            warnings.append("parser_fallback")
        if ocr_used:
            warnings.append("ocr_used")
        if parser_status == "failed":
            warnings.append("parser_failed")
        if parser_status == "failed" and replacement_ratio > 0:
            warnings.append("corrupted_text")
        if watermark_in_text:
            warnings.append("watermark_in_text")
        if table_quality == "warning":
            warnings.append("low_quality_table")
        if table_quality == "failed":
            warnings.append("failed_table")
        if (
            parser_status == "failed"
            or watermark_in_text
            or table_quality == "failed"
        ):
            rag_text_status = "quarantine"
        elif warnings:
            rag_text_status = "warning"
        else:
            rag_text_status = "pass"
        report = PageQualityReport(
            page=index,
            text_chars_before=len(raw_text),
            text_chars_after=len(cleaned_text),
            parser_status=parser_status,
            parser_strategy=parser_strategy,
            parser_fallback_reason=parser_fallback_reason,
            replacement_character_ratio=replacement_ratio,
            ocr_used=ocr_used,
            ocr_language=parsing.get("ocr_language"),
            ocr_error=parsing.get("ocr_error"),
            rag_text_status=rag_text_status,
            geometry_watermark_detected=geometry_watermark_detected,
            watermark_in_text=watermark_in_text,
            image_internal_text_detected=image_internal_text_detected,
            table_detected=table_detected,
            table_quality=table_quality,
            watermark_removed=watermark_count,
            geometry_watermark_removed=geometry_count,
            image_internal_text_removed=image_noise_count,
            table_count=len(tables),
            low_quality_tables=sum(table.quality == "low" for table in tables),
            tables=tables,
            warnings=warnings,
            quality="warning" if warnings else "ok",
        )
        cleaned_chunk["text"] = cleaned_text
        cleaned_chunk["post_processing"] = {
            "parser_status": parser_status,
            "parser_strategy": parser_strategy,
            "parser_fallback_reason": parser_fallback_reason,
            "replacement_character_ratio": replacement_ratio,
            "ocr_used": ocr_used,
            "ocr_language": parsing.get("ocr_language"),
            "ocr_error": parsing.get("ocr_error"),
            "rag_text_status": rag_text_status,
            "geometry_watermark_detected": geometry_watermark_detected,
            "watermark_in_text": watermark_in_text,
            "image_internal_text_detected": image_internal_text_detected,
            "table_quality": table_quality,
            "watermark_removed": watermark_count,
            "image_internal_text_removed": image_noise_count,
            "quality": report.quality,
        }
        clean_chunks.append(cleaned_chunk)
        page_reports.append(report)

    report = DocumentQualityReport(
        page_count=len(page_reports),
        normal_pages=sum(page.rag_text_status == "pass" for page in page_reports),
        warning_pages=sum(page.rag_text_status == "warning" for page in page_reports),
        failed_pages=sum(page.rag_text_status == "quarantine" for page in page_reports),
        rag_pass_pages=sum(page.rag_text_status == "pass" for page in page_reports),
        rag_warning_pages=sum(page.rag_text_status == "warning" for page in page_reports),
        quarantine_pages=sum(page.rag_text_status == "quarantine" for page in page_reports),
        geometry_watermark_pages=sum(
            page.geometry_watermark_detected for page in page_reports
        ),
        watermark_in_text_pages=sum(page.watermark_in_text for page in page_reports),
        watermark_removed=sum(page.watermark_removed for page in page_reports),
        geometry_watermark_removed=sum(
            page.geometry_watermark_removed for page in page_reports
        ),
        image_internal_text_removed=sum(page.image_internal_text_removed for page in page_reports),
        table_count=sum(page.table_count for page in page_reports),
        low_quality_tables=sum(page.low_quality_tables for page in page_reports),
        parser_fallback_pages=sum(
            page.parser_strategy not in {None, "pymupdf4llm_markdown"}
            for page in page_reports
        ),
        ocr_pages=sum(page.ocr_used for page in page_reports),
        parser_failed_pages=sum(page.parser_status == "failed" for page in page_reports),
        warnings=sorted({warning for page in page_reports for warning in page.warnings}),
        pages=page_reports,
    )
    return clean_chunks, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process PyMuPDF4LLM page chunks")
    parser.add_argument("raw_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--pdf-path",
        type=Path,
        help="Original PDF used for bbox/rotation watermark detection",
    )
    parser.add_argument(
        "--table-report",
        type=Path,
        help="Optional table_quality_report.json used by the final RAG gate",
    )
    args = parser.parse_args()
    raw = json.loads(args.raw_json.read_text(encoding="utf8"))
    pdf_path = args.pdf_path
    if pdf_path is None and raw and raw[0].get("metadata", {}).get("file_path"):
        pdf_path = Path(raw[0]["metadata"]["file_path"])
    table_report = None
    if args.table_report:
        table_report = json.loads(args.table_report.read_text(encoding="utf8"))
    clean, report = process_chunks(
        raw,
        pdf_path=pdf_path,
        table_report=table_report,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "clean_pages.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf8"
    )
    (args.output_dir / "quality_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf8"
    )
    markdown = "\n\n".join(chunk.get("text", "") for chunk in clean)
    (args.output_dir / "clean_document.md").write_text(markdown, encoding="utf8")
    print(json.dumps({
        "pages": report.page_count,
        "normal_pages": report.normal_pages,
        "warning_pages": report.warning_pages,
        "rag_pass_pages": report.rag_pass_pages,
        "rag_warning_pages": report.rag_warning_pages,
        "quarantine_pages": report.quarantine_pages,
        "geometry_watermark_pages": report.geometry_watermark_pages,
        "watermark_in_text_pages": report.watermark_in_text_pages,
        "watermark_removed": report.watermark_removed,
        "geometry_watermark_removed": report.geometry_watermark_removed,
        "image_internal_text_removed": report.image_internal_text_removed,
        "table_count": report.table_count,
        "low_quality_tables": report.low_quality_tables,
        "parser_fallback_pages": report.parser_fallback_pages,
        "ocr_pages": report.ocr_pages,
        "parser_failed_pages": report.parser_failed_pages,
        "warnings": report.warnings,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
