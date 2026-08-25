"""Independent table extraction and quality gating for automotive PDFs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from .post_processor import detect_geometry_watermarks

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass
class TableCandidate:
    parser: str
    table_id: str
    bbox: tuple[float, float, float, float]
    rows: list[list[str]]
    score: float
    status: str
    consistency: float
    empty_ratio: float
    short_ratio: float
    coverage: float
    watermark_hit_rate: float
    reason: str | None = None
    parser_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageTableReport:
    page: int
    candidate_count: int
    selected_table_id: str | None
    selected_parser: str | None
    selected_status: str | None
    candidates: list[TableCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TableDocumentReport:
    page_count: int
    pages_with_tables: int
    candidate_count: int
    accepted_tables: int
    warning_tables: int
    failed_tables: int
    camelot_available: bool
    camelot_candidates: int
    camelot_selected: int
    warnings: list[str]
    pages: list[PageTableReport]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(value) if len(token) > 1}


def _clean_rows(rows: Any) -> list[list[str]]:
    result: list[list[str]] = []
    for row in rows or []:
        result.append(["" if value is None else str(value).strip() for value in row])
    return result


def _quality(
    rows: list[list[str]],
    source_words: list[tuple[Any, ...]],
    watermark_keys: set[str],
) -> tuple[float, str, float, float, float, float, float, str | None]:
    if len(rows) < 2:
        return 0.0, "failed", 0.0, 1.0, 1.0, 0.0, 0.0, "fewer_than_two_rows"
    counts = [len(row) for row in rows]
    max_columns = max(counts, default=0)
    if max_columns == 0:
        return 0.0, "failed", 0.0, 1.0, 1.0, 0.0, 0.0, "no_columns"
    total_cells = len(rows) * max_columns
    empty = sum(
        1
        for row in rows
        for index in range(max_columns)
        if index >= len(row) or not row[index].strip()
    )
    empty_ratio = empty / total_cells
    common_columns = max(set(counts), key=counts.count)
    consistency = sum(count == common_columns for count in counts) / len(counts)
    cells = [cell for row in rows for cell in row if cell.strip()]
    short_ratio = sum(len(cell) <= 2 for cell in cells) / max(1, len(cells))
    extracted_tokens = _tokens(" ".join(cells))
    source_tokens = _tokens(" ".join(str(word[4]) for word in source_words))
    coverage = len(source_tokens & extracted_tokens) / max(1, len(source_tokens))
    watermark_hits = 0
    for word in source_words:
        key = "".join(_TOKEN_RE.findall(str(word[4]))).lower()
        if key and any(key == watermark or key in watermark for watermark in watermark_keys):
            watermark_hits += 1
    watermark_hit_rate = watermark_hits / max(1, len(source_words))
    score = (
        consistency * 0.35
        + (1 - empty_ratio) * 0.25
        + (1 - short_ratio) * 0.15
        + coverage * 0.25
        - watermark_hit_rate * 0.20
    )
    score = round(max(0.0, min(1.0, score)), 3)
    reasons: list[str] = []
    if max_columns < 2:
        reasons.append("single_column")
    if empty_ratio > 0.45:
        reasons.append("empty_cells")
    if consistency < 0.75:
        reasons.append("inconsistent_columns")
    if short_ratio > 0.45:
        reasons.append("fragmented_cells")
    if coverage < 0.55:
        reasons.append("low_source_coverage")
    if watermark_hit_rate > 0.10:
        reasons.append("watermark_contamination")
    reason = ", ".join(reasons) or None
    if score >= 0.80 and not reason:
        status = "good"
    elif score >= 0.60:
        status = "warning"
    else:
        status = "failed"
    return (
        score,
        status,
        consistency,
        empty_ratio,
        short_ratio,
        coverage,
        watermark_hit_rate,
        reason,
    )


def _grid_candidate(
    page: pymupdf.Page,
    bbox: tuple[float, float, float, float],
    rows: list[list[str]],
    parser: str,
) -> bool:
    """Reject prose/image regions before expensive quality scoring.

    A text table needs repeated x-column starts across at least three lines.
    A lines candidate additionally needs actual horizontal/vertical drawing
    evidence; this prevents full-page prose from becoming a table.
    """
    if len(rows) < 3 or max((len(row) for row in rows), default=0) < 2:
        return False
    words = page.get_text("words", clip=bbox)
    if not words:
        return False
    line_groups: dict[int, list[tuple[float, float, str]]] = {}
    for word in words:
        line_groups.setdefault(round(float(word[1]) / 4), []).append(
            (float(word[0]), float(word[2]), str(word[4]))
        )
    row_segments: list[list[tuple[float, float]]] = []
    for line in line_groups.values():
        line.sort()
        segments: list[tuple[float, float]] = []
        current_start = line[0][0]
        current_end = line[0][1]
        for x0, x1, _text in line[1:]:
            if x0 - current_end > 18:
                segments.append((current_start, current_end))
                current_start = x0
            current_end = x1
        segments.append((current_start, current_end))
        if len(segments) >= 2:
            row_segments.append(segments)
    repeated_x: dict[int, int] = {}
    for segments in row_segments:
        for x0, _x1 in segments:
            repeated_x[round(x0 / 12)] = repeated_x.get(round(x0 / 12), 0) + 1
    repeated_columns = sum(count >= 3 for count in repeated_x.values())
    if len(row_segments) < 3 or repeated_columns < 2:
        return False
    if parser == "pymupdf_lines":
        drawings = page.get_drawings()
        horizontal = vertical = 0
        x0, y0, x1, y1 = bbox
        for drawing in drawings:
            for item in drawing.get("items", []):
                if item[0] != "l":
                    continue
                points = item[1]
                try:
                    (ax, ay), (bx, by) = points
                except (TypeError, ValueError):
                    continue
                if not (x0 <= ax <= x1 or x0 <= bx <= x1):
                    continue
                if abs(ay - by) < 2 and y0 <= ay <= y1:
                    horizontal += 1
                if abs(ax - bx) < 2 and x0 <= ax <= x1:
                    vertical += 1
        if horizontal < 2 and vertical < 2:
            return False
    return True


def _extract_candidates(
    page: pymupdf.Page, page_number: int, watermark_keys: set[str]
) -> list[TableCandidate]:
    candidates: list[TableCandidate] = []
    strategies = [("pymupdf_lines", "lines"), ("pymupdf_text", "text")]
    for parser_name, strategy in strategies:
        try:
            result = page.find_tables(strategy=strategy)
        except Exception:
            continue
        for index, table in enumerate(result.tables, 1):
            bbox = tuple(float(value) for value in table.bbox)
            source_words = page.get_text("words", clip=table.bbox)
            rows = _clean_rows(table.extract())
            score, status, consistency, empty, short, coverage, watermark_rate, reason = _quality(
                rows, source_words, watermark_keys
            )
            # find_tables(strategy="text") can mistake ordinary multi-column
            # prose or rotated watermarks for a table. Keep only candidates
            # with a bounded grid and enough structural evidence.
            if len(rows) < 2 or len(rows) > 30:
                continue
            if max((len(row) for row in rows), default=0) > 12:
                continue
            if not _grid_candidate(page, bbox, rows, parser_name):
                continue
            if score < 0.60:
                continue
            candidates.append(
                TableCandidate(
                    parser=parser_name,
                    table_id=f"p{page_number}_{parser_name}_{index}",
                    bbox=bbox,
                    rows=rows,
                    score=score,
                    parser_report=None,
                    status=status,
                    consistency=round(consistency, 3),
                    empty_ratio=round(empty, 3),
                    short_ratio=round(short, 3),
                    coverage=round(coverage, 3),
                    watermark_hit_rate=round(watermark_rate, 3),
                    reason=reason,
                )
            )
    return candidates


def _extract_camelot_candidates(
    pdf_path: str | Path,
    page: pymupdf.Page,
    page_number: int,
    watermark_keys: set[str],
) -> list[TableCandidate]:
    try:
        import camelot
    except ImportError:
        return []
    candidates: list[TableCandidate] = []
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=str(page_number), flavor=flavor)
        except Exception:
            continue
        for index, table in enumerate(tables, 1):
            rows = _clean_rows(table.df.values.tolist())
            if len(rows) < 2 or len(rows) > 30:
                continue
            if max((len(row) for row in rows), default=0) > 12:
                continue
            bbox = tuple(float(value) for value in table._bbox)
            if not _grid_candidate(page, bbox, rows, f"camelot_{flavor}"):
                continue
            score, status, consistency, empty, short, coverage, watermark, reason = _quality(
                rows, page.get_text("words"), watermark_keys
            )
            parser_report = dict(table.parsing_report)
            # Camelot's own accuracy is useful evidence, but our quality gate
            # remains the final authority because watermarks can score well.
            external_accuracy = float(parser_report.get("accuracy", 0.0)) / 100
            score = round(score * 0.7 + external_accuracy * 0.3, 3)
            if score >= 0.80 and not reason:
                status = "good"
            elif score >= 0.60:
                status = "warning"
            else:
                status = "failed"
            candidates.append(
                TableCandidate(
                    parser=f"camelot_{flavor}",
                    table_id=f"p{page_number}_camelot_{flavor}_{index}",
                    bbox=bbox,
                    rows=rows,
                    score=score,
                    parser_report=parser_report,
                    status=status,
                    consistency=round(consistency, 3),
                    empty_ratio=round(empty, 3),
                    short_ratio=round(short, 3),
                    coverage=round(coverage, 3),
                    watermark_hit_rate=round(watermark, 3),
                    reason=reason,
                )
            )
    return candidates


def extract_tables(pdf_path: str | Path) -> tuple[list[dict[str, Any]], TableDocumentReport]:
    """Extract competing table candidates and quarantine low-quality results."""
    watermark_map = detect_geometry_watermarks(pdf_path)
    document = pymupdf.open(str(pdf_path))
    pages: list[PageTableReport] = []
    extracted: list[dict[str, Any]] = []
    camelot_candidate_count = 0
    camelot_selected_count = 0
    page_count = len(document)
    try:
        for page_number, page in enumerate(document, 1):
            candidates = _extract_candidates(
                page, page_number, watermark_map.get(page_number, set())
            )
            if not candidates or candidates[0].status != "good":
                camelot_candidates = _extract_camelot_candidates(
                    pdf_path, page, page_number, watermark_map.get(page_number, set())
                )
                camelot_candidate_count += len(camelot_candidates)
                candidates.extend(camelot_candidates)
            if not candidates:
                continue
            candidates.sort(key=lambda candidate: candidate.score, reverse=True)
            selected = candidates[0]
            if selected.parser.startswith("camelot_"):
                camelot_selected_count += 1
            warnings: list[str] = []
            if selected.status != "good":
                warnings.append("table_quarantined")
            pages.append(
                PageTableReport(
                    page=page_number,
                    candidate_count=len(candidates),
                    selected_table_id=selected.table_id,
                    selected_parser=selected.parser,
                    selected_status=selected.status,
                    candidates=candidates,
                    warnings=warnings,
                )
            )
            extracted.append(
                {
                    "type": "table" if selected.status != "failed" else "quarantine_table",
                    "page": page_number,
                    "bbox": selected.bbox,
                    "parser": selected.parser,
                    "quality_score": selected.score,
                    "quality": selected.status,
                    "rows": selected.rows,
                    "reason": selected.reason,
                    "parser_report": selected.parser_report,
                }
            )
    finally:
        document.close()
    try:
        import camelot  # noqa: F401

        camelot_available = True
    except ImportError:
        camelot_available = False
    warnings = []
    if not camelot_available and any(page.selected_status == "failed" for page in pages):
        warnings.append("camelot_unavailable_for_failed_tables")
    report = TableDocumentReport(
        page_count=page_count,
        pages_with_tables=len(pages),
        candidate_count=sum(page.candidate_count for page in pages),
        accepted_tables=sum(page.selected_status == "good" for page in pages),
        warning_tables=sum(page.selected_status == "warning" for page in pages),
        failed_tables=sum(page.selected_status == "failed" for page in pages),
        camelot_available=camelot_available,
        camelot_candidates=camelot_candidate_count,
        camelot_selected=camelot_selected_count,
        warnings=warnings,
        pages=pages,
    )
    return extracted, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and quality-gate PDF tables")
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    tables, report = extract_tables(args.pdf_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf8"
    )
    (args.output_dir / "table_quality_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
