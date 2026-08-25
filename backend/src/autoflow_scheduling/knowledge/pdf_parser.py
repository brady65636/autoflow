"""Adaptive PDF parsing with plain-text and OCR fallbacks."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm

_DEFAULT_REPLACEMENT_RATIO = 0.10
_DEFAULT_OCR_DPI = 200
_DEFAULT_OCR_LANGUAGE = "eng+chi_sim"


def replacement_character_ratio(text: str) -> float:
    visible = [character for character in text if not character.isspace()]
    return text.count("�") / max(len(visible), 1)


def _usable(text: str) -> bool:
    return any(character.isalnum() for character in text)


def _configure_tessdata(language: str) -> Path:
    candidates = [
        os.getenv("AUTOFLOW_TESSDATA_PREFIX"),
        os.getenv("TESSDATA_PREFIX"),
        str(Path(os.getenv("LOCALAPPDATA", "")) / "autoflow-ocr/share/tessdata"),
        str(Path(os.getenv("ProgramFiles", "")) / "Tesseract-OCR/tessdata"),
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/4.00/tessdata",
    ]
    tessdata = next(
        (Path(value) for value in candidates if value and Path(value).is_dir()),
        None,
    )
    if tessdata is None:
        raise RuntimeError(
            "Tesseract tessdata is unavailable; set AUTOFLOW_TESSDATA_PREFIX"
        )
    missing = [
        name for name in language.split("+") if not (tessdata / f"{name}.traineddata").is_file()
    ]
    if missing:
        raise RuntimeError(f"missing Tesseract language data: {', '.join(missing)}")
    os.environ["TESSDATA_PREFIX"] = str(tessdata)
    return tessdata


def _ocr_page(page: Any, *, dpi: int, language: str) -> str:
    _configure_tessdata(language)
    textpage = page.get_textpage_ocr(dpi=dpi, language=language, full=True)
    return str(page.get_text("text", textpage=textpage, sort=True))


def parse_pdf_pages(
    pdf_path: str | Path,
    *,
    replacement_ratio_threshold: float = _DEFAULT_REPLACEMENT_RATIO,
    ocr_dpi: int = _DEFAULT_OCR_DPI,
    ocr_language: str = _DEFAULT_OCR_LANGUAGE,
) -> list[dict[str, Any]]:
    """Parse each page, selecting the least destructive usable representation.

    Markdown remains the primary representation. Corrupted Markdown falls back
    to MuPDF plain text; image-only pages use OCR. Every page records why a
    fallback was selected so downstream quality gates cannot silently pass it.
    """
    if not 0 <= replacement_ratio_threshold <= 1:
        raise ValueError("replacement_ratio_threshold must be between 0 and 1")
    if ocr_dpi < 72:
        raise ValueError("ocr_dpi must be at least 72")
    if not ocr_language.strip():
        raise ValueError("ocr_language must not be empty")

    path = Path(pdf_path)
    markdown_pages = pymupdf4llm.to_markdown(
        str(path), page_chunks=True, write_images=False, use_ocr=False
    )
    if not isinstance(markdown_pages, list):
        raise RuntimeError("pymupdf4llm page_chunks output must be a list")

    document = pymupdf.open(str(path))
    try:
        if len(markdown_pages) != len(document):
            raise RuntimeError(
                f"parser returned {len(markdown_pages)} pages for a {len(document)} page PDF"
            )
        selected_pages: list[dict[str, Any]] = []
        for page_index, (markdown_page, page) in enumerate(
            zip(markdown_pages, document, strict=True), 1
        ):
            selected = copy.deepcopy(markdown_page)
            markdown_text = str(markdown_page.get("text", ""))
            markdown_ratio = replacement_character_ratio(markdown_text)
            # ``sort=True`` duplicates overlapping glyph layers in some legacy
            # Type1 PDFs (PDF-009 grew from 31k to 50k chars), so preserve the
            # native content-stream order for this fallback.
            direct_text = str(page.get_text("text"))
            direct_ratio = replacement_character_ratio(direct_text)
            strategy = "pymupdf4llm_markdown"
            reason = None
            status = "complete"
            ocr_used = False
            ocr_error = None
            selected_text = markdown_text

            markdown_corrupted = markdown_ratio > replacement_ratio_threshold
            if markdown_corrupted and _usable(direct_text) and direct_ratio < markdown_ratio:
                selected_text = direct_text
                strategy = "pymupdf_plain_text_fallback"
                reason = "markdown_replacement_ratio"
            elif not _usable(markdown_text) and _usable(direct_text):
                selected_text = direct_text
                strategy = "pymupdf_plain_text_fallback"
                reason = "markdown_empty"
            elif not _usable(markdown_text) and not _usable(direct_text):
                strategy = "tesseract_ocr_fallback"
                reason = "no_extractable_text"
                ocr_used = True
                try:
                    selected_text = _ocr_page(page, dpi=ocr_dpi, language=ocr_language)
                    if not _usable(selected_text):
                        status = "failed"
                        reason = "ocr_returned_no_text"
                except Exception as error:
                    selected_text = ""
                    status = "failed"
                    ocr_error = f"{type(error).__name__}: {str(error)[:240]}"

            selected_ratio = replacement_character_ratio(selected_text)
            if selected_ratio > replacement_ratio_threshold:
                status = "failed"
                reason = "selected_text_replacement_ratio"
            selected["text"] = selected_text
            selected["parsing"] = {
                "status": status,
                "strategy": strategy,
                "fallback_reason": reason,
                "page": page_index,
                "markdown_chars": len(markdown_text),
                "selected_chars": len(selected_text),
                "markdown_replacement_ratio": round(markdown_ratio, 4),
                "selected_replacement_ratio": round(selected_ratio, 4),
                "ocr_used": ocr_used,
                "ocr_dpi": ocr_dpi if ocr_used else None,
                "ocr_language": ocr_language if ocr_used else None,
                "ocr_error": ocr_error,
            }
            selected_pages.append(selected)
        return selected_pages
    finally:
        document.close()
