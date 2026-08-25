from pathlib import Path

import pytest

from autoflow_scheduling.knowledge import pdf_parser
from autoflow_scheduling.knowledge.pdf_parser import (
    parse_pdf_pages,
    replacement_character_ratio,
)


class Page:
    def __init__(self, direct, ocr="", ocr_error=None):
        self.direct = direct
        self.ocr = ocr
        self.ocr_error = ocr_error

    def get_text(self, _format, *, textpage=None, sort=False):
        return self.ocr if textpage is not None else self.direct

    def get_textpage_ocr(self, **_kwargs):
        if self.ocr_error:
            raise self.ocr_error
        return object()


class Document(list):
    def close(self):
        pass


def test_adaptive_parser_selects_markdown_plain_text_and_ocr(monkeypatch):
    markdown = [
        {"metadata": {"page_number": 1}, "text": "## Healthy\n\nNormal text."},
        {"metadata": {"page_number": 2}, "text": "�" * 90 + " few"},
        {"metadata": {"page_number": 3}, "text": ""},
        {"metadata": {"page_number": 4}, "text": ""},
    ]
    pages = Document(
        [
            Page("Healthy Normal text."),
            Page("Recovered direct text."),
            Page("", "Recognized image text."),
            Page("", ocr_error=RuntimeError("Tesseract unavailable")),
        ]
    )
    monkeypatch.setattr(pdf_parser.pymupdf4llm, "to_markdown", lambda *_a, **_k: markdown)
    monkeypatch.setattr(pdf_parser.pymupdf, "open", lambda *_a, **_k: pages)
    monkeypatch.setattr(pdf_parser, "_configure_tessdata", lambda _language: None)

    result = parse_pdf_pages("manual.pdf")

    assert result[0]["parsing"]["strategy"] == "pymupdf4llm_markdown"
    assert result[1]["text"] == "Recovered direct text."
    assert result[1]["parsing"]["strategy"] == "pymupdf_plain_text_fallback"
    assert result[2]["text"] == "Recognized image text."
    assert result[2]["parsing"]["strategy"] == "tesseract_ocr_fallback"
    assert result[2]["parsing"]["status"] == "complete"
    assert result[3]["parsing"]["status"] == "failed"
    assert "Tesseract unavailable" in result[3]["parsing"]["ocr_error"]


def test_tessdata_configuration_validates_requested_languages(
    monkeypatch, tmp_path: Path
):
    (tmp_path / "eng.traineddata").write_bytes(b"model")
    monkeypatch.setenv("AUTOFLOW_TESSDATA_PREFIX", str(tmp_path))

    assert pdf_parser._configure_tessdata("eng") == tmp_path
    with pytest.raises(RuntimeError, match="chi_sim"):
        pdf_parser._configure_tessdata("eng+chi_sim")


def test_replacement_ratio_and_parser_configuration_are_validated():
    assert replacement_character_ratio("abc��") == 0.4

    for kwargs in (
        {"replacement_ratio_threshold": 1.1},
        {"ocr_dpi": 50},
        {"ocr_language": " "},
    ):
        try:
            parse_pdf_pages("manual.pdf", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(kwargs)
