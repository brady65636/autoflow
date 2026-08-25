from autoflow_scheduling.knowledge.post_processor import check_tables, process_chunks


def test_repeated_picture_watermark_is_removed_but_caption_is_kept() -> None:
    watermark = "Protected by copyright. Copyright by Volkswagen AG."
    raw = [
        {"metadata": {"page_number": page}, "text": (
            "Cooling system\n"
            "<!-- Start of picture text -->\n"
            "Fig. 1 Cooling system overview<br>"
            f"{watermark}\n"
            "a\n"
            "<!-- End of picture text -->"
        )}
        for page in range(1, 4)
    ]

    clean, report = process_chunks(raw)

    assert all("Cooling system" in chunk["text"] for chunk in clean)
    assert all("Fig. 1 Cooling system overview" in chunk["text"] for chunk in clean)
    assert all("Protected by copyright" not in chunk["text"] for chunk in clean)
    assert report.watermark_removed >= 6
    assert report.image_internal_text_removed == 0


def test_parser_fallback_is_observable_and_failure_is_quarantined() -> None:
    raw = [
        {
            "metadata": {"page_number": 1},
            "text": "Recovered direct text.",
            "parsing": {
                "status": "complete",
                "strategy": "pymupdf_plain_text_fallback",
                "fallback_reason": "markdown_replacement_ratio",
                "selected_replacement_ratio": 0.0,
                "ocr_used": False,
            },
        },
        {
            "metadata": {"page_number": 2},
            "text": "",
            "parsing": {
                "status": "failed",
                "strategy": "tesseract_ocr_fallback",
                "fallback_reason": "ocr_returned_no_text",
                "selected_replacement_ratio": 0.0,
                "ocr_used": True,
                "ocr_language": "eng",
            },
        },
    ]

    clean, report = process_chunks(raw)

    assert clean[0]["post_processing"]["parser_strategy"] == (
        "pymupdf_plain_text_fallback"
    )
    assert clean[0]["post_processing"]["rag_text_status"] == "warning"
    assert clean[1]["post_processing"]["rag_text_status"] == "quarantine"
    assert report.parser_fallback_pages == 2
    assert report.ocr_pages == 1
    assert report.parser_failed_pages == 1


def test_good_table_is_kept_and_marked_good() -> None:
    text = """| Engine | Power | Torque |
|---|---|---|
| 1.0 TSI | 70 kW | 160 Nm |
"""

    tables = check_tables(text, page=18)

    assert len(tables) == 1
    assert tables[0].quality == "good"
    assert tables[0].columns == 3


def test_fragmented_table_is_warning() -> None:
    text = """| A | B | C |
|---|---|---|
| a | b | c |
| 1 | 2 | 3 |
"""

    tables = check_tables(text, page=32)

    assert len(tables) == 1
    assert tables[0].quality == "low"
    assert tables[0].reason == "fragmented_cells"
