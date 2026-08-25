import pytest

from autoflow_scheduling.knowledge.sectioning import split_sections


def _page(
    number: int,
    text: str,
    toc_items: list[list[object]] | None = None,
    title: str = "Test manual",
) -> dict:
    return {
        "metadata": {"title": title, "page_number": number},
        "toc_items": toc_items or [],
        "text": text,
        "post_processing": {"rag_text_status": "pass", "table_quality": None},
    }


def test_section_continues_across_page_boundary() -> None:
    pages = [
        _page(1, "## Cooling system\n\nFirst page explanation."),
        _page(2, "Continuation without another heading."),
    ]

    sections = split_sections(pages, document_id="PDF-006")

    assert len(sections) == 1
    assert sections[0]["title"] == "Cooling system"
    assert sections[0]["page_start"] == 1
    assert sections[0]["page_end"] == 2
    assert "Continuation" in sections[0]["text"]


def test_internal_label_does_not_create_a_section() -> None:
    pages = [
        _page(
            1,
            "### Fuel pressure sender G247\n\nDescription.\n\n"
            "##### Signal use\n\nThe ECU uses this signal.\n\n"
            "##### Effects upon signal failure\n\nEmergency operation.",
        )
    ]

    sections = split_sections(pages, document_id="PDF-006")

    assert len(sections) == 1
    assert sections[0]["title"] == "Fuel pressure sender G247"
    assert sections[0]["subheadings"] == ["Signal use", "Effects upon signal failure"]
    assert "Emergency operation" in sections[0]["text"]


def test_toc_match_restores_parent_path() -> None:
    pages = [
        _page(
            12,
            "## Poly V-belt drive\n\nBelt description.",
            [
                [1, "Engine mechanics", 12],
                [2, "Poly V-belt drive", 12],
            ],
        )
    ]

    sections = split_sections(pages, document_id="PDF-006")

    assert sections[0]["path"] == ["Engine mechanics", "Poly V-belt drive"]
    assert sections[0]["boundary"]["source"] == "pdf_toc+markdown"
    assert sections[0]["boundary"]["confidence"] == 1.0


def test_same_title_can_be_a_real_section_under_another_chapter() -> None:
    pages = [
        _page(
            10,
            "## Fuel system\n\nMechanical fuel system.",
            [[1, "Engine mechanics", 1], [2, "Fuel system", 10]],
        ),
        _page(
            20,
            "# Engine management system\n\n## Fuel system\n\nControl strategy.",
            [[1, "Engine management system", 20]],
        ),
    ]

    sections = split_sections(pages, document_id="PDF-006")

    fuel_sections = [section for section in sections if section["title"] == "Fuel system"]
    assert len(fuel_sections) == 2
    assert fuel_sections[1]["path"] == ["Engine management system", "Fuel system"]


def test_empty_metadata_titles_are_namespaced_without_collision() -> None:
    pages = [_page(1, "## Cooling system\n\nDescription.", title="")]

    first = split_sections(pages, document_id="PDF-001")
    second = split_sections(pages, document_id="PDF-002")

    assert first[0]["section_id"] == "PDF-001:s0001"
    assert second[0]["section_id"] == "PDF-002:s0001"
    assert first[0]["section_id"] != second[0]["section_id"]
    assert first[0]["document_id"] == "PDF-001"
    assert second[0]["document_id"] == "PDF-002"


def test_empty_document_id_fails() -> None:
    pages = [_page(1, "## Cooling system\n\nDescription.")]

    with pytest.raises(ValueError, match="document_id"):
        split_sections(pages, document_id="")


def test_quarantined_parser_page_is_not_sectioned() -> None:
    page = _page(1, "## Corrupted\n\nText that must not be indexed.")
    page["post_processing"]["rag_text_status"] = "quarantine"

    assert split_sections([page], document_id="PDF-006") == []


def test_picture_ocr_heading_does_not_create_boundary() -> None:
    pages = [
        _page(
            1,
            "## Oil circuit\n\nBefore image.\n\n"
            "<!-- Start of picture text -->\n"
            "### Not a real heading\nDiagram label\n"
            "<!-- End of picture text -->\n\nAfter image.",
        )
    ]

    sections = split_sections(pages, document_id="PDF-006")

    assert len(sections) == 1
    assert sections[0]["title"] == "Oil circuit"
    assert "Not a real heading" in sections[0]["text"]
