from autoflow_scheduling.knowledge.chunking import chunk_pages, chunk_sections
from autoflow_scheduling.knowledge.retrieval_profile import (
    DocumentContentType,
    DocumentRetrievalProfile,
)


def test_warning_table_keeps_raw_text_fallback() -> None:
    pages = [{"metadata": {"title": "doc"}, "text": "Engine\nPower\nTorque\n1.4 TSI 103 kW 250 Nm"}]
    report = {"pages": [{"page": 1, "selected_status": "warning"}]}

    chunks = chunk_pages(pages, report)

    assert len(chunks) == 1
    assert chunks[0]["table_fallback"] == "raw_text_preserved"
    assert "250 Nm" in chunks[0]["text"]


def test_section_chunks_never_cross_section_boundary() -> None:
    sections = [
        {
            "section_id": "s1",
            "title": "Cooling system",
            "path": ["Engine mechanics", "Cooling system"],
            "text": "Cooling paragraph. " * 30,
        },
        {
            "section_id": "s2",
            "title": "Fuel system",
            "path": ["Engine mechanics", "Fuel system"],
            "text": "Fuel paragraph. " * 30,
        },
    ]

    chunks = chunk_sections(sections, chunk_size=180, chunk_overlap=20)

    assert {chunk["section_id"] for chunk in chunks} == {"s1", "s2"}
    assert all(not ("Cooling" in chunk["text"] and "Fuel" in chunk["text"]) for chunk in chunks)
    assert all(chunk["section_path"] for chunk in chunks)


def test_chunk_keeps_namespaced_ids_and_section_document_association() -> None:
    section = {
        "document_id": "PDF-006",
        "section_id": "PDF-006:s0001",
        "title": "Cooling system",
        "path": ["Engine mechanics", "Cooling system"],
        "text": "## Cooling system\n\nThe coolant pump circulates coolant.",
    }

    profile = DocumentRetrievalProfile(
        document_id="PDF-006",
        content_type=DocumentContentType.OTHER,
        metadata_confidence=1.0,
    )
    chunks = chunk_sections(
        [section], chunk_size=500, chunk_overlap=50, document_profile=profile
    )

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "PDF-006:s0001:c001"
    assert chunks[0]["document_id"] == "PDF-006"
    assert chunks[0]["section_id"] == "PDF-006:s0001"
    assert chunks[0]["chunk_id"].startswith(f'{chunks[0]["section_id"]}:')


def test_subheading_is_preferred_and_kept_at_chunk_start() -> None:
    section = {
        "section_id": "sensor",
        "title": "Fuel pressure sender G247",
        "path": ["Sensors", "Fuel pressure sender G247"],
        "subheadings": ["Signal use", "Effects upon failure"],
        "text": (
            "### Fuel pressure sender G247\n\n"
            + "A" * 900
            + "\n\n##### Signal use\n\n"
            + "B" * 900
            + "\n\n##### Effects upon failure\n\n"
            + "C" * 900
        ),
    }

    chunks = chunk_sections([section], chunk_size=1200, chunk_overlap=100)

    assert len(chunks) == 3
    assert chunks[1]["text"].startswith("##### Signal use")
    assert chunks[1]["subheading"] == "Signal use"
    assert chunks[2]["text"].startswith("##### Effects upon failure")


def test_continuation_chunk_inherits_active_subheading() -> None:
    section = {
        "section_id": "sensor",
        "title": "Sensor",
        "path": ["Sensors", "Sensor"],
        "subheadings": ["Signal use"],
        "text": "### Sensor\n\nIntro.\n\n##### Signal use\n\n" + "word " * 300,
    }

    chunks = chunk_sections([section], chunk_size=500, chunk_overlap=50)

    signal_chunks = [chunk for chunk in chunks if chunk["subheading"] == "Signal use"]
    assert len(signal_chunks) > 1
    assert signal_chunks[0]["text"].startswith("word")
    assert all(chunk["subheadings"] == [] for chunk in signal_chunks)


def test_small_section_is_not_forced_apart_at_subheading() -> None:
    section = {
        "section_id": "sensor",
        "title": "Sensor",
        "path": ["Sensors", "Sensor"],
        "subheadings": ["Signal use"],
        "text": "### Sensor\n\nDescription.\n\n##### Signal use\n\nUsed by the ECU.",
    }

    chunks = chunk_sections([section], chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0]["subheadings"] == ["Signal use"]


def test_structural_shell_and_image_only_chunks_are_not_emitted() -> None:
    sections = [
        {
            "section_id": "sensors",
            "title": "Sensors",
            "path": ["Engine management system", "Sensors"],
            "text": "## Sensors",
        },
        {
            "section_id": "overview",
            "title": "Overview",
            "path": ["Introduction", "Overview"],
            "text": "6\n\n![](page-0007.png)",
        },
    ]

    chunks = chunk_sections(sections, chunk_size=500, chunk_overlap=50)

    assert chunks == []


def test_navigation_and_self_test_sections_are_not_indexed() -> None:
    sections = [
        {
            "section_id": "contents",
            "title": "Contents",
            "path": ["Contents"],
            "text": "# Contents\n\nOil circuit ........ 25",
        },
        {
            "section_id": "quiz",
            "title": "Test your knowledge",
            "path": ["Test your knowledge"],
            "text": "# Test your knowledge\n\nWhich oil pressure is correct?",
        },
    ]

    assert chunk_sections(sections) == []


def test_short_but_informative_chunk_is_kept() -> None:
    section = {
        "section_id": "camshaft",
        "title": "Camshaft adjustment",
        "path": ["Engine mechanics", "Camshaft adjustment"],
        "text": "### Camshaft adjustment\n\nMaximum adjustment is 40 degrees.",
    }

    chunks = chunk_sections([section], chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    assert "40 degrees" in chunks[0]["text"]
