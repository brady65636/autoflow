from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SHARDS = ROOT / "annotations"
SECTIONS = ROOT / "combined_sections.json"
CHUNKS = ROOT / "combined_chunks.json"
OUTPUT = ROOT / "evaluation_cases_80.json"
POSITIVE_OUTPUT = ROOT / "retrieval_positive_cases.json"

REQUIRED_POSITIVE = {
    "case_id",
    "case_kind",
    "language",
    "query",
    "question_type",
    "required_knowledge",
    "expected_document_id",
    "expected_section_id",
    "expected_title",
    "expected_path_contains",
    "evidence_page_start",
    "evidence_page_end",
    "evidence_quote",
    "annotation_status",
}
VALID_QUESTION_TYPES = {
    "principle",
    "diagnosis",
    "repair",
    "specification",
    "maintenance",
    "training",
    "competition",
    "general",
}
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
VALID_KINDS = {
    "positive",
    "no_answer",
    "cross_document_confusion",
    "authority_safety",
}


def load_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain an array of objects")
    return value


def validate() -> list[dict[str, Any]]:
    shard_paths = sorted(SHARDS.glob("cases_*.json"))
    if len(shard_paths) != 4:
        raise ValueError(f"expected 4 annotation shards, found {len(shard_paths)}")
    cases = [case for path in shard_paths for case in load_array(path)]
    sections = {item["section_id"]: item for item in load_array(SECTIONS)}
    indexed_section_ids = {item["section_id"] for item in load_array(CHUNKS)}
    errors: list[str] = []

    ids = [str(case.get("case_id", "")) for case in cases]
    if len(cases) != 80:
        errors.append(f"expected 80 cases, found {len(cases)}")
    if len(ids) != len(set(ids)):
        errors.append("duplicate case_id")
    queries = [str(case.get("query", "")).strip().casefold() for case in cases]
    if len(queries) != len(set(queries)):
        errors.append("duplicate query")

    kinds = Counter(str(case.get("case_kind", "")) for case in cases)
    languages = Counter(str(case.get("language", "")) for case in cases)
    positives_by_document: Counter[str] = Counter()
    negative_count = kinds["no_answer"] + kinds["cross_document_confusion"]
    if kinds["positive"] != 60:
        errors.append(f"expected 60 positives, found {kinds['positive']}")
    if negative_count != 10:
        errors.append(f"expected 10 no-answer/confusion cases, found {negative_count}")
    if kinds["authority_safety"] != 10:
        errors.append(f"expected 10 authority/safety cases, found {kinds['authority_safety']}")
    if languages != Counter({"en": 80}):
        errors.append(f"expected en 80 and no Chinese cases, found {dict(languages)}")

    for case in cases:
        case_id = str(case.get("case_id", "<missing>"))
        kind = str(case.get("case_kind", ""))
        if kind not in VALID_KINDS:
            errors.append(f"{case_id}: invalid case_kind {kind!r}")
        if case.get("annotation_status") != "verified":
            errors.append(f"{case_id}: annotation_status is not verified")
        query = str(case.get("query", "")).strip()
        if not query:
            errors.append(f"{case_id}: empty query")
        if case.get("language") == "zh" and not CJK_RE.search(query):
            errors.append(f"{case_id}: zh case has no Chinese query text")
        if case.get("language") == "en" and CJK_RE.search(query):
            errors.append(f"{case_id}: en case contains Chinese query text")
        if kind != "positive":
            if not str(case.get("expected_behavior", "")).strip():
                errors.append(f"{case_id}: missing expected_behavior")
            if not str(case.get("reason", "")).strip():
                errors.append(f"{case_id}: missing reason")
            continue

        missing = REQUIRED_POSITIVE - set(case)
        if missing:
            errors.append(f"{case_id}: missing fields {sorted(missing)}")
            continue
        if case["question_type"] not in VALID_QUESTION_TYPES:
            errors.append(f"{case_id}: invalid question_type")
        document_id = str(case["expected_document_id"])
        positives_by_document[document_id] += 1
        section_id = str(case["expected_section_id"])
        section = sections.get(section_id)
        if section is None:
            errors.append(f"{case_id}: unknown section {section_id}")
            continue
        if section_id not in indexed_section_ids:
            errors.append(f"{case_id}: section has no indexed chunk")
        if section.get("document_id") != document_id:
            errors.append(f"{case_id}: section belongs to another document")
        if section.get("title") != case["expected_title"]:
            errors.append(f"{case_id}: expected_title differs from section title")
        path = section.get("path") or []
        if not all(part in path for part in case.get("expected_path_contains") or []):
            errors.append(f"{case_id}: expected_path_contains does not match")
        start, end = case["evidence_page_start"], case["evidence_page_end"]
        if start < section.get("page_start", start) or end > section.get("page_end", end):
            errors.append(f"{case_id}: evidence pages outside section range")
        quote = str(case["evidence_quote"])
        if len(quote.strip()) < 40:
            errors.append(f"{case_id}: evidence_quote is too short")
        if not quote.strip() or quote not in str(section.get("text", "")):
            errors.append(f"{case_id}: evidence_quote not found verbatim in section")
        if "�" in quote:
            errors.append(f"{case_id}: evidence_quote contains replacement character")

    expected_documents = {f"PDF-{index:03d}" for index in range(1, 21)}
    if set(positives_by_document) != expected_documents:
        errors.append("positive document coverage is incomplete")
    for document_id in sorted(expected_documents):
        if positives_by_document[document_id] != 3:
            errors.append(
                f"{document_id}: expected 3 positives, found {positives_by_document[document_id]}"
            )
    if errors:
        raise ValueError("annotation validation failed:\n- " + "\n- ".join(errors))
    return sorted(cases, key=lambda item: str(item["case_id"]))


def main() -> None:
    cases = validate()
    OUTPUT.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf8")
    positives = [
        {
            "case_id": case["case_id"],
            "query": case["query"],
            "question_type": case["question_type"],
            "required_knowledge": case["required_knowledge"],
            "expected_document_id": case["expected_document_id"],
            "expected_section_id": case["expected_section_id"],
            "expected_title": case["expected_title"],
            "expected_path_contains": case["expected_path_contains"],
            "language": case["language"],
        }
        for case in cases
        if case["case_kind"] == "positive"
    ]
    POSITIVE_OUTPUT.write_text(
        json.dumps(positives, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps({"cases": len(cases), "positive_cases": len(positives)}))


if __name__ == "__main__":
    main()
