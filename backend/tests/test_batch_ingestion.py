"""Contract tests for the current batch ingestion coordinator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from autoflow_scheduling.knowledge import batch_ingestion


class _Future:
    def __init__(self, value: Any = None, error: Exception | None = None) -> None:
        self.value, self.error = value, error

    def result(self) -> Any:
        if self.error:
            raise self.error
        return self.value


class _FakeExecutor:
    instances: list[_FakeExecutor] = []

    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers
        self.futures: list[_Future] = []
        self.__class__.instances.append(self)

    def __enter__(self) -> _FakeExecutor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def submit(self, function: Any, *args: Any) -> _Future:
        try:
            future = _Future(function(*args))
        except Exception as error:  # emulate a process future failing remotely
            future = _Future(error=error)
        self.futures.append(future)
        return future


def _manifest(tmp_path: Path, count: int) -> tuple[Path, Path, list[dict[str, Any]]]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    entries = []
    for index in range(count):
        pdf_id = f"PDF-{index:03d}"
        relative = f"docs/{pdf_id}.pdf"
        path = corpus / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"placeholder {index}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(
            {
                "pdf_id": pdf_id,
                "relative_path": relative,
                "sha256": digest,
                "page_count": 1,
                "filename": path.name,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries), encoding="utf8")
    return manifest, corpus, entries


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returned_failures: set[str] | None = None,
    raised_failures: set[str] | None = None,
    skipped: set[str] | None = None,
) -> list[dict[str, Any]]:
    returned_failures = returned_failures or set()
    raised_failures = raised_failures or set()
    skipped = skipped or set()
    calls: list[dict[str, Any]] = []
    _FakeExecutor.instances.clear()

    def fake_worker(
        position: int,
        entry: dict[str, Any],
        corpus_root: str,
        output_dir: str,
        chunk_size: int,
        chunk_overlap: int,
        force: bool,
    ) -> dict[str, Any]:
        pdf_id = str(entry["pdf_id"])
        calls.append(
            {"pdf_id": pdf_id, "chunk_size": chunk_size, "overlap": chunk_overlap, "force": force}
        )
        if pdf_id in raised_failures:
            raise RuntimeError("future exploded")
        if pdf_id in returned_failures:
            return {"position": position, "pdf_id": pdf_id, "status": "failed", "error": "bad PDF"}
        document_dir = Path(output_dir) / pdf_id
        section_id = f"{pdf_id}:s0001"
        sections = [
            {"document_id": pdf_id, "section_id": section_id, "title": "Test", "text": pdf_id}
        ]
        chunks = [
            {
                "document_id": pdf_id,
                "section_id": section_id,
                "chunk_id": f"{section_id}:c001",
                "text": pdf_id,
            }
        ]
        for name, value in (
            ("sections.json", sections),
            ("chunks.json", chunks),
            ("quality_report.json", {"page_count": 1}),
        ):
            document_dir.mkdir(parents=True, exist_ok=True)
            (document_dir / name).write_text(json.dumps(value), encoding="utf8")
        actual = hashlib.sha256(
            (Path(corpus_root) / str(entry["relative_path"])).read_bytes()
        ).hexdigest()
        report = {
            "pipeline_fingerprint": "pipeline-test",
            "status": "skipped" if pdf_id in skipped else "complete",
        }
        return {
            "position": position,
            "pdf_id": pdf_id,
            "status": "success",
            "actual_sha256": actual,
            "manifest_sha256": entry["sha256"],
            "report": report,
            "quality": {"page_count": 1},
        }

    monkeypatch.setattr(batch_ingestion, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(batch_ingestion, "as_completed", lambda futures: reversed(list(futures)))
    monkeypatch.setattr(batch_ingestion, "_worker", fake_worker)
    return calls


def test_default_limit_and_stable_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, corpus, entries = _manifest(tmp_path, 25)
    _install_fakes(monkeypatch)

    report = batch_ingestion.run_batch(manifest, corpus, tmp_path / "out")

    assert report["selected"] == report["limit"] == 20
    assert [item["pdf_id"] for item in report["results"]] == [
        item["pdf_id"] for item in entries[:20]
    ]
    assert _FakeExecutor.instances[-1].max_workers == 20
    combined = json.loads((tmp_path / "out" / "combined_sections.json").read_text())
    assert [item["document_id"] for item in combined] == sorted(
        item["pdf_id"] for item in entries[:20]
    )


def test_partial_failure_does_not_publish_or_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, corpus, entries = _manifest(tmp_path, 3)
    _install_fakes(
        monkeypatch,
        returned_failures={entries[1]["pdf_id"]},
        raised_failures={entries[2]["pdf_id"]},
    )
    monkeypatch.setattr(
        batch_ingestion, "create_session_factory", lambda _: pytest.fail("unexpected DB")
    )

    report = batch_ingestion.run_batch(manifest, corpus, tmp_path / "out", database_url="sqlite://")

    assert report["succeeded"] == 1 and report["failed"] == 2
    assert not (tmp_path / "out" / "combined_sections.json").exists()
    assert not (tmp_path / "out" / "combined_chunks.json").exists()
    assert any(
        item["pdf_id"] == entries[2]["pdf_id"] and "future exploded" in item["error"]
        for item in report["results"]
    )


def test_success_merges_and_selection_contains_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, corpus, entries = _manifest(tmp_path, 2)
    _install_fakes(monkeypatch)

    report = batch_ingestion.run_batch(manifest, corpus, tmp_path / "out", offset=0, limit=2)

    assert report["validation"]["status"] == "pass"
    selection = json.loads((tmp_path / "out" / "selection.json").read_text())
    assert {"actual_sha256", "manifest_sha256", "pipeline_fingerprint"} <= selection[0].keys()
    assert selection[0]["actual_sha256"] == entries[0]["sha256"]
    assert selection[0]["pipeline_fingerprint"] == "pipeline-test"


@pytest.mark.parametrize(
    "sections,chunks,needle",
    [
        (
            [{"document_id": "PDF-001", "section_id": "PDF-001:s0001"}] * 2,
            [],
            "duplicate section_id",
        ),
        (
            [{"document_id": "PDF-001", "section_id": "PDF-001:s0001"}],
            [
                {
                    "document_id": "PDF-001",
                    "section_id": "PDF-001:s9999",
                    "chunk_id": "PDF-001:s9999:c001",
                }
            ],
            "unknown section reference",
        ),
        ([], [], "document has no sections"),
    ],
)
def test_validate_rejects_duplicate_dangling_and_missing_sections(
    sections: list[dict[str, Any]], chunks: list[dict[str, Any]], needle: str
) -> None:
    validation = batch_ingestion._validate(sections, chunks, ["PDF-001"])
    assert validation["status"] == "fail"
    assert any(needle in error for error in validation["errors"])


def test_sqlite_import_is_optional_and_skip_fields_are_transparent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, corpus, _ = _manifest(tmp_path, 1)
    calls = _install_fakes(monkeypatch, skipped={"PDF-000"})

    report = batch_ingestion.run_batch(
        manifest,
        corpus,
        tmp_path / "out",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'knowledge.db'}",
        force=True,
        chunk_size=99,
        chunk_overlap=7,
    )

    assert report["sqlite_import"]["documents"] == 1
    assert report["results"][0]["report"]["status"] == "skipped"
    assert calls == [{"pdf_id": "PDF-000", "chunk_size": 99, "overlap": 7, "force": True}]
    assert (tmp_path / "out" / "combined_chunks.json").exists()
