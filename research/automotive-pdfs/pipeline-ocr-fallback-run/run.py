from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "research" / "automotive-pdfs"
OUTPUT = Path(__file__).resolve().parent


def run(row):
    from autoflow_scheduling.knowledge.ingestion_pipeline import ingest_pdf
    from autoflow_scheduling.knowledge.retrieval_profile import DocumentRetrievalProfile

    pdf = CORPUS / Path(row["relative_path"].replace(chr(92), "/"))
    profile = DocumentRetrievalProfile.from_dict(
        {
            "document_id": row["pdf_id"],
            "content_type": "technical_training",
            "metadata_confidence": 0.95,
        }
    )
    started = time.perf_counter()
    try:
        report = ingest_pdf(pdf, OUTPUT / row["pdf_id"], profile, force=True)
        quality = json.loads(
            (OUTPUT / row["pdf_id"] / "quality_report.json").read_text(encoding="utf8")
        )
        return {
            "document_id": row["pdf_id"],
            "status": "ok",
            "wall_seconds": round(time.perf_counter() - started, 4),
            "report": report,
            "quality": {
                key: quality[key]
                for key in (
                    "page_count",
                    "rag_pass_pages",
                    "rag_warning_pages",
                    "quarantine_pages",
                    "parser_fallback_pages",
                    "ocr_pages",
                    "parser_failed_pages",
                    "warnings",
                )
            },
        }
    except Exception as error:
        return {
            "document_id": row["pdf_id"],
            "status": "error",
            "wall_seconds": round(time.perf_counter() - started, 4),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-4000:],
        }


def main():
    manifest = json.loads((CORPUS / "pdf_manifest.json").read_text(encoding="utf8"))
    selected = [row for row in manifest if row["pdf_id"] in {"PDF-008", "PDF-009"}]
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, row) for row in selected]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    results.sort(key=lambda item: item["document_id"])
    (OUTPUT / "validation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8"
    )


if __name__ == "__main__":
    main()
