from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "research" / "automotive-pdfs"
OUTPUT = Path(__file__).resolve().parent


def source_path(row):
    return CORPUS / Path(row["relative_path"].replace(chr(92), "/"))


def run(row):
    from autoflow_scheduling.knowledge.ingestion_pipeline import ingest_pdf, sha256_file
    from autoflow_scheduling.knowledge.retrieval_profile import DocumentRetrievalProfile

    pdf = source_path(row)
    content_type = "official_news" if row["pdf_id"] == "PDF-001" else "technical_training"
    profile = DocumentRetrievalProfile.from_dict(
        {
            "document_id": row["pdf_id"],
            "content_type": content_type,
            "metadata_confidence": 0.95,
        }
    )
    started = time.perf_counter()
    actual_hash = sha256_file(pdf)
    base = {
        "pdf_id": row["pdf_id"],
        "filename": row["filename"],
        "source": str(pdf),
        "manifest_pages": int(row["page_count"]),
        "sha256_match": actual_hash == row["sha256"],
    }
    try:
        report = ingest_pdf(pdf, OUTPUT / row["pdf_id"], profile)
        quality = json.loads(
            (OUTPUT / row["pdf_id"] / "quality_report.json").read_text(encoding="utf8")
        )
        return {
            **base,
            "status": "ok",
            "wall_seconds": round(time.perf_counter() - started, 4),
            "report": report,
            "quality": {
                key: quality.get(key)
                for key in (
                    "page_count",
                    "rag_pass_pages",
                    "rag_warning_pages",
                    "quarantine_pages",
                    "geometry_watermark_pages",
                    "watermark_in_text_pages",
                    "watermark_removed",
                    "table_count",
                    "low_quality_tables",
                    "warnings",
                )
            },
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "wall_seconds": round(time.perf_counter() - started, 4),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "traceback": traceback.format_exc()[-3000:],
        }


def main():
    manifest = json.loads((CORPUS / "pdf_manifest.json").read_text(encoding="utf8"))[:10]
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run, row): row["pdf_id"] for row in manifest}
        for future in as_completed(futures):
            item = future.result()
            results.append(item)
            print(
                json.dumps(
                    {
                        "finished": item["pdf_id"],
                        "status": item["status"],
                        "seconds": item["wall_seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    results.sort(key=lambda item: item["pdf_id"])
    report = {
        "run": "initial_or_resume",
        "parallel_workers": 3,
        "chunk_size": 1800,
        "chunk_overlap": 200,
        "wall_seconds": round(time.perf_counter() - started, 4),
        "results": results,
        "counts": {
            "total": len(results),
            "ok": sum(item["status"] == "ok" for item in results),
            "error": sum(item["status"] == "error" for item in results),
        },
    }
    (OUTPUT / "batch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(report["counts"]), flush=True)


if __name__ == "__main__":
    main()
