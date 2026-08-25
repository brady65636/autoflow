import csv, hashlib, os, sys
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parent
manifest = ROOT / 'pdf_manifest.csv'
rows = list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
errors = []
seen_sha = set()
for row in rows:
    path = ROOT / row['relative_path']
    if not path.is_file():
        errors.append(f"missing: {row['pdf_id']} {path}")
        continue
    data = path.read_bytes()
    if not data.startswith(b'%PDF-'):
        errors.append(f"not-pdf: {row['pdf_id']} {path}")
    sha = hashlib.sha256(data).hexdigest()
    if sha != row['sha256']:
        errors.append(f"sha256: {row['pdf_id']}")
    if sha in seen_sha:
        errors.append(f"duplicate: {row['pdf_id']}")
    seen_sha.add(sha)
    try:
        doc = fitz.open(path)
        if len(doc) != int(row['page_count']):
            errors.append(f"page-count: {row['pdf_id']}")
        if len(doc) == 0:
            errors.append(f"empty: {row['pdf_id']}")
        doc.close()
    except Exception as exc:
        errors.append(f"unreadable: {row['pdf_id']} {exc}")

expected = {'pdf_id','filename','relative_path','source','page_count','sha256','validation'}
if set(rows[0]) < expected:
    errors.append(f"manifest fields missing: {expected - set(rows[0])}")
if len(rows) < 100:
    errors.append(f"manifest rows below 100: {len(rows)}")
actual_pdf_count = sum(1 for p in ROOT.rglob('*.pdf') if p.is_file())
if actual_pdf_count < 100:
    errors.append(f"actual pdf count below 100: {actual_pdf_count}")
print(f"manifest_rows={len(rows)}")
print(f"actual_pdf_count={actual_pdf_count}")
print(f"unique_sha256={len(seen_sha)}")
print(f"errors={len(errors)}")
for error in errors:
    print(error)
raise SystemExit(1 if errors else 0)
