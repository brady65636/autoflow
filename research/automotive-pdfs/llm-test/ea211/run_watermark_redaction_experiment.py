from __future__ import annotations

import math
from pathlib import Path

import pymupdf


SOURCE = Path("../../framework/01_SSP_511_EA211_petrol_engine.pdf")
OUTPUT = Path("watermark_redacted_EA211.pdf")


def main() -> None:
    doc = pymupdf.open(SOURCE)
    total = 0
    page_counts: list[int] = []
    for page in doc:
        data = page.get_text("dict")
        rotated = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", []))
                if not text.strip():
                    continue
                dx, dy = line.get("dir", (1.0, 0.0))
                angle = math.degrees(math.atan2(dy, dx))
                if abs(angle) > 8:
                    rotated.append(line["bbox"])
        if len(rotated) >= 20:
            for bbox in rotated:
                page.add_redact_annot(bbox, fill=(1, 1, 1))
            page.apply_redactions(images=0, graphics=0, text=0)
            total += len(rotated)
            page_counts.append(len(rotated))
        else:
            page_counts.append(0)
    doc.save(OUTPUT, garbage=4, deflate=True)
    doc.close()
    print({"output": str(OUTPUT), "pages": len(page_counts), "redacted": total})
    print("per_page_first_10", page_counts[:10])


if __name__ == "__main__":
    main()
