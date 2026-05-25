"""One-shot diagnostic: render N sample pages, dump text layer alongside.

Picks pages spread across the document, plus a few requested ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "Eye on the Monster.pdf"
OUT_DIR = ROOT / "_diag" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    doc = fitz.open(PDF_PATH)
    n = doc.page_count
    # Spread sample across the document plus the [p.3456] from plan.md
    sample_pages = sorted(set([
        1, 2, 50,
        int(n * 0.10), int(n * 0.25), int(n * 0.40),
        int(n * 0.55), int(n * 0.70), int(n * 0.85),
        3456,
        n - 1,
    ]))
    rows = []
    for p in sample_pages:
        page = doc.load_page(p - 1)
        text = page.get_text("text")
        # Render at 1.5x for clarity but keep file size moderate
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_path = OUT_DIR / f"page_{p:05d}.png"
        pix.save(img_path)
        rows.append({
            "page": p,
            "image": str(img_path.relative_to(ROOT)),
            "text_chars": len(text),
            "text_words": len(text.split()),
            "text_preview": text[:600],
        })
    out = OUT_DIR / "sample.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"total_pages={n}")
    print(f"wrote {len(rows)} samples to {OUT_DIR}")
    for r in rows:
        print(f"  p.{r['page']:>5}  chars={r['text_chars']:>5}  words={r['text_words']:>4}")


if __name__ == "__main__":
    main()
