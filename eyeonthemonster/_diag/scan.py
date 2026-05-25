"""Diagnostic pass over the full PDF.

For each page record:
  - text + word/char counts
  - number of raster images embedded
  - number of vector draw operations (charts in JPM publications are usually vector,
    so this is the real chart-density signal — not image count)
  - a detected date from the first ~400 chars of text, if present
  - the issue-internal page number printed in the page (best-effort)

Writes _diag/pages.jsonl and prints summary stats.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "Eye on the Monster.pdf"
OUT = ROOT / "_diag" / "pages.jsonl"

MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|"
    r"Sept?\.?|Oct\.?|Nov\.?|Dec\.?|"
    r"JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|"
    r"NOVEMBER|DECEMBER)"
)
DATE_RE = re.compile(rf"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}\b")


def detect_date(text: str) -> str | None:
    head = text[:600]
    m = DATE_RE.search(head)
    return m.group(0) if m else None


def main() -> None:
    doc = fitz.open(PDF_PATH)
    n = doc.page_count
    t0 = time.time()
    text_lens, drawing_counts, image_counts = [], [], []
    dated_pages = 0
    out = OUT.open("w", encoding="utf-8")
    for i in range(n):
        page = doc.load_page(i)
        text = page.get_text("text")
        n_img = len(page.get_images())
        # get_drawings() can be slow; cap is implicit via PyMuPDF
        try:
            n_draw = len(page.get_drawings())
        except Exception:
            n_draw = -1
        date = detect_date(text)
        rec = {
            "page": i + 1,
            "text_chars": len(text),
            "text_words": len(text.split()),
            "n_images": n_img,
            "n_drawings": n_draw,
            "date": date,
            "text": text,
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        text_lens.append(len(text))
        drawing_counts.append(n_draw if n_draw >= 0 else 0)
        image_counts.append(n_img)
        if date:
            dated_pages += 1
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{n} ({(i+1)/(time.time()-t0):.0f} pages/s)")
    out.close()
    elapsed = time.time() - t0
    print(f"\nDone: {n} pages in {elapsed:.1f}s ({n/elapsed:.0f} pages/s)")
    print(f"Wrote {OUT}")

    # Summary stats
    def bucket(vals, edges):
        c = Counter()
        for v in vals:
            label = f"<={edges[0]}"
            for e in edges:
                if v <= e:
                    label = f"<={e}"
                    break
            else:
                label = f">{edges[-1]}"
            if v > edges[-1]:
                label = f">{edges[-1]}"
            c[label] += 1
        return c

    print(f"\nDated pages: {dated_pages}/{n} ({100*dated_pages/n:.1f}%)")

    print("\nText-word buckets (words/page):")
    for k, v in sorted(bucket(
        [w for w in (len(open(OUT).readline()) for _ in [0])],
        [0]
    ).items()):
        pass  # placeholder; we'll use a simpler print below

    # Simpler text/drawing histograms inline
    def hist(values, edges, label):
        b = [0] * (len(edges) + 1)
        for v in values:
            placed = False
            for i, e in enumerate(edges):
                if v <= e:
                    b[i] += 1
                    placed = True
                    break
            if not placed:
                b[-1] += 1
        print(f"\n{label}")
        prev = "-inf"
        for i, e in enumerate(edges):
            print(f"  ({prev}, {e}]: {b[i]}")
            prev = str(e)
        print(f"  ({prev}, +inf): {b[-1]}")

    word_counts = [r["text_words"] for r in (json.loads(l) for l in open(OUT))]
    hist(word_counts, [50, 150, 300, 500, 800], "Words per page:")
    hist(drawing_counts, [0, 50, 200, 1000, 5000], "Drawing ops per page (chart-density proxy):")
    hist(image_counts, [0, 1, 3, 10], "Raster images per page:")

    # Issue-boundary estimate: count distinct dates that appear, in document order,
    # by scanning page-by-page and counting changes.
    last = None
    issues = 0
    for rec_line in open(OUT):
        rec = json.loads(rec_line)
        d = rec["date"]
        if d and d != last:
            issues += 1
            last = d
    print(f"\nApprox issue count via date-change heuristic: {issues}")


if __name__ == "__main__":
    main()
