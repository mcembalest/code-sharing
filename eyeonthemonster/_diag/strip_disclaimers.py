"""Locate the disclaimer BLOCK (not just disclaimer-flavored phrases) within each
page and split the text into (content, disclaimer) parts.

Rules:
- The block opens with one of a small set of stable opener phrases.
- Everything from the opener to end-of-page is the disclaimer.
- If the opener appears in the first ~10% of page text => page is "skip entirely".
- If it appears later => keep prefix as content; drop the suffix.
- If no opener found => keep whole page as content.

Writes pages_clean.jsonl and prints stats + spot-checks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "_diag" / "pages_enriched.jsonl"
OUT = ROOT / "_diag" / "pages_clean.jsonl"

# Opener phrases for the legal disclaimer block, lowercase. These should be
# distinctive enough that they essentially never appear in body prose.
OPENER_PATTERNS = [
    # Modern publication era ("IMPORTANT INFORMATION" as a heading)
    r"important information\s*\n",
    r"^\s*important information\s*$",
    # Early email era
    r"the above summary/prices/quotes/statistics",
    r"the above summary/prices/quotes",
    r"obtained from sources deemed to be\s+reliable, but we do not guarantee",
    # Common transitional opener
    r"this material is for information(?:al)? purposes only",
    r"the views,? opinions,? estimates? and strategies expressed",
]
OPENER_RE = re.compile("|".join(OPENER_PATTERNS), re.IGNORECASE | re.MULTILINE)


def find_disclaimer_start(text: str) -> int | None:
    """Return the char index where the disclaimer block begins, or None."""
    if not text:
        return None
    m = OPENER_RE.search(text)
    return m.start() if m else None


def main() -> None:
    pages = [json.loads(l) for l in IN.open()]
    print(f"Loaded {len(pages)} pages")

    stats = {"no_opener": 0, "footer_strip": 0, "skip_entire_page": 0}
    cleaned_words_total = 0
    original_words_total = 0
    skip_pages = []
    sample_strips = []

    for p in pages:
        text = p.get("text", "")
        original_words_total += len(text.split())
        start = find_disclaimer_start(text)
        if start is None:
            p["content_text"] = text
            p["disclaimer_text"] = ""
            p["disclaimer_class"] = "none"
            stats["no_opener"] += 1
        else:
            content = text[:start].strip()
            disclaimer = text[start:].strip()
            content_chars = len(content)
            page_chars = len(text)
            ratio = content_chars / max(1, page_chars)
            if ratio < 0.10:
                p["content_text"] = ""
                p["disclaimer_text"] = text
                p["disclaimer_class"] = "skip"
                stats["skip_entire_page"] += 1
                skip_pages.append(p["page"])
            else:
                p["content_text"] = content
                p["disclaimer_text"] = disclaimer
                p["disclaimer_class"] = "strip_footer"
                stats["footer_strip"] += 1
                if len(sample_strips) < 6:
                    sample_strips.append({
                        "page": p["page"],
                        "issue_date": p["issue_date"],
                        "content_preview": content[-200:].replace("\n", " "),
                        "disclaimer_preview": disclaimer[:160].replace("\n", " "),
                    })
        cleaned_words_total += len(p["content_text"].split())
        # Replace prior boolean with the new classification but keep old field too
        p["is_disclaimer_strict"] = p["disclaimer_class"] == "skip"

    with OUT.open("w") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}")

    print(f"\nClassification:")
    for k, v in stats.items():
        print(f"  {k:>20}: {v}")

    pct_words_kept = 100 * cleaned_words_total / max(1, original_words_total)
    print(f"\nWords kept after stripping: {cleaned_words_total:,} / "
          f"{original_words_total:,}  ({pct_words_kept:.1f}%)")

    # Compare to old heuristic
    old_disc = sum(1 for p in pages if p.get("is_disclaimer"))
    new_disc = stats["skip_entire_page"]
    print(f"\nOld heuristic flagged pages (any disclaimer phrases): {old_disc}")
    print(f"New strict skip-page count:                              {new_disc}")
    print(f"Pages with footer-strip only (keep most of page):       {stats['footer_strip']}")
    print(f"Pages with no disclaimer at all (keep whole page):      {stats['no_opener']}")

    # Indexable pages after new classification
    indexable = sum(1 for p in pages if p["disclaimer_class"] != "skip")
    chart_indexable = sum(
        1 for p in pages
        if p["disclaimer_class"] != "skip" and p["is_chart_bearing"]
    )
    text_indexable = sum(
        1 for p in pages
        if p["disclaimer_class"] != "skip" and not p["is_chart_bearing"]
    )
    print(f"\nIndexable pages now: {indexable}")
    print(f"  chart-bearing (vision targets): {chart_indexable}")
    print(f"  text-only:                      {text_indexable}")

    print("\n--- Skip-page positions: spot check (first 10) ---")
    for pg in skip_pages[:10]:
        p = next(x for x in pages if x["page"] == pg)
        print(f"  p.{pg:>5}  issue={p['issue_date']}  pos={p['position_in_issue']}  "
              f"chars={p['text_chars']}")

    print("\n--- Footer-strip spot check (last 200 chars of kept content + start of stripped block) ---")
    for s in sample_strips:
        print(f"  p.{s['page']:>5} [{s['issue_date']}]")
        print(f"    KEPT TAIL:   ...{s['content_preview']!r}")
        print(f"    STRIPPED:    {s['disclaimer_preview']!r}")


if __name__ == "__main__":
    main()
