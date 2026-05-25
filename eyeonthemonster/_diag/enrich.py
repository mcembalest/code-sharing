"""Enrich pages.jsonl with structural labels:

- issue_date: date for this page's containing issue, propagated forward from the
  most recent prior page that had a header date (per user prior: PDF is in
  chronological order).
- position_in_issue: 1, 2, 3, ... within the contiguous block sharing one date.
- is_disclaimer: boolean from a phrase-density heuristic; disclaimer text grows
  over time but uses stable boilerplate phrases.
- is_chart_bearing: drawings > 50 (rough cut for whether vision adds anything).

Writes pages_enriched.jsonl and prints stats.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "_diag" / "pages.jsonl"
OUT = ROOT / "_diag" / "pages_enriched.jsonl"

DISCLAIMER_PHRASES = [
    "past performance is not a guarantee",
    "past performance is not",
    "for information purposes only",
    "for informational purposes only",
    "this material is for information",
    "obtained from sources deemed",
    "sources deemed to be reliable",
    "j.p. morgan securities",
    "jpmorgan chase",
    "jpmorgan",
    "member nyse",
    "member sipc",
    "nasd and sipc",
    "fdic",
    "investment risks",
    "subject to investment",
    "securities products",
    "no warranty",
    "the views, opinions",
    "constitutes michael cembalest's judgment",
    "michael cembalest's judgment",
    "should not be treated as such",
    "in no way constitutes",
    "estimates and strategies expressed",
    "tax advice",
    "redistribute",
    "intended recipient",
]


def disclaimer_signal(text: str) -> tuple[int, float]:
    """Return (distinct-phrase hits, density = hits per 1000 chars)."""
    if not text:
        return 0, 0.0
    low = text.lower()
    hits = sum(1 for p in DISCLAIMER_PHRASES if p in low)
    density = hits / max(1, len(low) / 1000)
    return hits, density


def main() -> None:
    pages = [json.loads(l) for l in IN.open()]
    print(f"Loaded {len(pages)} pages")

    # 1. Date propagation forward (chronological order assumption)
    current_date = None
    for p in pages:
        if p.get("date"):
            current_date = p["date"]
        p["issue_date"] = current_date

    # 2. Position within issue
    last_date = None
    pos = 0
    for p in pages:
        if p["issue_date"] != last_date:
            pos = 1
            last_date = p["issue_date"]
        else:
            pos += 1
        p["position_in_issue"] = pos

    # 3. Disclaimer detection
    for p in pages:
        hits, density = disclaimer_signal(p.get("text", ""))
        p["disclaimer_hits"] = hits
        p["disclaimer_density"] = round(density, 3)
        # A disclaimer page has multiple distinct boilerplate phrases.
        # 3+ hits is a strong signal; below that we require density too.
        p["is_disclaimer"] = hits >= 3 or (hits >= 2 and density >= 1.5)

    # 4. Chart-bearing: vector-heavy OR has raster images beyond a likely JPM logo
    for p in pages:
        nd = p.get("n_drawings") or 0
        ni = p.get("n_images") or 0
        p["is_chart_bearing"] = nd > 50 or ni >= 2

    # Write enriched file
    with OUT.open("w") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}")

    # --- Summary stats ---
    undated = sum(1 for p in pages if not p["issue_date"])
    disclaimer = sum(1 for p in pages if p["is_disclaimer"])
    chart_bearing = sum(1 for p in pages if p["is_chart_bearing"])
    chart_bearing_real = sum(1 for p in pages if p["is_chart_bearing"] and not p["is_disclaimer"])
    text_only_real = sum(1 for p in pages if not p["is_chart_bearing"] and not p["is_disclaimer"])

    print(f"\nPropagated date coverage: {len(pages) - undated}/{len(pages)} "
          f"({100*(len(pages)-undated)/len(pages):.1f}%)")
    print(f"Pages with NO inferred date (likely doc preamble): {undated}")

    print(f"\nDisclaimer pages flagged: {disclaimer}/{len(pages)} "
          f"({100*disclaimer/len(pages):.1f}%)")
    print(f"Chart-bearing pages (drawings>50): {chart_bearing}")
    print(f"  ... of which non-disclaimer (real vision targets): {chart_bearing_real}")
    print(f"Text-only non-disclaimer pages: {text_only_real}")
    print(f"Indexable content pages total: {chart_bearing_real + text_only_real}")

    # Issue-level rollup
    issue_pages = defaultdict(list)
    for p in pages:
        if p["issue_date"]:
            issue_pages[p["issue_date"]].append(p)
    n_issues = len(issue_pages)
    print(f"\nDistinct issues by propagated date: {n_issues}")

    # Pages per issue distribution
    lengths = [len(v) for v in issue_pages.values()]
    if lengths:
        lengths_sorted = sorted(lengths)
        def pct(p):
            return lengths_sorted[min(len(lengths_sorted)-1, int(len(lengths_sorted)*p))]
        print(f"Pages/issue  min={min(lengths)}  p25={pct(0.25)}  med={pct(0.5)}  "
              f"p75={pct(0.75)}  p95={pct(0.95)}  max={max(lengths)}")

    # Disclaimer pages per issue distribution
    disc_per_issue = [sum(1 for p in v if p["is_disclaimer"]) for v in issue_pages.values()]
    if disc_per_issue:
        print(f"Disclaimer pages/issue  min={min(disc_per_issue)}  "
              f"med={sorted(disc_per_issue)[len(disc_per_issue)//2]}  "
              f"max={max(disc_per_issue)}")
        no_disc = sum(1 for d in disc_per_issue if d == 0)
        print(f"  issues with 0 disclaimer pages: {no_disc}/{n_issues}")

    # Cross-check: are disclaimers actually at the END of issues?
    end_check = Counter()  # bucket "is disclaimer at position pos/total"
    for v in issue_pages.values():
        total = len(v)
        for p in v:
            if p["is_disclaimer"]:
                frac = p["position_in_issue"] / total
                bucket = "first half" if frac <= 0.5 else "second half"
                if p["position_in_issue"] == total:
                    bucket = "last page"
                end_check[bucket] += 1
    print(f"\nDisclaimer position validation (should mostly be 'last page' or 'second half'):")
    for k, v in end_check.most_common():
        print(f"  {k}: {v}")

    # Sample a few flagged disclaimer pages and a few not-flagged for spot-check
    print("\n--- 5 disclaimer-flagged samples (page, hits, density, first 120 chars) ---")
    flagged = [p for p in pages if p["is_disclaimer"]]
    for p in flagged[:5]:
        prev = p["text"][:120].replace("\n", " ")
        print(f"  p.{p['page']:>5}  hits={p['disclaimer_hits']}  d={p['disclaimer_density']:.2f}  {prev!r}")

    print("\n--- 5 NOT-flagged but disclaimer-adjacent samples (hits=1 or 2) ---")
    edge = [p for p in pages if not p["is_disclaimer"] and p["disclaimer_hits"] >= 1]
    for p in edge[:5]:
        prev = p["text"][:120].replace("\n", " ")
        print(f"  p.{p['page']:>5}  hits={p['disclaimer_hits']}  d={p['disclaimer_density']:.2f}  {prev!r}")


if __name__ == "__main__":
    main()
