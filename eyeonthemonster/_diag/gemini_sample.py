"""Generate 20 sample page-cards via Gemini 3.5 Flash on chart-bearing pages.

Stratifies the sample across years and chart densities, renders each page to PNG,
sends (image + cleaned text + metadata) to Gemini, writes results to
_diag/sample_cards.jsonl plus pretty-printed _diag/sample_cards.md for review.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import fitz
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from google import genai
from google.genai import types

PDF_PATH = ROOT / "Eye on the Monster.pdf"
PAGES = ROOT / "_diag" / "pages_clean.jsonl"
IMG_DIR = ROOT / "_diag" / "samples"
IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = ROOT / "_diag" / "sample_cards.jsonl"
OUT_MD = ROOT / "_diag" / "sample_cards.md"

PROMPT = """You are documenting one page from "Eye on the Market," a chart-heavy \
financial publication by Michael Cembalest at J.P. Morgan, so a downstream \
search system can find this page when a user asks about its topics.

Page metadata:
- Issue date: {issue_date}
- Page in PDF: {page_num}

You have the rendered page image and the extracted page text (legal footer \
already removed).

Write a natural-language description of this page that someone could read \
*instead of* seeing the page. Cover:

- The argument, observation, or topic being made
- For every chart, table, or visual: what it shows — title, axes, data series, \
the point it makes, specific numbers worth citing
- Any forecasts, predictions, or open questions the author poses
- Specific entities (countries, companies, sectors, policies, people) discussed

Style: flowing prose, no bullet lists, no headers, no markdown. Length should \
match the page's information density — a short commentary page might be ~100 \
words; a dense outlook page with multiple charts might be 300-500 words. Do \
not restate the issue date or page number. Do not include any boilerplate.

If the page is mostly a transition, cover image, or section break with little \
content, say so briefly in one sentence.

Extracted text (footer already removed):
---
{cleaned_text}
---"""


def pick_sample(pages: list[dict], n: int = 20, seed: int = 7) -> list[dict]:
    """Stratify sample across eras x chart-density."""
    chart_pages = [
        p for p in pages
        if p["is_chart_bearing"] and p["disclaimer_class"] != "skip"
        and p["issue_date"]
    ]

    def year(p):
        m = re.search(r"(19|20)\d{2}", p["issue_date"] or "")
        return int(m.group(0)) if m else 2010

    eras = {
        "2005-2009": [p for p in chart_pages if year(p) <= 2009],
        "2010-2014": [p for p in chart_pages if 2010 <= year(p) <= 2014],
        "2015-2019": [p for p in chart_pages if 2015 <= year(p) <= 2019],
        "2020-2024": [p for p in chart_pages if 2020 <= year(p) <= 2024],
        "2025+":     [p for p in chart_pages if year(p) >= 2025],
    }
    rng = random.Random(seed)
    picks: list[dict] = []
    per_era = max(1, n // len(eras))
    for era, ps in eras.items():
        if not ps:
            continue
        ps_sorted = sorted(ps, key=lambda p: p["n_drawings"])
        # take one low, one high, plus randoms to fill
        slots = []
        if ps_sorted:
            slots.append(ps_sorted[0])               # lowest drawing count in era
            slots.append(ps_sorted[-1])              # highest
        remaining = [p for p in ps if p not in slots]
        rng.shuffle(remaining)
        slots.extend(remaining[: per_era - len(slots)])
        picks.extend(slots[:per_era])

    # also ensure p.3456 is in there if it's chart-bearing (we eyeballed it)
    p3456 = next((p for p in chart_pages if p["page"] == 3456), None)
    if p3456 and not any(p["page"] == 3456 for p in picks):
        picks.append(p3456)
    return picks[:n]


def render_page(page_num: int) -> Path:
    out = IMG_DIR / f"page_{page_num:05d}.png"
    if not out.exists():
        doc = fitz.open(PDF_PATH)
        pg = doc.load_page(page_num - 1)
        pix = pg.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x for chart legibility
        pix.save(out)
        doc.close()
    return out


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not loaded")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    pages = [json.loads(l) for l in PAGES.open()]
    sample = pick_sample(pages, n=20)
    print(f"Selected {len(sample)} pages:")
    for p in sample:
        print(f"  p.{p['page']:>5}  [{p['issue_date']}]  drawings={p['n_drawings']}  "
              f"words={p['text_words']}")

    results = []
    md_lines = ["# Gemini 3.5 Flash sample page cards\n"]
    t0 = time.time()
    for i, p in enumerate(sample, 1):
        img_path = render_page(p["page"])
        img_bytes = img_path.read_bytes()
        prompt = PROMPT.format(
            issue_date=p["issue_date"],
            page_num=p["page"],
            cleaned_text=p["content_text"][:8000],
        )
        try:
            resp = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            card = (resp.text or "").strip()
        except Exception as e:
            card = f"<ERROR: {e}>"
        usage = getattr(resp, "usage_metadata", None) if "resp" in dir() else None
        rec = {
            "page": p["page"],
            "issue_date": p["issue_date"],
            "n_drawings": p["n_drawings"],
            "text_words": p["text_words"],
            "card": card,
            "card_words": len(card.split()),
        }
        if usage:
            rec["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
            }
        results.append(rec)
        md_lines.append(
            f"\n## p.{p['page']} — {p['issue_date']}  "
            f"(drawings={p['n_drawings']}, words={p['text_words']} → card={rec['card_words']}w)\n\n"
            f"![](samples/page_{p['page']:05d}.png)\n\n{card}\n"
        )
        print(f"  [{i}/{len(sample)}] p.{p['page']}  -> {rec['card_words']} word card")

    with OUT_JSONL.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    OUT_MD.write_text("".join(md_lines))

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    # Token totals if available
    total_in = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in results)
    total_out = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in results)
    print(f"Tokens — in: {total_in:,}  out: {total_out:,}")
    print(f"Outputs: {OUT_JSONL}  and  {OUT_MD}")


if __name__ == "__main__":
    main()
