"""Render index/cards.jsonl into a self-contained review_cards.html for eyeballing.

Page image (left) beside its generated card text (right), so you can verify the card
against the actual page. Inlines images as base64 — no server, just open the file.
Auto-flags short / section-break cards so the risky ones surface first.

    uv run python review_cards.py            # reviews everything in index/cards.jsonl
    uv run python review_cards.py --out x.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "Eye on the Monster.pdf"
CARDS = ROOT / "index" / "cards.jsonl"
PAGES = ROOT / "_diag" / "pages_clean.jsonl"

FALLBACK = re.compile(r"transition|cover image|section break|mostly blank|little content", re.I)


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def page_png_b64(doc: fitz.Document, page_num: int) -> str:
    pix = doc.load_page(page_num - 1).get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    return base64.b64encode(pix.tobytes("png")).decode()


def flags(card_text: str) -> list[str]:
    f = []
    n = len(card_text.split())
    if n < 60:
        f.append(f"short ({n}w)")
    if FALLBACK.search(card_text):
        f.append("section-break?")
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="review_cards.html")
    ap.add_argument("--sample", type=int, default=None, help="Render at most N cards after sorting flagged first.")
    args = ap.parse_args()

    cards = load_jsonl(CARDS)
    meta = {int(p["page"]): p for p in load_jsonl(PAGES)}
    doc = fitz.open(PDF_PATH)

    # flagged cards first, then by page
    for c in cards:
        c["_flags"] = flags(c.get("card_text") or "")
    cards.sort(key=lambda c: (not c["_flags"], int(c["page"])))

    n_flag = sum(1 for c in cards if c["_flags"])
    if args.sample is not None:
        cards = cards[: args.sample]
    rows = []
    for c in cards:
        pg = int(c["page"])
        m = meta.get(pg, {})
        date = m.get("issue_date") or "?"
        words = len((c.get("card_text") or "").split())
        badge = (" ".join(f'<span class="flag">{html.escape(x)}</span>' for x in c["_flags"])) or \
                '<span class="ok">ok</span>'
        try:
            img = f'data:image/png;base64,{page_png_b64(doc, pg)}'
        except Exception as e:  # noqa: BLE001
            img = ""
            badge += f'<span class="flag">render error: {html.escape(str(e))}</span>'
        rows.append(f"""
        <div class="row">
          <div class="meta">p.{pg} &middot; {html.escape(str(date))} &middot; {words}w {badge}</div>
          <div class="pair">
            <div class="img">{'<img src="'+img+'">' if img else '(no image)'}</div>
            <div class="card">{html.escape(c.get('card_text') or '')}</div>
          </div>
        </div>""")

    out = f"""<!doctype html><meta charset="utf-8"><title>Card review</title>
<style>
  body{{font:14px/1.55 system-ui;margin:0;background:#f4f4f5}}
  header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:.7rem 1.2rem;font-weight:600}}
  .row{{background:#fff;margin:1rem;border:1px solid #e4e4e7;border-radius:8px;overflow:hidden}}
  .meta{{padding:.5rem .9rem;background:#fafafa;border-bottom:1px solid #eee;font-size:13px;color:#555}}
  .pair{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:1rem;padding:1rem;align-items:start}}
  .img img{{width:100%;border:1px solid #ddd;border-radius:4px}}
  .card{{white-space:pre-wrap;color:#222}}
  .flag{{background:#fee2e2;color:#b91c1c;border-radius:4px;padding:0 .4em;margin-left:.3em;font-size:12px}}
  .ok{{background:#dcfce7;color:#166534;border-radius:4px;padding:0 .4em;margin-left:.3em;font-size:12px}}
</style>
<header>{len(cards)} cards &middot; {n_flag} flagged for review &middot; flagged shown first</header>
{''.join(rows)}
"""
    Path(args.out).write_text(out)
    print(f"wrote {args.out} — {len(cards)} cards, {n_flag} flagged")


if __name__ == "__main__":
    main()
