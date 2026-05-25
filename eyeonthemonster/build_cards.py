from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import fitz
from dotenv import load_dotenv
from google import genai
from google.genai import types


ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "Eye on the Monster.pdf"
PAGES_PATH = ROOT / "_diag" / "pages_clean.jsonl"
OUT = ROOT / "index" / "cards.jsonl"
MODEL = "gemini-3.5-flash"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Gemini vision cards for chart-bearing pages.")
    parser.add_argument("--workers", type=int, default=8, help="Maximum concurrent Gemini calls.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N missing pages.")
    parser.add_argument("--dry-run", action="store_true", help="Print the worklist without API calls.")
    return parser.parse_args()


def is_true(value: object) -> bool:
    return value is True or str(value).lower() == "true"


def load_pages() -> list[dict]:
    with PAGES_PATH.open() as f:
        return [json.loads(line) for line in f]


def load_done_pages() -> set[int]:
    if not OUT.exists():
        return set()

    done: set[int] = set()
    with OUT.open() as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: ignoring invalid JSON on {OUT}:{line_num}", file=sys.stderr)
                continue
            if rec.get("card_text"):
                done.add(int(rec["page"]))
    return done


def chart_pages(pages: list[dict], done: set[int]) -> list[dict]:
    return [
        p
        for p in pages
        if is_true(p.get("is_chart_bearing"))
        and p.get("disclaimer_class") != "skip"
        and int(p["page"]) not in done
    ]


def render_page_png(page_num: int) -> bytes:
    with fitz.open(PDF_PATH) as doc:
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        return pix.tobytes("png")


def generate_card(client: genai.Client, page: dict) -> dict:
    page_num = int(page["page"])
    prompt = PROMPT.format(
        issue_date=page.get("issue_date") or "unknown",
        page_num=page_num,
        cleaned_text=(page.get("content_text") or "")[:8000],
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=render_page_png(page_num), mime_type="image/png"),
            prompt,
        ],
    )
    card_text = (resp.text or "").strip()
    rec = {"page": page_num, "card_text": card_text}

    usage = getattr(resp, "usage_metadata", None)
    if usage:
        rec["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        }
    return rec


async def worker(
    name: int,
    client: genai.Client,
    queue: asyncio.Queue[tuple[int, dict]],
    write_lock: asyncio.Lock,
    counters: dict[str, int],
    total: int,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        ordinal, page = item
        page_num = int(page["page"])
        try:
            rec = await asyncio.to_thread(generate_card, client, page)
            async with write_lock:
                with OUT.open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counters["ok"] += 1
                words = len(rec["card_text"].split())
                print(f"[{ordinal}/{total}] worker={name} p.{page_num} -> {words} words")
        except Exception as exc:
            async with write_lock:
                counters["error"] += 1
                print(f"[{ordinal}/{total}] worker={name} p.{page_num} ERROR: {exc}", file=sys.stderr)
        finally:
            queue.task_done()


async def run(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("GEMINI_API_KEY not loaded")

    done = load_done_pages()
    pending = chart_pages(load_pages(), done)
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"cards already present: {len(done)}")
    print(f"pending chart-bearing kept pages selected: {len(pending)}")

    if args.dry_run:
        for page in pending[:20]:
            print(f"  p.{int(page['page']):>5}  {page.get('issue_date') or 'unknown-date'}")
        if len(pending) > 20:
            print(f"  ... {len(pending) - 20} more")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=api_key)
    queue: asyncio.Queue[tuple[int, dict]] = asyncio.Queue()
    for ordinal, page in enumerate(pending, 1):
        queue.put_nowait((ordinal, page))

    counters = {"ok": 0, "error": 0}
    write_lock = asyncio.Lock()
    tasks = [
        asyncio.create_task(worker(i + 1, client, queue, write_lock, counters, len(pending)))
        for i in range(max(1, args.workers))
    ]
    for _ in tasks:
        queue.put_nowait(None)

    t0 = time.time()
    await queue.join()
    await asyncio.gather(*tasks)
    print(
        f"finished in {time.time() - t0:.1f}s: "
        f"{counters['ok']} written, {counters['error']} errors"
    )


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
