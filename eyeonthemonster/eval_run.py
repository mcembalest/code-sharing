from __future__ import annotations

import argparse
import asyncio
import json
import re
from typing import Any

from agent import run_agent


PAGE_RE = re.compile(r"\[p{1,2}\.\s*(\d+)(?:\s*-\s*(\d+))?\]")


def cited_pages_from_text(text: str) -> set[int]:
    pages: set[int] = set()
    for match in PAGE_RE.finditer(text or ""):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        pages.update(range(start, end + 1))
    return pages


def event_pages(event: dict[str, Any]) -> set[int]:
    pages: set[int] = set()
    page = event.get("page")
    if isinstance(page, int):
        pages.add(page)
    for field in ("text", "caption", "summary"):
        pages.update(cited_pages_from_text(str(event.get(field) or "")))
    return pages


async def collect(query: str) -> dict[str, Any]:
    coverage = []
    cited_pages: set[int] = set()
    events = 0
    async for event in run_agent(query):
        events += 1
        cited_pages.update(event_pages(event))
        if event.get("lane") == "trace" and event.get("type") == "tool_result" and event.get("name") == "coverage_status":
            coverage.append(event.get("summary") or "")
    return {"query": query, "events": events, "coverage_status": coverage, "cited_pages": sorted(cited_pages)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Eye on the Monster agent headlessly for evals.")
    parser.add_argument("query", nargs="?", help="Query to run. If omitted, stdin is used.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    query = args.query or input().strip()
    result = await collect(query)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print("coverage_status trajectory:")
    if result["coverage_status"]:
        for item in result["coverage_status"]:
            print(f"- {item}")
    else:
        print("- none")
    print("cited pages:")
    print(", ".join(f"p. {page}" for page in result["cited_pages"]) or "none")


if __name__ == "__main__":
    asyncio.run(main())
