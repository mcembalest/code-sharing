from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from agent import run_agent


PAGE_RE = re.compile(r"\[p{1,2}\.\s*(\d+)(?:\s*-\s*(\d+))?\]")
ENUMERATE_RE = re.compile(
    r"(?P<coverage_total>\d+) pages tagged across (?P<n_topics>\d+) topic\(s\); "
    r"surfaced (?P<surfaced>\d+) on-point \((?P<primary>\d+) primary\) at floor (?P<floor>[0-9.]+)"
    r"(?:, aggregated (?P<below_floor>\d+) below floor)?"
)


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


def parse_enumerate_summary(summary: str) -> dict[str, Any] | None:
    match = ENUMERATE_RE.search(summary or "")
    if not match:
        return None
    out: dict[str, Any] = {
        "coverage_total": int(match.group("coverage_total")),
        "n_topics": int(match.group("n_topics")),
        "surfaced": int(match.group("surfaced")),
        "primary": int(match.group("primary")),
        "floor": float(match.group("floor")),
        "below_floor": int(match.group("below_floor") or 0),
        "summary": summary,
    }
    return out


async def collect(query: str) -> dict[str, Any]:
    coverage = []
    tool_calls = []
    tool_results = []
    enumerate_runs = []
    report_items = 0
    committed_pages: set[int] = set()
    cited_pages: set[int] = set()
    answer_pages: set[int] = set()
    events = 0
    started = time.perf_counter()
    async for event in run_agent(query):
        events += 1
        cited_pages.update(event_pages(event))
        if event.get("lane") == "report":
            report_items += 1
            if event.get("kind") in {"quote", "chart"}:
                committed_pages.update(event_pages(event))
            if event.get("kind") == "answer":
                answer_pages.update(cited_pages_from_text(str(event.get("text") or "")))
        if event.get("lane") != "trace":
            continue
        if event.get("type") == "tool_call":
            tool_calls.append({"name": event.get("name"), "args": event.get("args")})
        if event.get("type") == "tool_result":
            summary = event.get("summary") or ""
            tool_results.append({"name": event.get("name"), "summary": summary})
            if event.get("name") == "coverage_status":
                coverage.append(summary)
            if event.get("name") == "enumerate":
                # Prefer the structured accounting now emitted on the event; fall back to parsing
                # the summary string only if those fields are absent (older agent build).
                if event.get("coverage_total") is not None:
                    enumerate_runs.append({
                        "coverage_total": event.get("coverage_total"),
                        "surfaced": event.get("surfaced"),
                        "below_floor": event.get("below_floor"),
                        "primary": event.get("primary"),
                        "floor": event.get("floor"),
                        "topics": event.get("topics"),
                        "summary": summary,
                    })
                else:
                    parsed = parse_enumerate_summary(summary)
                    if parsed:
                        enumerate_runs.append(parsed)
    elapsed = time.perf_counter() - started
    return {
        "query": query,
        "elapsed_sec": round(elapsed, 3),
        "events": events,
        "tool_call_count": len(tool_calls),
        "tool_path": [str(call.get("name") or "") for call in tool_calls],
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "enumerate_runs": enumerate_runs,
        "report_items": report_items,
        "coverage_status": coverage,
        "committed_pages": sorted(committed_pages),
        "committed_page_count": len(committed_pages),
        "cited_pages": sorted(cited_pages),
        "cited_page_count": len(cited_pages),
        "answer_pages": sorted(answer_pages),
        "answer_page_count": len(answer_pages),
    }


def score(result: dict[str, Any], expected: list[int]) -> dict[str, Any]:
    """Recall/precision of the surfaced (committed) page set against a human golden key. This is the
    signal that calibrates RELEVANCE_FLOOR: low recall => the floor pushed relevant pages into the
    below-floor tail (lower it); low precision => the floor is too permissive (raise it). No golden
    key ships in-repo — it is owner-gated; this stays dormant until --golden is provided."""
    expected_set = set(expected)
    got = set(result.get("committed_pages") or [])
    hit = expected_set & got
    return {
        "expected": len(expected_set),
        "committed": len(got),
        "hit": len(hit),
        "recall": round(len(hit) / len(expected_set), 3) if expected_set else None,
        "precision": round(len(hit) / len(got), 3) if got else None,
        "missed_pages": sorted(expected_set - got),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Eye on the Monster agent headlessly for evals.")
    parser.add_argument("query", nargs="?", help="Query to run. If omitted, stdin is used.")
    parser.add_argument("--golden", help="Path to a JSON golden key {query: [expected_page, ...]} for recall/precision scoring.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    query = args.query or input().strip()
    result = await collect(query)
    golden = json.loads(Path(args.golden).read_text()) if args.golden else {}
    if query in golden:
        result["score"] = score(result, golden[query])
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"elapsed: {result['elapsed_sec']:.3f}s")
    print(f"events: {result['events']}")
    print(f"tool calls: {result['tool_call_count']}")
    print("tool path:")
    for name in result["tool_path"]:
        print(f"- {name}")
    print("enumerate runs:")
    if result["enumerate_runs"]:
        for run in result["enumerate_runs"]:
            print(
                "- "
                f"coverage_total={run['coverage_total']}, surfaced={run['surfaced']}, "
                f"below_floor={run['below_floor']}, primary={run['primary']}, floor={run['floor']}"
            )
    else:
        print("- none")
    print("coverage_status trajectory:")
    if result["coverage_status"]:
        for item in result["coverage_status"]:
            print(f"- {item}")
    else:
        print("- none")
    print(f"report items: {result['report_items']}")
    print(f"committed pages ({result['committed_page_count']}):")
    print(", ".join(f"p. {page}" for page in result["committed_pages"]) or "none")
    print("cited pages:")
    print(", ".join(f"p. {page}" for page in result["cited_pages"]) or "none")
    print(f"answer pages ({result['answer_page_count']}):")
    print(", ".join(f"p. {page}" for page in result["answer_pages"]) or "none")
    if result.get("score"):
        sc = result["score"]
        print(f"SCORE vs golden: recall={sc['recall']} precision={sc['precision']} "
              f"(hit {sc['hit']}/{sc['expected']}); missed {sc['missed_pages'] or 'none'}")


if __name__ == "__main__":
    asyncio.run(main())
