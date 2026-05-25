from __future__ import annotations

import asyncio
import contextvars
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)
from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "index"
# Load .env at IMPORT time, before query() spawns the claude CLI subprocess: the subprocess
# inherits os.environ at spawn, so ANTHROPIC_API_KEY must be set first for the SDK to auth with
# the API key. If it's loaded lazily (inside _load), the subprocess is already running with the
# parent env (no key) and falls back to the Claude subscription/OAuth login instead.
load_dotenv(ROOT / ".env")
MODEL_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"
AGENT_MODEL = "claude-haiku-4-5-20251001"
SYSTEM = """You are the research agent for "Eye on the Market" — 20 years (~5,100 pages) of \
Michael Cembalest's J.P. Morgan letters, indexed page-by-page. Your job is EXHAUSTIVE recall: \
given a question like "what is everything ever said or shown about X," find and present every \
relevant page, cite a page for every claim, and assemble the answer LIVE as you work.

OUTPUT CONTRACT
- Every claim carries a page citation: [p. N] for one page, [pp. N-M] for a span.
- Commit each finding the moment you confirm it, via add_to_report; the report IS the
  accumulation of those calls (do NOT hoard findings for a single end dump). Kinds:
    heading   {text}                                          open a section / sub-topic
    quote     {text, page, issue_date, title, relevance, src} a textual finding (verbatim or tight)
    chart     {page, caption, relevance, src}                 a relevant chart/figure (image is rendered)
    narrative {text}                                          connective synthesis across findings
    answer    {text}                                          ONE final direct answer (see below)
  - relevance (quote/chart): "primary" if the finding DIRECTLY answers the question, "supporting"
    for context/corroboration. Default is "supporting" — reserve "primary" for the strongest,
    most on-point evidence so the reader's eye lands there first.
  - src = a short note on what surfaced it (e.g. "search_topic:solar-pv-costs" or
    "search_semantic:'LCOE'") so the trace and report stay linked.
- FINISH with exactly one kind="answer": a direct, self-contained answer to the question in a few
  sentences, with inline [p. N] citations to your primary findings. Emit it only after
  coverage_status is exhausted for your scope. This pinned block is the first thing the reader sees.

RETRIEVAL — use several signals; trust no single one
1. Decompose the question into sub-aspects and likely sub-topics.
2. DISCOVER topics: call list_topics for the high-level buckets (page counts + a few sample
   granular ids each), then list_topics(high_level=<bucket id>) to expand the full granular id
   list under a relevant bucket. Pick the relevant GRANULAR topic ids — those are what
   search_topic takes. Buckets are keyword-routed and mostly coherent, but "other" is a broad
   catch-all and any single label can mislead, so route on granular topics and corroborate.
3. ENUMERATE: call search_topic on each relevant granular topic to pull its full tagged page set.
   This is your recall backbone for "everything about X."
4. WIDEN: run search_semantic AND search_keyword with several reformulations — vocabulary drifts
   across 20 years ("solar cost" 2005 vs "module ASP / LCOE" 2025). Tagging misses things; search
   catches stragglers. Use chart_only=true when the question is about what was *shown*.
5. READ context with get_page / get_pages / get_issue, and follow cross-references
   ("as discussed in last year's energy paper").

EXHAUSTIVENESS — coverage is tracked by tools, never by memory
- The system compacts context, so DO NOT rely on remembering what you checked. The coverage
  tools are the only source of truth.
- Decide your scope (a topic_id and/or a date_range).
- As you examine an issue's pages, call mark_inspected with that issue's id.
- Periodically call coverage_status (scoped by the same topic_id/date_range) to see what remains.
  Keep working until remaining is empty — or until the only remaining issues are, on inspection,
  genuinely irrelevant (say so in a narrative note).

ANALYTICAL QUESTIONS ("what did I get wrong", "what open questions did I pose")
Not findable by similarity alone. Decompose: for "got wrong," gather the author's predictions on a
topic, then find later issues whose events contradict them, and cite both sides. For "open
questions," scan for explicit question framings and unresolved threads.

STYLE
- Group findings by sub-topic and/or chronology. Prefer verbatim quotes and real charts over paraphrase.
- When a chart matters, surface it (kind=chart) with the actual numbers in the caption.
- Every line the reader sees carries a citation. Stop when coverage_status is exhausted for your
  scope, every confirmed aspect is in the report, and you have emitted the final answer block.
"""

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_WS = re.compile(r"\s+")
# Stored issue_date is human-readable ("April 06, 2005", "May 8th, 2006"), but the agent scopes
# searches with ISO bounds ("2005-01-01"). A raw string compare puts 'A' > '2', so any date-scoped
# search silently returned 0 hits. Normalize both sides to sortable YYYY-MM-DD before comparing.
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}
_ISO_DATE = re.compile(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?")
_HUMAN_DATE = re.compile(r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", re.I)


def _norm_date(value: object) -> str | None:
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    m = _ISO_DATE.match(s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3) or 1):02d}"
    m = _HUMAN_DATE.search(s)
    if m and m.group(1).lower() in _MONTHS:
        return f"{int(m.group(3)):04d}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    return None
_emit_queue: contextvars.ContextVar[asyncio.Queue | None] = contextvars.ContextVar("emit_queue", default=None)
# Coverage state is per-run, not global: the server is long-lived and serves many queries
# (possibly concurrently), so a module-level set would leak one query's inspected issues into
# the next and corrupt coverage_status. A contextvar isolates it per run_agent call.
_inspected_var: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar("inspected", default=None)
MAX_SEARCH_LIMIT = 10
MAX_LIST_LIMIT = 10
MAX_TOPIC_CHILDREN = 5
MAX_GRANULAR_DRILL = 100  # granular are sorted by page count, so this is the top-100 by coverage
MAX_TEXT_CHARS = 400
MAX_ENUMERATE = 500  # safety ceiling on a single bulk enumeration (vs. the 10-cap on previews)
DEFAULT_PRIMARY_K = 6  # top-N by query similarity marked "primary"; the rest "supporting"


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _load() -> dict[str, Any]:
    if hasattr(_load, "state"):
        return _load.state  # type: ignore[attr-defined]
    emb = np.load(INDEX_DIR / "embeddings.npy")
    pages = json.loads((INDEX_DIR / "pages.json").read_text())
    with (INDEX_DIR / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)["bm25"]
    issues = []
    with (INDEX_DIR / "issues.jsonl").open() as f:
        for line in f:
            if line.strip():
                issues.append(json.loads(line))
    topics = json.loads((INDEX_DIR / "topics.json").read_text()) if (INDEX_DIR / "topics.json").exists() else {"high_level": []}
    model = SentenceTransformer(MODEL_NAME)
    page_by_num = {int(p["page"]): p for p in pages}
    row_by_page = {int(p["page"]): i for i, p in enumerate(pages)}  # page num -> embeddings.npy row
    issue_by_id = {i["issue_id"]: i for i in issues}
    pages_by_topic: dict[str, list[dict]] = defaultdict(list)
    for page in pages:
        for topic_id in set(page.get("high_level", []) + page.get("granular", [])):
            pages_by_topic[topic_id].append(page)
    label_by_topic: dict[str, str] = {}  # topic id -> human label, for server-side headings
    for bucket in topics.get("high_level", []):
        if bucket.get("id"):
            label_by_topic[bucket["id"]] = bucket.get("label") or bucket["id"]
        for gran in bucket.get("granular", []):
            if gran.get("id"):
                label_by_topic[gran["id"]] = gran.get("label") or gran["id"]
    _load.state = {
        "emb": emb,
        "pages": pages,
        "bm25": bm25,
        "issues": issues,
        "topics": topics,
        "model": model,
        "page_by_num": page_by_num,
        "row_by_page": row_by_page,
        "issue_by_id": issue_by_id,
        "pages_by_topic": pages_by_topic,
        "label_by_topic": label_by_topic,
    }
    return _load.state  # type: ignore[attr-defined]


async def emit(ev: dict[str, Any]) -> None:
    queue = _emit_queue.get()
    if queue is not None:
        await queue.put(ev)


def inspected() -> set[str]:
    s = _inspected_var.get()
    if s is None:
        s = set()
        _inspected_var.set(s)
    return s


def in_date_range(rec: dict, date_range: list[str] | None) -> bool:
    if not date_range or len(date_range) != 2 or not date_range[0] or not date_range[1]:
        return True
    value = _norm_date(rec.get("issue_date"))
    if value is None:
        return False  # undated pages can't be placed in a scoped range
    lo = _norm_date(date_range[0]) or date_range[0]
    hi = _norm_date(date_range[1]) or date_range[1]
    return lo <= value <= hi


def snippet(text: str, limit: int = 300) -> str:
    return _WS.sub(" ", text).strip()[:limit]


def bounded_limit(value: object, default: int = MAX_SEARCH_LIMIT, maximum: int = MAX_SEARCH_LIMIT) -> int:
    try:
        limit = int(value or default)
    except (TypeError, ValueError):
        limit = default
    return max(1, min(limit, maximum))


def compact_issue(issue: dict) -> dict:
    return {
        "issue_id": issue.get("issue_id"),
        "issue_date": issue.get("issue_date"),
        "title": issue.get("title"),
        "page_start": issue.get("page_start"),
        "page_end": issue.get("page_end"),
        "n_pages": issue.get("n_pages"),
    }


def compact_page(page: dict) -> dict:
    return {
        "page": int(page["page"]),
        "issue_id": page.get("issue_id"),
        "issue_date": page.get("issue_date"),
        "title": page.get("title"),
        "is_chart_bearing": bool(page.get("is_chart_bearing")),
        "high_level": page.get("high_level", [])[:8],
        "granular": page.get("granular", [])[:20],
        "content_text": snippet(page.get("content_text") or "", MAX_TEXT_CHARS),
        "card_text": snippet(page.get("card_text") or "", MAX_TEXT_CHARS),
    }


def format_hit(page: dict, score: float) -> dict:
    return {
        "page": int(page["page"]),
        "issue_id": page.get("issue_id"),
        "issue_date": page.get("issue_date"),
        "title": page.get("title"),
        "score": float(score),
        "snippet": snippet((page.get("card_text") or "") or (page.get("content_text") or "")),
    }


def filtered_indices(date_range: list[str] | None, chart_only: bool, topic_id: str | None) -> list[int]:
    pages = _load()["pages"]
    out = []
    for i, page in enumerate(pages):
        if chart_only and not page.get("is_chart_bearing"):
            continue
        if topic_id and topic_id not in set(page.get("high_level", []) + page.get("granular", [])):
            continue
        if not in_date_range(page, date_range):
            continue
        out.append(i)
    return out


@tool(
    "search_keyword",
    "Keyword search over merged page text, chart cards, and topic labels.",
    {"q": str, "date_range": list, "chart_only": bool, "limit": int, "topic_id": str},
)
async def search_keyword(args: dict) -> dict:
    s = _load()
    limit = bounded_limit(args.get("limit"))
    indices = filtered_indices(args.get("date_range"), bool(args.get("chart_only", False)), args.get("topic_id"))
    scores = np.asarray(s["bm25"].get_scores(tokenize(args["q"])), dtype=np.float32)
    ordered = sorted(indices, key=lambda i: float(scores[i]), reverse=True)[:limit]
    hits = [format_hit(s["pages"][i], float(scores[i])) for i in ordered]
    await emit({"lane": "trace", "type": "tool_result", "name": "search_keyword", "summary": f"{len(hits)} hits: " + ", ".join(f"p.{h['page']}" for h in hits[:8])})
    return {"content": [{"type": "text", "text": json.dumps(hits)}]}


@tool(
    "search_semantic",
    "Semantic search over merged page text, chart cards, and topic labels.",
    {"q": str, "date_range": list, "chart_only": bool, "limit": int, "topic_id": str},
)
async def search_semantic(args: dict) -> dict:
    s = _load()
    limit = bounded_limit(args.get("limit"))
    indices = filtered_indices(args.get("date_range"), bool(args.get("chart_only", False)), args.get("topic_id"))
    q = s["model"].encode([args["q"]], convert_to_numpy=True)[0]
    q = q / (np.linalg.norm(q) or 1.0)
    scores = (s["emb"] @ q).astype(np.float32)
    ordered = sorted(indices, key=lambda i: float(scores[i]), reverse=True)[:limit]
    hits = [format_hit(s["pages"][i], float(scores[i])) for i in ordered]
    await emit({"lane": "trace", "type": "tool_result", "name": "search_semantic", "summary": f"{len(hits)} hits: " + ", ".join(f"p.{h['page']}" for h in hits[:8])})
    return {"content": [{"type": "text", "text": json.dumps(hits)}]}


@tool("list_issues", "List issue records, optionally scoped by date range.", {"date_range": list})
async def list_issues(args: dict) -> dict:
    issues = [compact_issue(i) for i in _load()["issues"] if in_date_range(i, args.get("date_range"))]
    total = len(issues)
    issues = issues[:MAX_LIST_LIMIT]
    await emit({"lane": "trace", "type": "tool_result", "name": "list_issues", "summary": f"{total} issues; returned {len(issues)}"})
    return {"content": [{"type": "text", "text": json.dumps({"total": total, "issues": issues})}]}


@tool(
    "list_topics",
    "Without args: list the high-level buckets (page counts + a few sample granular ids each). "
    "Pass high_level=<bucket id> to expand the granular topic ids in that bucket — those ids are what search_topic takes.",
    {"high_level": str},
)
async def list_topics(args: dict) -> dict:
    buckets = _load()["topics"].get("high_level", [])
    hid = args.get("high_level")
    if hid:
        bucket = next((b for b in buckets if b.get("id") == hid), None)
        granular = bucket.get("granular", []) if bucket else []
        items = [{"id": g.get("id"), "label": g.get("label"), "n_pages": int(g.get("n_pages", 0))} for g in granular[:MAX_GRANULAR_DRILL]]
        await emit({"lane": "trace", "type": "tool_result", "name": "list_topics", "summary": f"{hid}: {len(items)}/{len(granular)} granular topics"})
        return {"content": [{"type": "text", "text": json.dumps({"high_level": hid, "n_granular": len(granular), "returned": len(items), "granular": items})}]}
    overview = []
    for bucket in buckets:
        granular = bucket.get("granular", [])
        overview.append({
            "id": bucket.get("id"),
            "label": bucket.get("label"),
            "n_granular": len(granular),
            "n_pages": int(bucket.get("n_pages", 0)),
            "top_granular": [{"id": g.get("id"), "label": g.get("label"), "n_pages": int(g.get("n_pages", 0))} for g in granular[:MAX_TOPIC_CHILDREN]],
        })
    await emit({"lane": "trace", "type": "tool_result", "name": "list_topics", "summary": f"{len(overview)} high-level buckets"})
    return {"content": [{"type": "text", "text": json.dumps({"high_level": overview})}]}


@tool(
    "search_topic",
    "PREVIEW a topic: returns the true total tagged-page count plus a small sample (top 10). To "
    "actually pull every tagged page into the report, use enumerate (it has no 10-cap).",
    {"topic_id": str, "date_range": list},
)
async def search_topic(args: dict) -> dict:
    pages = [p for p in _load()["pages_by_topic"].get(args["topic_id"], []) if in_date_range(p, args.get("date_range"))]
    hits = [format_hit(p, 1.0) for p in pages[:MAX_LIST_LIMIT]]
    await emit({"lane": "trace", "type": "tool_result", "name": "search_topic", "summary": f"{len(pages)} tagged pages for {args['topic_id']}; returned {len(hits)}"})
    return {"content": [{"type": "text", "text": json.dumps({"total": len(pages), "sample": hits, "note": "sample only; call enumerate for the full set"})}]}


@tool(
    "enumerate",
    "RECALL BACKBONE for \"everything about X\": given one or more granular topic_ids, commit EVERY "
    "tagged page to the report in a SINGLE call -- no per-page get_page/add_to_report needed. Unions "
    "the topics, dedups by page, ranks by similarity to q (top results -> relevance=primary, rest "
    "supporting), and emits each page as a chart (if chart-bearing) or quote. Pass the user's "
    "question as q to drive ranking. Prefer this over looping search_topic + add_to_report by hand.",
    {"topic_ids": list, "q": str, "date_range": list, "chart_only": bool, "primary_k": int},
)
async def enumerate_topics(args: dict) -> dict:
    s = _load()
    topic_ids = [str(t) for t in (args.get("topic_ids") or []) if t]
    date_range = args.get("date_range")
    chart_only = bool(args.get("chart_only", False))
    seen: set[int] = set()
    pages: list[dict] = []
    for tid in topic_ids:
        for p in s["pages_by_topic"].get(tid, []):
            n = int(p["page"])
            if n in seen or (chart_only and not p.get("is_chart_bearing")) or not in_date_range(p, date_range):
                continue
            seen.add(n)
            pages.append(p)
    q = (args.get("q") or "").strip()
    if q and pages:  # rank by query similarity; without q, keep page (chronological) order
        rows = [s["row_by_page"][int(p["page"])] for p in pages]
        qv = s["model"].encode([q], convert_to_numpy=True)[0]
        qv = qv / (np.linalg.norm(qv) or 1.0)
        order = np.argsort(-(s["emb"][rows] @ qv).astype(np.float32))
        pages = [pages[i] for i in order]
    pages = pages[:MAX_ENUMERATE]
    primary_k = bounded_limit(args.get("primary_k"), default=DEFAULT_PRIMARY_K, maximum=MAX_ENUMERATE)
    labels = [s["label_by_topic"].get(t, t) for t in topic_ids]
    if labels:
        await emit({"lane": "report", "kind": "heading", "text": "; ".join(labels[:4]) + (" …" if len(labels) > 4 else "")})
    src = "enumerate:" + ",".join(topic_ids[:3])
    n_primary = 0
    for rank, p in enumerate(pages):
        relevance = "primary" if rank < primary_k else "supporting"
        n_primary += relevance == "primary"
        if p.get("is_chart_bearing"):
            await emit({"lane": "report", "kind": "chart", "page": int(p["page"]),
                        "issue_date": p.get("issue_date"), "title": p.get("title"),
                        "caption": snippet((p.get("card_text") or p.get("content_text") or ""), 160),
                        "relevance": relevance, "src": src})
        else:
            await emit({"lane": "report", "kind": "quote", "page": int(p["page"]),
                        "issue_date": p.get("issue_date"), "title": p.get("title"),
                        "text": snippet((p.get("content_text") or p.get("card_text") or ""), MAX_TEXT_CHARS),
                        "relevance": relevance, "src": src})
    await emit({"lane": "trace", "type": "tool_result", "name": "enumerate",
                "summary": f"committed {len(pages)} pages across {len(topic_ids)} topic(s) ({n_primary} primary)"})
    # tiny ack: the agent does NOT need the pages back -- they're already committed to the report
    return {"content": [{"type": "text", "text": json.dumps({"committed": len(pages), "primary": n_primary, "topics": topic_ids})}]}


@tool("get_issue", "Return full page records for one issue.", {"issue_id": str})
async def get_issue(args: dict) -> dict:
    issue = _load()["issue_by_id"].get(args["issue_id"])
    pages = []
    if issue:
        pages = [p for p in _load()["pages"] if int(issue["page_start"]) <= int(p["page"]) <= int(issue["page_end"])]
    await emit({"lane": "trace", "type": "tool_result", "name": "get_issue", "summary": f"{args['issue_id']} -> {len(pages)} pages"})
    return {"content": [{"type": "text", "text": json.dumps({"issue": compact_issue(issue) if issue else None, "pages": [compact_page(p) for p in pages[:MAX_LIST_LIMIT]], "total_pages": len(pages)})}]}


@tool("get_page", "Return one merged page record.", {"page": int})
async def get_page(args: dict) -> dict:
    page = _load()["page_by_num"].get(int(args["page"]))
    await emit({"lane": "trace", "type": "tool_result", "name": "get_page", "summary": f"p.{args['page']}"})
    return {"content": [{"type": "text", "text": json.dumps(compact_page(page) if page else None)}]}


@tool("get_pages", "Return an inclusive span of merged page records.", {"start": int, "end": int})
async def get_pages(args: dict) -> dict:
    pages = [p for p in _load()["pages"] if int(args["start"]) <= int(p["page"]) <= int(args["end"])]
    await emit({"lane": "trace", "type": "tool_result", "name": "get_pages", "summary": f"p.{args['start']}-p.{args['end']} -> {len(pages)} pages"})
    return {"content": [{"type": "text", "text": json.dumps({"total": len(pages), "pages": [compact_page(p) for p in pages[:MAX_LIST_LIMIT]]})}]}


@tool("add_to_report", "Commit a confirmed finding to the live report.", {"kind": str, "text": str, "page": int, "issue_date": str, "title": str, "caption": str, "relevance": str, "src": str})
async def add_to_report(args: dict) -> dict:
    args = dict(args)
    if args.get("kind") not in {"quote", "chart", "heading", "narrative", "answer"}:
        args["kind"] = "narrative"
    if args["kind"] in {"quote", "chart"} and args.get("relevance") not in {"primary", "supporting"}:
        args["relevance"] = "supporting"
    await emit({"lane": "report", **{k: v for k, v in args.items() if v is not None}})
    return {"content": [{"type": "text", "text": "added"}]}


@tool("mark_inspected", "Record issue ids that have been examined.", {"issue_ids": list})
async def mark_inspected(args: dict) -> dict:
    seen = inspected()
    seen.update(str(i) for i in args.get("issue_ids", []))
    await emit({"lane": "trace", "type": "tool_result", "name": "mark_inspected", "summary": f"{len(args.get('issue_ids', []))} issues marked inspected"})
    return {"content": [{"type": "text", "text": json.dumps({"inspected": sorted(seen)})}]}


@tool("coverage_status", "Report issue coverage for an optional date range and/or topic id.", {"date_range": list, "topic_id": str})
async def coverage_status(args: dict) -> dict:
    topic_id = args.get("topic_id")
    if topic_id:
        issue_ids = {p.get("issue_id") for p in _load()["pages_by_topic"].get(topic_id, []) if p.get("issue_id") and in_date_range(p, args.get("date_range"))}
    else:
        issue_ids = {i["issue_id"] for i in _load()["issues"] if in_date_range(i, args.get("date_range"))}
    seen = inspected()
    remaining = sorted(issue_ids - seen)
    out = {"inspected": len(issue_ids & seen), "total": len(issue_ids), "remaining": remaining}
    await emit({"lane": "trace", "type": "tool_result", "name": "coverage_status", "summary": f"{out['inspected']}/{out['total']} inspected"})
    return {"content": [{"type": "text", "text": json.dumps(out)}]}


TOOLS = [
    search_keyword,
    search_semantic,
    list_issues,
    list_topics,
    search_topic,
    enumerate_topics,
    get_issue,
    get_page,
    get_pages,
    add_to_report,
    mark_inspected,
    coverage_status,
]

eom = create_sdk_mcp_server(name="eom", version="1.0.0", tools=TOOLS)


async def _only_eom(tool_name: str, input_data: dict, context):
    # allowed_tools only governs auto-approval; the SDK's built-in tools (Bash, Read,
    # Grep, ...) stay reachable and the agent WILL shell out (e.g. grep its own on-disk
    # tool-result cache) if left open. Deny anything that isn't one of our tools so it
    # must go through the indexes — never the filesystem or shell.
    if tool_name.startswith("mcp__eom__"):
        return PermissionResultAllow(updated_input=input_data)
    return PermissionResultDeny(message="Use only the eom search tools.", interrupt=False)


options = ClaudeAgentOptions(
    model=AGENT_MODEL,
    system_prompt=SYSTEM,
    mcp_servers={"eom": eom},
    allowed_tools=[f"mcp__eom__{t.name}" for t in TOOLS],
    disallowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep",
                      "WebSearch", "WebFetch", "NotebookEdit"],
    can_use_tool=_only_eom,
    setting_sources=[],
)


async def run_agent(prompt: str):
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    token = _emit_queue.set(q)
    inspected_token = _inspected_var.set(set())

    async def drive() -> None:
        try:
            async def prompts():
                yield {"type": "user", "message": {"role": "user", "content": prompt}}

            async for msg in query(prompt=prompts(), options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            await q.put({"lane": "trace", "type": "thought", "text": block.text})
                        elif isinstance(block, ToolUseBlock):
                            await q.put({"lane": "trace", "type": "tool_call", "name": block.name, "args": block.input})
                elif isinstance(msg, ResultMessage):
                    await q.put({"lane": "trace", "type": "done", "is_error": msg.is_error})
        except Exception as exc:  # noqa: BLE001
            await q.put({"lane": "trace", "type": "error", "text": str(exc)})
            await q.put({"lane": "trace", "type": "done", "is_error": True})

    task = asyncio.create_task(drive())
    trace_seq = 0
    try:
        while True:
            ev = await q.get()
            # Stamp a per-run monotonic id on every trace event so the report's `src` can be
            # linked back to the tool call/result that produced a finding (U4 provenance).
            if ev.get("lane") == "trace":
                trace_seq += 1
                ev["id"] = trace_seq
            yield ev
            if ev.get("type") == "done":
                break
        await task
    finally:
        _emit_queue.reset(token)
        _inspected_var.reset(inspected_token)


if __name__ == "__main__":
    async def _main() -> None:
        async for event in run_agent("Find one page about solar energy and add a short report item."):
            print(json.dumps(event, ensure_ascii=False))

    asyncio.run(_main())
