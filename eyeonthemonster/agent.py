from __future__ import annotations

import asyncio
import contextvars
import json
import pickle
import re
import time
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
# claude-haiku-4-5 list price, USD per token. Used only for the LIVE running-cost estimate emitted
# per turn; the SDK's ResultMessage.total_cost_usd is the authoritative figure we reconcile to at the
# end. Cache write = 1.25x input, cache read = 0.1x input (standard Anthropic multipliers).
TOKEN_PRICE = {
    "input": 1.00 / 1_000_000,
    "output": 5.00 / 1_000_000,
    "cache_write": 1.25 / 1_000_000,
    "cache_read": 0.10 / 1_000_000,
}


def usage_cost(usage: dict[str, Any] | None) -> float:
    if not usage:
        return 0.0
    return (
        usage.get("input_tokens", 0) * TOKEN_PRICE["input"]
        + usage.get("output_tokens", 0) * TOKEN_PRICE["output"]
        + usage.get("cache_creation_input_tokens", 0) * TOKEN_PRICE["cache_write"]
        + usage.get("cache_read_input_tokens", 0) * TOKEN_PRICE["cache_read"]
    )
SYSTEM = """You are the research agent for "Eye on the Market" — 20 years (~5,100 pages) of \
Michael Cembalest's J.P. Morgan letters, indexed page-by-page. The user is usually the author \
or someone working directly with the author's corpus. Your default job is EXHAUSTIVE recall: \
for terse topical inputs, enumerate the relevant things written or shown in the corpus without \
requiring the user to write "what is everything I have ever written about..." each time. The user's \
input describes the desired OUTPUT (subject + scope + shape), not query text: derive concise topical \
query strings (subject + entities + synonyms) for search/enumerate from it — never pass the \
conversational request sentence as q, which dilutes ranking and tanks recall. Preserve the SCOPE \
(stay exhaustive over the subject); never narrow it.

OUTPUT CONTRACT
- Every claim carries a page citation: [p. N] for one page, [pp. N-M] for a span.
- Commit each finding the moment you confirm it, via add_to_report; the report IS the
  accumulation of those calls (do NOT hoard findings for a single end dump). On the topical fast
  path, enumerate commits the quote/chart findings for you; after enumerate, do NOT duplicate those
  findings with add_to_report. Kinds:
    heading   {text}                                          open a section / sub-topic
    quote     {text, page, issue_date, title, relevance, src} a textual finding (verbatim or tight)
    chart     {page, caption, relevance, src}                 a relevant chart/figure (image is rendered)
    narrative {text}                                          connective synthesis across findings
    answer    {text}                                          ONE final coverage map (via finish_report)
  - relevance (quote/chart): "primary" if the finding DIRECTLY answers the question, "supporting"
    for context/corroboration. Default is "supporting" — reserve "primary" for the strongest,
    most on-point evidence so the reader's eye lands there first.
  - src = a short note on what surfaced it (e.g. "search_topic:solar-pv-costs" or
    "search_semantic:'LCOE'") so the trace and report stay linked.
- QUOTE FIDELITY (hard rule): any text you wrap in quotation marks in a quote finding MUST be the
  author's actual words, copied verbatim from that page's content_text — NEVER from card_text. Each
  page has two text fields: content_text is what the author wrote; card_text is an AI vision summary
  of the charts. card_text paraphrases ("US natural gas demand will remain highly resilient,
  declining by only 13% by 2035") are NOT the author's words. Quoting them in quotation marks
  fabricates author wording and will be rejected by add_to_report. If you want to convey a chart's
  content, either quote the verbatim sentence from content_text, or write it as your own paraphrase
  with NO quotation marks. Chart captions are AI-generated chart descriptions, not quotes.
- FINISH by calling finish_report exactly once with a compact author-facing coverage map, not a
  generic essay. Keep it tight: 4-6 bullets or short paragraphs, focused on coverage areas,
  chronology, and primary cited pages. Chronology labels must be coherent date ranges (for example,
  "2014–2026" or "2014, then 2025–2026"), never reversed or malformed ranges. This pinned block is
  the first thing the reader sees.

RETRIEVAL — use several signals; trust no single one
1. Decompose the question into sub-aspects and likely sub-topics.
2. NAMED PEOPLE / ENTITIES: exact mentions are the recall backbone. Before relying on topics,
   call find_mentions with the user's raw query plus exact phrase/distinctive variants. For
   "Hillary Clinton", use terms such as "Hillary Clinton" and "Hillary"; avoid surname-only
   variants and generic role descriptors ("Secretary of State", "candidate", "president") when
   they are ambiguous unless the user asked for that broader role/surname. If find_mentions returns
   pages, commit them all and use keyword/semantic search only to catch aliases or indirect
   references; do not cite widening hits unless you verify and commit them as directly relevant. Do
   not conclude an entity is absent without this exact mention check.
3. DISCOVER topics: call list_topics for the high-level buckets (page counts + a few sample
   granular ids each), then list_topics(high_level=<bucket id>) to expand the full granular id
   list under a relevant bucket. Pick the relevant GRANULAR topic ids — those are what
   search_topic takes. Buckets are keyword-routed and mostly coherent, but "other" is a broad
   catch-all and any single label can mislead, so route on granular topics and corroborate.
4. ENUMERATE: once you've chosen relevant granular topic ids, call enumerate with all of them
   together and pass the user's original query as q. It unions the topics (that union is your
   coverage_total), surfaces the on-point pages in one call, and returns a compact primary-page
   summary plus the coverage accounting for your final map. Use search_topic only to preview a
   topic's size before enumerating.
5. WIDEN: run search_semantic AND search_keyword with several reformulations — vocabulary drifts
   across 20 years ("solar cost" 2005 vs "module ASP / LCOE" 2025). Tagging misses things; search
   catches stragglers. Use chart_only=true when the question is about what was *shown*.
6. READ context with get_page / get_pages / get_issue, and follow cross-references
   ("as discussed in last year's energy paper").

FAST PATH FOR TOPICAL / QUANTITATIVE QUERIES
- A short topic phrase is usually a topical recall query. For these, do not prove coverage by
  reading every issue. The bulk enumerate call is the coverage backbone over the chosen topics.
- Choose the GRANULAR topics that match the user's specific words and likely synonyms, not a broad
  umbrella bucket when a narrower label exists. For a MULTI-ASPECT query (a subject plus a specific
  metric or qualifier — an index plus a margin measure, a sector plus its costs, a country plus a
  policy), pick topics covering each aspect AND pass the user's full query as q. enumerate ranks the
  union by similarity to q and surfaces only the pages that are on-point for the WHOLE query,
  folding the merely-tagged remainder into one aggregate line — so you never hand-filter and a broad
  subject topic can't flood the report. If the on-point set looks too small, widen with search or
  re-run enumerate with a lower min_relevance; if it looks too loose, raise min_relevance.
- After enumerate, run at most a small widening pass (keyword + semantic reformulations). Add any
  truly missing direct findings that enumerate did not already emit, with a hard ceiling of 5
  manual add_to_report calls after enumerate. If there are more possible widening hits, summarize
  them in finish_report as "widening surfaced additional candidates" rather than committing them
  one by one. Do not loop over add_to_report for already-emitted enumerate findings. Do not loop
  over get_page for pages already summarized in enumerate.primary_pages. Do not loop over get_issue,
  get_pages, mark_inspected, or coverage_status on the topical fast path unless you are resolving a
  specific ambiguity.

EXHAUSTIVENESS — coverage is tracked by tools, never by memory
- The system compacts context, so DO NOT rely on remembering what you checked. The coverage
  tools are the only source of truth.
- For topical fast-path runs, enumerate's coverage_total (the full tagged union) is the coverage
  substrate; exhaustive means exhaustive over chosen topic ids plus a small explicit widening
  search, not all issues in the corpus. Reflect both numbers in your map (e.g. "N pages on-point of
  M tagged").
- Decide your scope (a topic_id and/or a date_range).
- As you examine an issue's pages, call mark_inspected with that issue's id.
- Periodically call coverage_status (scoped by the same topic_id/date_range) to see what remains.
  Keep working until remaining is empty — or until the only remaining issues are, on inspection,
  genuinely irrelevant (say so in a narrative note). This full issue-coverage loop is for analytical
  questions, not the topical fast path.

ANALYTICAL QUESTIONS ("what did I get wrong", "what open questions did I pose")
Not findable by similarity alone. Decompose: for "got wrong," gather the author's predictions on a
topic, then find later issues whose events contradict them, and cite both sides. For "open
questions," scan for explicit question framings and unresolved threads.

STYLE
- Group findings by sub-topic and/or chronology. Prefer verbatim quotes and real charts over paraphrase.
- When a chart matters, surface it (kind=chart) with the actual numbers in the caption.
- Every line the reader sees carries a citation. Stop when coverage_status is exhausted for your
  scope on analytical runs, or when enumerate + widening are complete on topical runs, every
  confirmed aspect is in the report, and you have called finish_report with the coverage map.
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
_enumerated_var: contextvars.ContextVar[bool] = contextvars.ContextVar("enumerated", default=False)
_post_enumerate_adds_var: contextvars.ContextVar[int] = contextvars.ContextVar("post_enumerate_adds", default=0)
MAX_SEARCH_LIMIT = 10
MAX_LIST_LIMIT = 10
MAX_TOPIC_CHILDREN = 5
MAX_GRANULAR_DRILL = 100  # granular are sorted by page count, so this is the top-100 by coverage
MAX_TEXT_CHARS = 400
MAX_ENUMERATE = 500  # safety ceiling on a single bulk enumeration (vs. the 10-cap on previews)
DEFAULT_PRIMARY_K = 6  # top-N by query similarity marked "primary"; the rest "supporting"
# Semantic relevance floor: the single knob that replaced the per-query regex filters. enumerate
# unions pages by TOPIC MEMBERSHIP, but for a multi-aspect query that union over-recalls — a page
# tagged "S&P 500" need not be about "profit margins". So we rank the union by cosine similarity to
# the user's FULL query and surface as individual findings only the pages clearing this floor; the
# rest stay in the coverage count and are reported as one aggregate line (never silently dropped).
# Because it is absolute (not relative-to-top), a loose compound union gets trimmed hard while a
# tight on-topic union is barely touched — exactly the compound-vs-broad signal, with no per-query
# logic. It is calibrated to MODEL_NAME's similarity scale (static-retrieval-mrl-en-v1: on-topic
# pages score ~0.4–0.75, off-topic ~0.1–0.3); re-check it if MODEL_NAME changes or against the
# golden key. Overridable per call via enumerate(min_relevance=...).
RELEVANCE_FLOOR = 0.38


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


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
    # topic_alias: normalized id/label -> canonical topic id. Models frequently pass the human label
    # ("us-china trade") or a space/hyphen variant instead of the exact id ("us-china-trade"), which
    # otherwise resolves to zero tagged pages and silently guts enumerate's recall. Normalizing both
    # the id and the label to a hyphen/lowercase key lets either form resolve.
    topic_alias: dict[str, str] = {}

    def _reg_alias(canonical: str, *forms: object) -> None:
        for form in (canonical, *forms):
            if form:
                topic_alias.setdefault(_norm_topic(form), canonical)

    for bucket in topics.get("high_level", []):
        if bucket.get("id"):
            label_by_topic[bucket["id"]] = bucket.get("label") or bucket["id"]
            _reg_alias(bucket["id"], bucket.get("label"))
        for gran in bucket.get("granular", []):
            if gran.get("id"):
                label_by_topic[gran["id"]] = gran.get("label") or gran["id"]
                _reg_alias(gran["id"], gran.get("label"))
    # Any topic that pages are actually tagged with, even if absent from the taxonomy file.
    for tid in pages_by_topic:
        topic_alias.setdefault(_norm_topic(tid), tid)
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
        "topic_alias": topic_alias,
    }
    return _load.state  # type: ignore[attr-defined]


async def emit(ev: dict[str, Any]) -> None:
    queue = _emit_queue.get()
    if queue is not None:
        ev.setdefault("_ts", time.perf_counter())  # producer-side stamp for latency tracking
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


# Conversational scaffolding that states the desired OUTPUT, not the search subject. Stripped from
# the ranking query (q) as a backstop: the static embedding model weights these filler words and a
# request sentence ("give me a timeline of my views on china") then ranks far below the bare subject
# ("china"), silently gutting recall. The agent is also told (system prompt) to pass concise queries;
# this just keeps a slip from tanking the whole run. Conservative + never returns empty.
_QUERY_SCAFFOLD = re.compile(
    r"\b(?:give|show|tell|find|get)\s+me\b"
    r"|\bi(?:'d| would)?\s+(?:want|like)(?:\s+to\s+(?:see|know))?\b"
    r"|\bcan\s+you\b|\bplease\b|\bi\s+have\s+ever\b|\bi've\s+ever\b"
    r"|\b(?:a|the)\s+(?:timeline|summary|history|overview|rundown|breakdown|recap|list)\s+of\b"
    r"|\bwhat(?:'s| is| are| did| do| have| was| were)\s+(?:i|you|we|they)?\b"
    r"|\beverything\s+(?:about\s+)?\b"
    r"|\bhow\s+(?:my|your|our)\b"
    r"|\bmy\s+(?:views?|thoughts?|opinions?|takes?|perspectives?)\s+(?:on|about|regarding|of)\b"
    r"|\bviews?\s+(?:on|about|regarding)\b"
    r"|\b(?:evolved?|changed|developed|shifted)(?:\s+over\s+time)?\b",
    re.I,
)


def distill_query(text: str) -> str:
    s = _WS.sub(" ", str(text or "")).strip()
    stripped = _WS.sub(" ", _QUERY_SCAFFOLD.sub(" ", s)).strip(" ?,.;:")
    return stripped or s  # never empty — fall back to the original if scaffolding was the whole string


def merged_text(page: dict) -> str:
    return "\n".join(str(page.get(k) or "") for k in ("content_text", "card_text"))


# Anything inside straight or curly double-quotes, length-gated so we don't police 2-word
# scare-quotes ("transition") but do catch sentence-length passages presented as the author's words.
_QUOTED_SPAN_RE = re.compile(r'[\"“”„‟]([^\"“”„‟]{16,})[\"“”„‟]')


def _normalize_for_match(text: str) -> str:
    # OCR/whitespace/punctuation-insensitive: collapse everything non-alphanumeric to single spaces.
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def unverified_quoted_spans(text: str, page: dict | None) -> list[str]:
    """Quoted spans in `text` that do NOT appear verbatim in the page's content_text. content_text is
    the author's actual words; card_text is the Gemini vision summary. A quote attributed to the
    author must trace to content_text — quoting the vision paraphrase and presenting it in quotation
    marks fabricates author wording. Returns the offending spans (empty == clean)."""
    content = _normalize_for_match((page or {}).get("content_text") or "")
    bad: list[str] = []
    for m in _QUOTED_SPAN_RE.finditer(text or ""):
        span = _normalize_for_match(m.group(1))
        if len(span) < 16 or span in content:
            continue
        bad.append(m.group(1).strip())
    return bad


def exact_terms_from_query(q: str) -> list[str]:
    q = _WS.sub(" ", (q or "").strip(" ?!."))
    if not q:
        return []
    tail = q
    m = re.search(r"\b(?:about|on|regarding|re|for)\s+(.+)$", q, re.I)
    if m:
        tail = m.group(1).strip(" ?!.")
    tail = re.sub(r"^(?:the|a|an)\s+", "", tail, flags=re.I)
    tail = re.sub(r"\b(?:coverage|pages|mentions|references|writings)\b", "", tail, flags=re.I)
    tail = _WS.sub(" ", tail).strip(" ?!.")
    if not tail:
        return []
    terms = [tail]
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z.'-]*", tail) if len(w) >= 4]
    # For two-token personal names, the first name is often the distinctive corpus mention while
    # the surname alone can be ambiguous ("Clinton" may mean Bill, Hillary, or the administration).
    if len(words) == 2 and words[0].lower() not in {"what", "when", "where", "which", "that", "this"}:
        terms.append(words[0])
    return terms


def remove_ambiguous_surname_terms(terms: list[str]) -> list[str]:
    tokenized = [(term, re.findall(r"[A-Za-z][A-Za-z.'-]*", term)) for term in terms]
    full_names = [words for _, words in tokenized if len(words) in {2, 3}]
    if not full_names:
        return terms
    banned: set[str] = set()
    term_keys = {term.lower() for term in terms}
    for words in full_names:
        first = words[0].lower()
        last = words[-1].lower()
        # If the tool has both the full name and the distinctive first-name variant, surname-only
        # would usually broaden to relatives, administrations, buildings, foundations, etc.
        if first in term_keys:
            banned.add(last)
    return [term for term in terms if term.lower() not in banned]


_GENERIC_PERSON_ROLE_RE = re.compile(
    r"^(?:secretary of state|president|vice president|senator|governor|congress(?:man|woman)?|"
    r"representative|candidate|nominee|first lady|administration|campaign)$",
    re.I,
)


def remove_generic_role_terms(terms: list[str]) -> list[str]:
    tokenized = [(term, re.findall(r"[A-Za-z][A-Za-z.'-]*", term)) for term in terms]
    full_names = [words for _, words in tokenized if len(words) in {2, 3}]
    first_names = {words[0].lower() for words in full_names}
    has_distinctive_first = bool(first_names & {term.lower() for term in terms})
    if not full_names or not has_distinctive_first:
        return terms
    return [term for term in terms if not _GENERIC_PERSON_ROLE_RE.match(term.strip())]


def _term_key(term: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", term.lower()))


def terms_look_like_aliases(terms: list[str]) -> bool:
    keys = [_term_key(t) for t in terms if _term_key(t)]
    if len(keys) < 2:
        return True
    longest = max(keys, key=len)
    return all(k in longest or longest in k for k in keys)


def term_pattern(term: str) -> re.Pattern[str]:
    escaped = r"\s+".join(re.escape(part) for part in _WS.split(term.strip()) if part)
    if not escaped:
        escaped = r"$."
    if re.match(r"^\w", term) and re.search(r"\w$", term):
        escaped = rf"\b{escaped}\b"
    return re.compile(escaped, re.I)


def mention_snippet(text: str, patterns: list[re.Pattern[str]], limit: int = 420) -> str:
    flat = _WS.sub(" ", text).strip()
    first = None
    for pat in patterns:
        m = pat.search(flat)
        if m and (first is None or m.start() < first.start()):
            first = m
    if not first:
        return snippet(flat, limit)
    half = max(60, limit // 2)
    start = max(0, first.start() - half)
    end = min(len(flat), first.end() + half)
    out = flat[start:end].strip()
    if start > 0:
        out = "..." + out
    if end < len(flat):
        out += "..."
    return out


def _norm_topic(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def clean_id_arg(value: object) -> str | None:
    """Normalize a string filter argument (topic_id / high_level) to a real id or None.

    Models routinely express "no filter" as the literal 2-char string '""' (or '', whitespace, a
    quote pair, 'null'/'none') rather than omitting the field. The tools treated any truthy string as
    a real filter, so '""' matched no topic/bucket and silently nuked the entire result set — every
    search returned 0 hits and every list_topics returned 0/0. Strip surrounding quotes/whitespace and
    map the empties to None so the tool runs unfiltered instead of empty."""
    if not isinstance(value, str):
        return None
    s = value.strip().strip("'\"").strip()
    if s.lower() in {"", "null", "none", "undefined", "n/a"}:
        return None
    return s


def resolve_topic_id(value: object) -> str | None:
    """Resolve a topic id/label/variant to the canonical tagged id, or None. Tolerates the label
    form ("us-china trade") and space/hyphen/case drift that would otherwise miss the tag entirely."""
    cleaned = clean_id_arg(value)
    if not cleaned:
        return None
    alias = _load()["topic_alias"]
    return alias.get(_norm_topic(cleaned), cleaned)


def clean_id_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        resolved = resolve_topic_id(v)
        if resolved and resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


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
    topic_id = resolve_topic_id(topic_id)
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
    "find_mentions",
    "Deterministic exact mention search over full merged page text and chart cards. Use this FIRST "
    "for named people, organizations, products, legislation, or exact phrases before topic search.",
    {"q": str, "terms": list, "date_range": list, "mode": str, "commit": bool},
)
async def find_mentions(args: dict) -> dict:
    s = _load()
    raw_terms = [str(t).strip() for t in (args.get("terms") or []) if str(t).strip()]
    if not raw_terms:
        raw_terms = exact_terms_from_query(str(args.get("q") or ""))
    # Keep order but drop case-insensitive duplicates.
    terms: list[str] = []
    seen_terms: set[str] = set()
    for term in raw_terms:
        key = term.lower()
        if key not in seen_terms:
            seen_terms.add(key)
            terms.append(term)
    terms = remove_ambiguous_surname_terms(terms)
    terms = remove_generic_role_terms(terms)
    if not terms:
        await emit({"lane": "trace", "type": "tool_result", "name": "find_mentions",
                    "summary": "0 pages matched exact mentions; no terms supplied or inferred"})
        return {"content": [{"type": "text", "text": json.dumps({
            "total": 0, "committed": 0, "terms": [], "mode": args.get("mode") or "any", "pages": [],
        })}]}
    patterns = [term_pattern(t) for t in terms]
    mode = str(args.get("mode") or "any").lower()
    require_all = mode == "all" or (len(terms) > 1 and not terms_look_like_aliases(terms))
    commit = bool(args.get("commit", True))
    date_range = args.get("date_range")
    matches: list[dict[str, Any]] = []
    for page in s["pages"]:
        if not in_date_range(page, date_range):
            continue
        text = merged_text(page)
        found = [terms[i] for i, pat in enumerate(patterns) if pat.search(text)]
        if (require_all and len(found) != len(patterns)) or (not require_all and not found):
            continue
        matches.append({
            "page": int(page["page"]),
            "issue_id": page.get("issue_id"),
            "issue_date": page.get("issue_date"),
            "title": page.get("title"),
            "matched_terms": found,
            "snippet": mention_snippet(text, patterns),
        })
    matches.sort(key=lambda p: p["page"])
    matches = matches[:MAX_ENUMERATE]
    if commit and matches:
        label = ", ".join(terms[:4]) + (" ..." if len(terms) > 4 else "")
        await emit({"lane": "report", "kind": "heading", "text": f"Exact mentions: {label}"})
        src = "find_mentions:" + ",".join(terms[:3])
        for m in matches:
            await emit({"lane": "report", "kind": "quote",
                        "page": m["page"], "issue_date": m.get("issue_date"),
                        "title": m.get("title"), "text": m["snippet"],
                        "relevance": "primary", "src": src})
    await emit({"lane": "trace", "type": "tool_result", "name": "find_mentions",
                "summary": f"{len(matches)} pages matched exact mentions" +
                           (": " + ", ".join(f"p.{m['page']}" for m in matches[:12]) if matches else "")})
    return {"content": [{"type": "text", "text": json.dumps({
        "total": len(matches),
        "committed": len(matches) if commit else 0,
        "terms": terms,
        "mode": "all" if require_all else "any",
        "pages": matches[:50],
    })}]}


@tool(
    "search_keyword",
    "Keyword search over merged page text, chart cards, and topic labels.",
    {"q": str, "date_range": list, "chart_only": bool, "limit": int, "topic_id": str},
)
async def search_keyword(args: dict) -> dict:
    s = _load()
    limit = bounded_limit(args.get("limit"))
    indices = filtered_indices(args.get("date_range"), bool(args.get("chart_only", False)), args.get("topic_id"))
    scores = np.asarray(s["bm25"].get_scores(tokenize(distill_query(args["q"]))), dtype=np.float32)
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
    q = s["model"].encode([distill_query(args["q"])], convert_to_numpy=True)[0]
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
    hid = resolve_topic_id(args.get("high_level"))
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
    topic_id = resolve_topic_id(args.get("topic_id"))
    pages = [p for p in _load()["pages_by_topic"].get(topic_id, []) if in_date_range(p, args.get("date_range"))]
    hits = [format_hit(p, 1.0) for p in pages[:MAX_LIST_LIMIT]]
    await emit({"lane": "trace", "type": "tool_result", "name": "search_topic", "summary": f"{len(pages)} tagged pages for {args['topic_id']}; returned {len(hits)}"})
    return {"content": [{"type": "text", "text": json.dumps({"total": len(pages), "sample": hits, "note": "sample only; call enumerate for the full set"})}]}


@tool(
    "enumerate",
    "RECALL BACKBONE for \"everything about X\": given one or more granular topic_ids, commit the "
    "tagged pages to the report in a SINGLE call -- no per-page get_page/add_to_report needed. Unions "
    "the topics, dedups by page, ranks by similarity to your FULL query q, and emits each on-topic "
    "page as a chart (if chart-bearing) or quote (top results -> relevance=primary, rest supporting). "
    "Pages that are tagged but score below the relevance floor for q are kept in the coverage count "
    "and summarized as ONE aggregate line, not dumped individually -- so a multi-aspect query "
    "(e.g. an index plus a margin measure, a country plus a policy) does not flood the report with "
    "every page that merely mentions one aspect. Pass the user's full question as q. Optionally set "
    "min_relevance (0-1, default ~0.38) lower to surface more or higher to surface only the closest.",
    {"topic_ids": list, "q": str, "date_range": list, "chart_only": bool, "primary_k": int, "min_relevance": float},
)
async def enumerate_topics(args: dict) -> dict:
    _enumerated_var.set(True)
    s = _load()
    topic_ids = clean_id_list(args.get("topic_ids"))
    # Distill conversational scaffolding so a request sentence ("give me a timeline of my views on
    # china") ranks like its subject ("china") instead of near-zero against every page.
    q = distill_query(args.get("q") or "")
    date_range = args.get("date_range")
    chart_only = bool(args.get("chart_only", False))
    seen: set[int] = set()
    union: list[dict] = []
    for tid in topic_ids:
        for p in s["pages_by_topic"].get(tid, []):
            n = int(p["page"])
            if n in seen or (chart_only and not p.get("is_chart_bearing")) or not in_date_range(p, date_range):
                continue
            seen.add(n)
            union.append(p)
    coverage_total = len(union)  # the full tagged set = exhaustive coverage denominator (never dropped)

    # Rank the union by similarity to the full query so the floor can separate on-topic pages from
    # ones merely tagged with a single aspect. Without q there is nothing to rank against, so we keep
    # chronological order and surface everything (the union itself is the answer).
    raw_floor = args.get("min_relevance")
    floor = float(raw_floor) if isinstance(raw_floor, (int, float)) else RELEVANCE_FLOOR
    scores: dict[int, float] = {}
    if q and union:
        rows = [s["row_by_page"][int(p["page"])] for p in union]
        qv = s["model"].encode([q], convert_to_numpy=True)[0]
        qv = qv / (np.linalg.norm(qv) or 1.0)
        sims = (s["emb"][rows] @ qv).astype(np.float32)
        order = list(np.argsort(-sims))
        union = [union[i] for i in order]
        scores = {int(union[r]["page"]): float(sims[order[r]]) for r in range(len(union))}

    primary_k = bounded_limit(args.get("primary_k"), default=DEFAULT_PRIMARY_K, maximum=MAX_ENUMERATE)
    if q and scores:
        surfaced = [p for p in union if scores[int(p["page"])] >= floor]
        # A genuinely niche query may have no page above the floor; always surface at least the top
        # primary_k so enumerate never returns an empty report for a real topic.
        if len(surfaced) < primary_k:
            surfaced = union[:primary_k]
    else:
        surfaced = union
    surfaced = surfaced[:MAX_ENUMERATE]
    surfaced_pages = {int(p["page"]) for p in surfaced}
    below = [p for p in union if int(p["page"]) not in surfaced_pages]

    labels = [s["label_by_topic"].get(t, t) for t in topic_ids]
    if labels:
        await emit({"lane": "report", "kind": "heading", "text": "; ".join(labels[:4]) + (" …" if len(labels) > 4 else "")})
    src = "enumerate:" + ",".join(topic_ids[:3])
    n_primary = 0
    primary_pages: list[dict[str, Any]] = []
    for rank, p in enumerate(surfaced):
        relevance = "primary" if rank < primary_k else "supporting"
        n_primary += relevance == "primary"
        if relevance == "primary":
            primary_pages.append({
                "page": int(p["page"]),
                "issue_date": p.get("issue_date"),
                "title": p.get("title"),
                "kind": "chart" if p.get("is_chart_bearing") else "quote",
                "snippet": snippet((p.get("card_text") or p.get("content_text") or ""), 420),
            })
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
    if below:
        lo = min(scores[int(p["page"])] for p in below)
        hi = max(scores[int(p["page"])] for p in below)
        await emit({"lane": "report", "kind": "narrative",
                    "text": f"+{len(below)} more page(s) are tagged these topics but score below the "
                            f"relevance floor ({floor:.2f}) for this query (similarity {lo:.2f}–{hi:.2f}); "
                            f"they remain in the coverage count and are not listed individually. Re-run "
                            f"enumerate with a lower min_relevance to surface them.",
                    "src": src})
    await emit({"lane": "trace", "type": "tool_result", "name": "enumerate",
                "summary": f"{coverage_total} pages tagged across {len(topic_ids)} topic(s); "
                           f"surfaced {len(surfaced)} on-point ({n_primary} primary) at floor {floor:.2f}" +
                           (f", aggregated {len(below)} below floor" if below else ""),
                # Structured accounting so eval_run can score without parsing the summary string.
                "coverage_total": coverage_total, "surfaced": len(surfaced),
                "below_floor": len(below), "primary": n_primary, "floor": round(floor, 3),
                "topics": topic_ids})
    # The report lane already holds every surfaced page. Return only the top pages plus the coverage
    # accounting so the agent can write a compact, honest coverage map without looping get_page.
    return {"content": [{"type": "text", "text": json.dumps({
        "committed": len(surfaced),
        "coverage_total": coverage_total,
        "below_floor": len(below),
        "relevance_floor": round(floor, 3),
        "primary": n_primary,
        "topics": topic_ids,
        "primary_pages": primary_pages[:20],
    })}]}


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
    if args["kind"] == "quote" and args.get("page") is not None:
        page = _load()["page_by_num"].get(int(args["page"]))
        bad = unverified_quoted_spans(str(args.get("text") or ""), page)
        if bad:
            await emit({"lane": "trace", "type": "tool_result", "name": "add_to_report",
                        "summary": f"rejected p.{args['page']} quote: quoted span not verbatim in "
                                   f"content_text ({bad[0][:60]!r})"})
            return {"content": [{"type": "text", "text": json.dumps({
                "rejected": True,
                "reason": "A quoted span you wrapped in quotation marks does not appear verbatim in "
                          "this page's content_text. Quotation marks must enclose the author's actual "
                          "words copied from content_text — never the AI chart/vision summary "
                          "(card_text) or a paraphrase. Either copy the exact words from content_text, "
                          "or drop the quotation marks and present it as paraphrase.",
                "page": int(args["page"]),
                "unverified_spans": bad,
            })}]}
    if _enumerated_var.get() and args["kind"] in {"quote", "chart"}:
        n = _post_enumerate_adds_var.get()
        if n >= 5:
            await emit({"lane": "trace", "type": "tool_result", "name": "add_to_report",
                        "summary": "skipped post-enumerate manual finding; 5-item cap reached"})
            return {"content": [{"type": "text", "text": "skipped: post-enumerate manual finding cap reached"}]}
        _post_enumerate_adds_var.set(n + 1)
    if args["kind"] in {"heading", "narrative", "answer"}:
        for key in ("page", "issue_date", "title", "caption", "relevance", "src"):
            args.pop(key, None)
    await emit({"lane": "report", **{k: v for k, v in args.items() if v is not None}})
    return {"content": [{"type": "text", "text": "added"}]}


@tool("finish_report", "Emit the final compact coverage-map answer.", {"text": str})
async def finish_report(args: dict) -> dict:
    await emit({"lane": "report", "kind": "answer", "text": args.get("text") or ""})
    return {"content": [{"type": "text", "text": "finished"}]}


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
    find_mentions,
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
    finish_report,
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


def _agent_user_message(user_query: str) -> str:
    return f"""Context for this run:
The user's input below states the DESIRED OUTPUT (scope + shape of the answer) for an Eye on the
Market authoring workflow. It is NOT query text. Do not pass the user's sentence verbatim as the q
to search_semantic, search_keyword, or enumerate: conversational scaffolding ("give me a timeline
of my views on", "what is everything I have ever written about", "how my views evolved") dilutes
the embedding and BM25 ranking and silently tanks recall (the sentence "give me a timeline of my
views on china" ranks far worse than "china"). Instead, READ the prompt for its subject and derive
the search inputs yourself:
  - the core SUBJECT/entities and their synonyms + vocabulary drift across 20 years (e.g. "china",
    "RMB / renminbi / yuan", "PBoC", "us-china trade") — these are your q strings and find_mentions
    terms;
  - the SCOPE (exhaustive over the whole subject unless the prompt narrows it) — preserve this, do
    not shrink the subject;
  - the OUTPUT SHAPE ("timeline" => organize chronologically and make sure every era is represented;
    "what I got wrong" => predictions vs later contradicting events) — this shapes the report, not
    the query strings.
Use concise topical queries (a few keywords/entities), not the request sentence.
If the prompt names a person, organization, product, law, or other specific entity, start with
find_mentions over the entity name and exact phrase/distinctive variants — NOT the full request
sentence. Do not add ambiguous surname-only variants or generic role descriptors for person-name
queries unless the prompt asks for that broader scope. Topic labels are conceptual, not an entity index.

User query, verbatim:
{user_query}

Task:
Enumerate the relevant coverage in Eye on the Market for that query. Prefer the bulk enumerate
tool for topical/quantitative coverage, widen with keyword and semantic search for vocabulary drift,
and finish with a compact coverage map rather than a generic explanatory essay. For topical queries,
do not enter the full issue-by-issue coverage loop after enumerate; use enumerate's committed pages
plus a small widening pass as the recall basis. After enumerate, do not re-add enumerate's findings
with add_to_report; only use add_to_report if a widening search finds a genuinely missing page, and
use at most 5 manual add_to_report calls after enumerate.
Do not call get_page just to summarize pages already returned in enumerate.primary_pages. Finish by
calling finish_report once; do not use add_to_report for the final answer."""


async def run_agent(prompt: str):
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    token = _emit_queue.set(q)
    inspected_token = _inspected_var.set(set())
    enumerated_token = _enumerated_var.set(False)
    post_enumerate_adds_token = _post_enumerate_adds_var.set(0)

    run_start = time.perf_counter()
    cost_total = 0.0  # cumulative LIVE estimate from per-turn usage; reconciled to SDK total at done

    async def drive() -> None:
        nonlocal cost_total
        last_turn = time.perf_counter()
        counted_msg_ids: set[str] = set()  # dedupe repeated AssistantMessage streams by message_id
        try:
            async def prompts():
                yield {"type": "user", "message": {"role": "user", "content": _agent_user_message(prompt)}}

            async for msg in query(prompt=prompts(), options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            await q.put({"lane": "trace", "type": "thought", "text": block.text})
                        elif isinstance(block, ToolUseBlock):
                            await q.put({"lane": "trace", "type": "tool_call", "name": block.name, "args": block.input})
                    # Per-turn (LLM component) cost + latency. The SDK streams the SAME AssistantMessage
                    # multiple times (one emission per content block group), each echoing the same
                    # usage + message_id — so count/log a turn only ONCE per message_id, else cost and
                    # turn count inflate (the duplicate rows were the source of the ~2x overestimate).
                    if msg.usage and (msg.message_id is None or msg.message_id not in counted_msg_ids):
                        if msg.message_id is not None:
                            counted_msg_ids.add(msg.message_id)
                        now = time.perf_counter()
                        turn_cost = usage_cost(msg.usage)
                        cost_total += turn_cost
                        fresh_in = msg.usage.get("input_tokens", 0)
                        cache_r = msg.usage.get("cache_read_input_tokens", 0)
                        cache_w = msg.usage.get("cache_creation_input_tokens", 0)
                        await q.put({
                            "lane": "trace", "type": "usage", "name": "llm_turn", "model": msg.model,
                            "input_tokens": fresh_in,
                            "output_tokens": msg.usage.get("output_tokens", 0),
                            "cache_read_tokens": cache_r,
                            "cache_write_tokens": cache_w,
                            # Total input context processed this turn (fresh + cached), which is the
                            # meaningful size — fresh input alone is tiny once the prompt is cached.
                            "context_tokens": fresh_in + cache_r + cache_w,
                            "turn_cost": round(turn_cost, 6),
                            "cost_total": round(cost_total, 6),
                            "latency_ms": round((now - last_turn) * 1000, 1),
                        })
                        last_turn = now
                elif isinstance(msg, ResultMessage):
                    # ResultMessage.total_cost_usd is authoritative; fall back to our estimate.
                    final_cost = msg.total_cost_usd if msg.total_cost_usd is not None else cost_total
                    await q.put({
                        "lane": "trace", "type": "done", "is_error": msg.is_error,
                        "cost_total": round(final_cost, 6),
                        "duration_ms": msg.duration_ms,
                        "num_turns": msg.num_turns,
                    })
        except Exception as exc:  # noqa: BLE001
            await q.put({"lane": "trace", "type": "error", "text": str(exc)})
            await q.put({"lane": "trace", "type": "done", "is_error": True, "cost_total": round(cost_total, 6)})

    task = asyncio.create_task(drive())
    trace_seq = 0
    pending_call_ts: float | None = None
    try:
        while True:
            ev = await q.get()
            # Stamp a per-run monotonic id on every trace event so the report's `src` can be
            # linked back to the tool call/result that produced a finding (U4 provenance).
            if ev.get("lane") == "trace":
                trace_seq += 1
                ev["id"] = trace_seq
                # Per-tool latency: time from a tool_call to its tool_result (tools run sequentially).
                ev_ts = ev.get("_ts", time.perf_counter())
                if ev.get("type") == "tool_call":
                    pending_call_ts = ev_ts
                elif ev.get("type") == "tool_result" and pending_call_ts is not None:
                    ev["latency_ms"] = round((ev_ts - pending_call_ts) * 1000, 1)
                    pending_call_ts = None
            ev.pop("_ts", None)
            yield ev
            if ev.get("type") == "done":
                break
        await task
    finally:
        _emit_queue.reset(token)
        _inspected_var.reset(inspected_token)
        _enumerated_var.reset(enumerated_token)
        _post_enumerate_adds_var.reset(post_enumerate_adds_token)


if __name__ == "__main__":
    async def _main() -> None:
        async for event in run_agent("Find one page about solar energy and add a short report item."):
            print(json.dumps(event, ensure_ascii=False))

    asyncio.run(_main())
