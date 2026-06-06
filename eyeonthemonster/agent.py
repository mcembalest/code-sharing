from __future__ import annotations

import asyncio
import contextvars
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    CLIConnectionError,
    ProcessError,
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
# Selectable agent models. Keys are the short labels the UI dropdown uses; values are the API
# model ids passed to ClaudeAgentOptions. `run_agent(model=...)` accepts either the label or the
# full id (case-insensitive on the label). Default = sonnet (the post-2026-05-30 choice).
AGENT_MODELS: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}
AGENT_MODEL = AGENT_MODELS["sonnet"]


def _prices(input_per_m: float, output_per_m: float) -> dict[str, float]:
    # Standard Anthropic multipliers: cache write = 1.25x input, cache read = 0.10x input.
    return {
        "input": input_per_m / 1_000_000,
        "output": output_per_m / 1_000_000,
        "cache_write": input_per_m * 1.25 / 1_000_000,
        "cache_read": input_per_m * 0.10 / 1_000_000,
    }


# Per-model token prices used ONLY for the live per-turn cost estimate emitted to the UI; the
# SDK's ResultMessage.total_cost_usd is the authoritative figure we reconcile to on `done`.
TOKEN_PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": _prices(1.00, 5.00),
    "claude-sonnet-4-6": _prices(3.00, 15.00),
    "claude-opus-4-8": _prices(15.00, 75.00),
}
DEFAULT_PRICES = TOKEN_PRICES[AGENT_MODEL]


def resolve_model(value: str | None) -> str:
    # Accept either a short label ("haiku"/"sonnet"/"opus") or a full API model id.
    if not value:
        return AGENT_MODEL
    s = str(value).strip()
    if not s:
        return AGENT_MODEL
    return AGENT_MODELS.get(s.lower(), s)


def authoritative_tokens(model_usage: dict[str, Any] | None) -> int:
    # Sum the real cumulative token counts from ResultMessage.model_usage (per-model breakdown).
    # Used at `done` to reconcile the live token tally, whose per-turn output was a stub (~1).
    if not model_usage:
        return 0
    fields = ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens")
    return sum(int(u.get(f, 0) or 0) for u in model_usage.values() for f in fields)


def usage_cost(usage: dict[str, Any] | None, model: str | None = None) -> float:
    if not usage:
        return 0.0
    prices = TOKEN_PRICES.get(model or "", DEFAULT_PRICES)
    return (
        usage.get("input_tokens", 0) * prices["input"]
        + usage.get("output_tokens", 0) * prices["output"]
        + usage.get("cache_creation_input_tokens", 0) * prices["cache_write"]
        + usage.get("cache_read_input_tokens", 0) * prices["cache_read"]
    )
SYSTEM = """You are the research agent for "Eye on the Market" — 20 years (~5,100 pages) of \
Michael Cembalest's J.P. Morgan letters, indexed page-by-page. The user is usually the author. \
The user's input describes the desired OUTPUT (subject + scope + shape), not query text. Derive \
concise topical query strings (subject + entities + synonyms) — never pass the conversational \
request sentence as q; it dilutes ranking and tanks recall.

QUERY SHAPE — classify BEFORE retrieval, then match effort to shape
- NARROW / FACTUAL ("did I mention X in 2007?", "earliest mention of Y", "highest single-issue
  page count"): one targeted lookup. Often answerable from find_mentions + a couple of `get`
  calls. When the question names a year or period, pass it as date_range to find_mentions/search
  so the result actually answers the temporal scope — don't return all-time hits for a "in 2007"
  question. Produce a short direct prose answer with [p. N] citations. Do NOT open with a
  "Coverage map:" header and do NOT enumerate broad topics — that's over-engineering here.
- TOPICAL / EXHAUSTIVE ("everything I wrote about X", "my view on X over time", "show me every
  chart of X"): coverage-map mode. Use enumerate as the backbone, group by sub-topic or
  chronology, emit headings, finish with a synthesizing answer block.
- ANALYTICAL ("what did I get wrong", "what open questions did I pose"): see ANALYTICAL QUESTIONS
  below. Synthesis-heavy, not findable by similarity alone.
When the shape is ambiguous, default to TOPICAL — coverage maps degrade more gracefully than
narrow answers do. The cost of over-investigating a narrow question is real (latency + tokens) —
don't keep widening just for completeness when the answer is already in hand.

TOOL SURFACE (7 tools)
- find_mentions(terms, [date_range], [mode]): exact regex over page text + chart cards. Default
  UNION across terms — a page matches if any term hits. Pass mode="all" only when you truly need
  every term on the same page. AUTO-COMMITS matched pages as quote findings. Use FIRST for named
  people, organizations, products, legislation, exact phrases.
- search(q, [date_range], [chart_only], [topic_id], [limit]): hybrid BM25 + semantic (RRF fusion).
  Returns ranked hits but does NOT commit. Use for widening / vocabulary-drift passes.
- list_topics([high_level]): no args → high-level buckets. high_level=<bucket id> → granular topic
  ids inside that bucket (those are what `enumerate` takes).
- enumerate(topic_ids, q, [date_range], [chart_only], [min_relevance], [must_contain]): union the
  tagged pages of the given granular topics, rank by similarity to q, and AUTO-COMMIT the on-point
  pages. The coverage backbone for topical recall. Pass must_contain=[entity terms] when the query
  is about a specific named entity but you want broad topical context — see ENTITY GUARDRAIL.
- get({page} | {start,end} | {issue_id}): fetch full merged page records.
- ask_user(question, why): pause the run with ONE terse clarifying question. High bar — only call
  when concrete evidence (numbers, not vibes) says clarification would materially change retrieval.
  See CLARIFICATION below. After calling, STOP.
- add_report_items(items): manual commits. Use this for synthesis sections, curated analytical
  findings, isolated late additions, and the final answer. Item kinds: heading {text}; narrative
  {text}; answer {text} (the final coverage map — emit exactly ONCE at the very end); quote
  {text, page, issue_date, title, relevance, src}; chart {page, caption, relevance, src}. Put
  representative inline [p. N] / [pp. N-M] citations in the answer item; do not keep searching just
  to add more citations once the evidence is in hand.

CLARIFICATION — use sparingly
Default is to proceed without asking. The user dislikes pointless back-and-forth. Only call
`ask_user` when there is concrete evidence that clarifying would change WHAT you retrieve, not
just polish the framing. Real triggers:
- An entity name maps to multiple distinct corpus subjects and committing to one would require
  redoing the run (e.g. "Clinton" → Bill / Hillary / the administration; "Powell" → Fed chair /
  Colin Powell).
- A topic union shows >5x over-recall vs the user's phrasing implies, AND the user's words point
  to a plausible narrower scope (e.g. user asked about "China economy", topic union spans 14
  sub-aspects across 800 pages — ask which sub-aspect).
- The user's words have two materially different corpus interpretations and you cannot pick by
  surface evidence (e.g. "my views on AI" could mean the technology vs the equity bubble).
Bad triggers (do NOT ask): generic ambiguity, scope feels broad, "just to be sure". When in
doubt, proceed and produce the broader recall; the user can re-run narrower.
After calling ask_user, STOP — no further tool calls, no answer. The run ends; the user refines
and re-runs.

OUTPUT CONTRACT
- Every claim carries a page citation: [p. N] for one page, [pp. N-M] for a span. These are not
  decoration — the UI turns every [p. N] into a clickable link that opens the source page, and the
  reader relies on them to verify and explore. A line the reader sees without a [p. N] is a dead end
  and counts as a defect. Treat "did I attach a page link?" as the last check on every line you emit,
  ESPECIALLY each bullet of the final answer block.
- The report IS the accumulation of report-lane events. find_mentions and enumerate auto-commit
  their findings — do NOT use add_report_items to re-add them.
- Manual synthesis must be batched: call add_report_items once with the ordered item list rather
  than spending one tool turn per heading/quote/narrative.
- relevance (quote/chart): "primary" for findings that DIRECTLY answer the question, "supporting"
  for context. Default supporting; reserve primary for the strongest evidence.
- src = a short note on what surfaced it (e.g. "search:'LCOE'" or "find_mentions:china") so the
  trace and report stay linked.
- QUOTE FIDELITY (hard rule): any text you wrap in quotation marks in a quote finding MUST be the
  author's actual words, copied verbatim from that page's content_text — NEVER from card_text.
  content_text is what the author wrote; card_text is an AI vision summary of the charts. Quoting
  card_text paraphrases in quotation marks fabricates author wording and will be rejected. Either
  copy verbatim from content_text, or drop the quotes and present as paraphrase.
- ANSWER BLOCK depends on shape. TOPICAL and ANALYTICAL queries: FINISH by calling add_report_items
  with exactly one answer item containing a compact author-facing coverage map (4–6 bullets / short
  paragraphs; coverage areas, chronology, primary cited pages). EVERY bullet/line of the answer
  MUST carry at least one inline [p. N] / [pp. N-M] link to its strongest source — a linkless answer
  bullet is the single most common defect and is not acceptable; pull the page number from a finding
  you already committed rather than running extra searches just to be citation-dense. Chronology
  labels must be
  coherent date ranges ("2014–2026", never reversed). NARROW / FACTUAL queries: ALWAYS finish
  with a one- or two-sentence answer block when the question is yes/no, temporal ("in 2007"),
  superlative ("earliest", "highest"), or negative — the reader needs the direct verdict, not a
  pile of quotes to infer it from. Lead with the verdict and cite ("Yes — first in 2009 [p. N],
  then [p. M] …"; "No — no iPhone mention in 2007; the earliest is [p. N], 2009."). Skip the
  answer block only for a pure single-fact lookup where the one committed finding already IS the
  answer. Don't pad with "Coverage map:" scaffolding.

RETRIEVAL PLAYBOOK
1. Decompose the question into sub-aspects and likely sub-topics.
2. NAMED PEOPLE / ENTITIES: exact mentions are the recall backbone. Call find_mentions with the
   exact phrase plus distinctive alias variants (for "Hillary Clinton", terms=["Hillary Clinton",
   "Hillary"]; avoid bare surnames and generic role descriptors unless the user asked for that
   broader scope). Remember: terms are UNIONed by default. Don't conclude an entity is absent
   without this check.
3. TOPICAL queries: list_topics to discover buckets, list_topics(high_level=<bucket>) to expand
   granular ids, then enumerate the relevant granular topic_ids together with the user's full
   query as q. For multi-aspect queries (subject + qualifier — an index + a margin measure, a
   country + a policy), pick topics covering each aspect; enumerate ranks the union by similarity
   to q and surfaces only on-point pages, folding merely-tagged remainder into one aggregate line.

ENTITY GUARDRAIL on enumerate
When the query is about a specific named entity (a country, person, company, product) and you
enumerate BROADER topics for context (e.g. "Argentina charts" -> sovereign-debt + currency-pegs +
emerging-market-vulnerabilities), ALWAYS pass must_contain=[entity terms] to enumerate. Embedding
similarity is conceptual, not lexical — without must_contain, a Greek-default chart will rank high
against "Argentina peso default" and get committed. Use partial stems so morphology is covered
(must_contain=["Argentin"] catches both Argentina and Argentine; must_contain=["Putin","Russia",
"Kremlin"] for a Putin query). Find_mentions does NOT need this — it already filters lexically.
4. WIDEN with `search` using several reformulations — vocabulary drifts across 20 years ("solar
   cost" 2005 vs "module ASP / LCOE" 2025). Use chart_only=true when the question is about what
   was *shown*. Add genuinely missing widening hits via add_report_items (small handful at most — if
   widening turns up many candidates, summarize them in the final answer rather than dumping each).
5. READ context with `get` (single page, range, or whole issue) only when you need to verify a
   cross-reference or pull more text — don't loop `get` over pages already summarized by enumerate.

ANALYTICAL QUESTIONS ("what did I get wrong", "what open questions did I pose")
Not findable by similarity alone. Decompose: for "got wrong," gather the author's predictions on a
topic, then find later issues whose events contradict them, and cite both sides. For "open
questions," scan for explicit question framings and unresolved threads. Once the evidence is in
hand, commit the curated analytical synthesis with one add_report_items call (headings,
narratives, selected quotes/charts, final answer). Do not spend a separate tool turn per finding.

STYLE
- Group findings by sub-topic and/or chronology. Prefer verbatim quotes and real charts over paraphrase.
- Quote excerpts should be wide and readable — a self-contained passage of 2–4 sentences from
  content_text (NOT a one-line snippet or a header fragment; a snippet is a search artifact, the
  reader wants enough surrounding context to understand the point), with the specific phrase that
  answers the query wrapped in markdown **bold**. find_mentions and enumerate already format their
  auto-committed excerpts this way; when you compose your own quote item for add_report_items,
  follow the same shape. If a committed excerpt is too thin to stand on its own, `get` that page
  and widen it — use the chunk hit as a locator, then quote the readable passage around it.
- When a chart matters, surface it (kind=chart) with the actual numbers in the caption.
- Every line the reader sees carries a citation. Stop when you have enough evidence to answer
  the question at the chosen shape — do not keep widening just for completeness. TOPICAL queries
  stop when find_mentions / enumerate / widening are complete and a coverage-map answer block has
  been emitted. NARROW queries stop as soon as the answer is in hand (sometimes that's a single
  find_mentions + a direct prose answer).
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
MAX_SEARCH_LIMIT = 10
MAX_LIST_LIMIT = 10
MAX_TOPIC_CHILDREN = 5
MAX_GRANULAR_DRILL = 100  # granular are sorted by page count, so this is the top-100 by coverage
MAX_TEXT_CHARS = 400
MAX_ENUMERATE = 500  # safety ceiling on a single bulk enumeration (vs. the 10-cap on previews)
DEFAULT_PRIMARY_K = 6  # top-N by query similarity marked "primary"; the rest "supporting"
# find_mentions precision-by-volume. find_mentions does NOT rank, so it can't justify "primary" the
# way enumerate (similarity-ranked) can. A distinctive entity ("iPhone", "Volcker rule") returns few
# hits and those ARE the answer -> primary, shown in full. A common phrase ("watch for", "we don't
# know", "remains to be seen") over-matches and would flood the report on a phrase-heavy analytical
# query -> mark the whole batch "supporting" so the U3 UI tucks it behind the per-section disclosure
# and the agent's curated findings + answer block stay the visible report. Threshold is per call.
FIND_MENTIONS_PRIMARY_MAX = 12
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


# Sentence splitter: break on terminal punctuation followed by whitespace + a sentence-start
# character. Tolerant enough for chart-rich finance prose (skips decimals like "3.5%" because they
# aren't followed by whitespace+capital). Not perfect — but the excerpt is bounded to ±1 sentence
# from an anchor, so over- or under-splitting at most widens or narrows by one fragment.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(\[])")
# Excerpt width: snippets are a search-granularity artifact; what the reader wants is a readable
# passage. Default to a wide, multi-sentence window centered on the query-relevant anchor, with the
# relevant clause bolded. The agent can still `get` a whole page when it needs more.
EXCERPT_MAX_CHARS = 950
EXCERPT_NEIGHBORS = 2  # sentences on each side of the anchor


def _split_sentences(text: str) -> list[str]:
    flat = _WS.sub(" ", text).strip()
    if not flat:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(flat) if s.strip()]


def _bold_matches(text: str, patterns: list[re.Pattern[str]]) -> str:
    # Escape any existing asterisks so they don't collide with the bold markers we're about to add.
    safe = text.replace("*", r"\*")
    # Collect non-overlapping match spans across all patterns (earliest wins on overlap).
    spans: list[tuple[int, int]] = []
    for pat in patterns:
        for m in pat.finditer(safe):
            spans.append((m.start(), m.end()))
    if not spans:
        return safe
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    out = []
    cursor = 0
    for s, e in merged:
        out.append(safe[cursor:s])
        out.append("**" + safe[s:e] + "**")
        cursor = e
    out.append(safe[cursor:])
    return "".join(out)


def _assemble_excerpt(sentences: list[str], anchor: int, max_chars: int) -> tuple[str, int, int]:
    # Greedy ±EXCERPT_NEIGHBORS expansion, then trim symmetrically to stay under max_chars while
    # keeping the anchor sentence whole. Returns (text, lo, hi) where lo/hi are sentence indices.
    lo = max(0, anchor - EXCERPT_NEIGHBORS)
    hi = min(len(sentences), anchor + EXCERPT_NEIGHBORS + 1)
    while lo < hi:
        joined = " ".join(sentences[lo:hi])
        if len(joined) <= max_chars or (lo == anchor and hi == anchor + 1):
            return joined, lo, hi
        # Drop whichever neighbor is farther from the anchor; ties drop the trailing one.
        if (anchor - lo) > (hi - 1 - anchor):
            lo += 1
        else:
            hi -= 1
    return sentences[anchor], anchor, anchor + 1


def _wrap_ellipses(text: str, lo: int, total: int, hi: int) -> str:
    out = text
    if lo > 0:
        out = "…" + out
    if hi < total:
        out = out + "…"
    return out


def query_excerpt(
    page: dict,
    *,
    patterns: list[re.Pattern[str]] | None = None,
    query: str | None = None,
    query_vec: "np.ndarray | None" = None,
    model=None,
    max_chars: int = EXCERPT_MAX_CHARS,
) -> str:
    """Wider, sentence-bounded excerpt with the query-relevant span bolded.

    Term mode (patterns given): pick the sentence with the most regex hits, expand ±1, bold each
    hit inside the excerpt. Semantic mode (query+query_vec+model given): pick the sentence whose
    embedding is closest to query_vec, expand ±1, bold the whole anchor sentence.

    Bolding is only applied when sourcing from content_text (the author's actual words), so the
    excerpt remains a faithful quote that passes unverified_quoted_spans. card_text fallback
    returns a plain snippet — bolded AI-summary text would read as a fabricated author quote.
    """
    content = page.get("content_text") or ""
    sentences = _split_sentences(content)
    if not sentences:
        # No author text on this page (chart-only). Fall back to the legacy snippet behavior over
        # card_text without bolding — we can't claim verbatim authorship of an AI summary.
        return snippet(page.get("card_text") or "", max_chars)

    anchor = 0
    if patterns:
        best_hits = 0
        for i, sent in enumerate(sentences):
            hits = sum(1 for pat in patterns if pat.search(sent))
            if hits > best_hits:
                best_hits = hits
                anchor = i
        if best_hits == 0:
            # No regex hit landed in content_text (term matched only in card_text). Fall back.
            return mention_snippet("\n".join([content, page.get("card_text") or ""]), patterns, max_chars)
    elif query is not None and model is not None:
        try:
            sent_vecs = model.encode(sentences, convert_to_numpy=True)
            norms = np.linalg.norm(sent_vecs, axis=1, keepdims=True)
            sent_vecs = sent_vecs / np.where(norms == 0, 1.0, norms)
            if query_vec is None:
                qv = model.encode([query], convert_to_numpy=True)[0]
                qv = qv / (np.linalg.norm(qv) or 1.0)
            else:
                qv = query_vec
            sims = sent_vecs @ qv
            anchor = int(np.argmax(sims))
        except Exception:
            anchor = 0
    # else: no signal — fall through and just take the first sentence(s).

    text, lo, hi = _assemble_excerpt(sentences, anchor, max_chars)
    if patterns:
        text = _bold_matches(text, patterns)
    elif query is not None:
        # Bold the anchor sentence inside the assembled excerpt.
        anchor_text = sentences[anchor].replace("*", r"\*")
        safe = text.replace("*", r"\*")
        idx = safe.find(anchor_text)
        if idx >= 0:
            text = safe[:idx] + "**" + anchor_text + "**" + safe[idx + len(anchor_text):]
        else:
            text = safe
    return _wrap_ellipses(text, lo, len(sentences), hi)


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
    "Exact regex search over merged page text + chart cards. Pass `terms` — the list of exact "
    "phrases/aliases to look for (e.g. [\"China\",\"Chinese\",\"US-China\"]). Default behavior is "
    "UNION: a page matches if ANY term appears. Pass mode=\"all\" to require every term on the same "
    "page (rare — only for true conjunctions like \"Powell\" AND \"yield curve\"). Auto-commits "
    "matched pages to the report as quote findings under one heading.",
    {"terms": list, "date_range": list, "mode": str},
)
async def find_mentions(args: dict) -> dict:
    s = _load()
    raw_terms = [str(t).strip() for t in (args.get("terms") or []) if str(t).strip()]
    # De-dupe case-insensitively while preserving order.
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
                    "summary": "0 pages matched: no terms supplied"})
        return {"content": [{"type": "text", "text": json.dumps({
            "total": 0, "terms": [], "mode": "any", "pages": [],
        })}]}
    mode = str(args.get("mode") or "any").lower()
    if mode not in {"any", "all"}:
        mode = "any"
    require_all = mode == "all"
    patterns = [term_pattern(t) for t in terms]
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
            "snippet": query_excerpt(page, patterns=patterns),
        })
    matches.sort(key=lambda p: p["page"])
    matches = matches[:MAX_ENUMERATE]
    if matches:
        label = ", ".join(terms[:4]) + (" ..." if len(terms) > 4 else "")
        await emit({"lane": "report", "kind": "heading", "text": f"Exact mentions: {label}"})
        src = "find_mentions:" + ",".join(terms[:3])
        # Precision-by-volume: a distinctive entity returns few hits (those are the answer -> primary);
        # a common phrase over-matches and would flood the report -> commit the batch as supporting so
        # the UI tucks it behind the disclosure. See FIND_MENTIONS_PRIMARY_MAX.
        relevance = "primary" if len(matches) <= FIND_MENTIONS_PRIMARY_MAX else "supporting"
        for m in matches:
            await emit({"lane": "report", "kind": "quote",
                        "page": m["page"], "issue_date": m.get("issue_date"),
                        "title": m.get("title"), "text": m["snippet"],
                        "relevance": relevance, "src": src})
    await emit({"lane": "trace", "type": "tool_result", "name": "find_mentions",
                "summary": f"{len(matches)} pages matched (mode={mode})" +
                           (": " + ", ".join(f"p.{m['page']}" for m in matches[:12]) if matches else "")})
    return {"content": [{"type": "text", "text": json.dumps({
        "total": len(matches),
        "terms": terms,
        "mode": mode,
        "pages": matches[:50],
    })}]}


@tool(
    "search",
    "Hybrid keyword+semantic search (RRF fusion of BM25 + embedding cosine) over merged page text, "
    "chart cards, and topic labels. Returns ranked hits — does NOT commit to the report. Use for "
    "widening passes and aliasing; for committing topic-tagged pages, use enumerate; for committing "
    "exact mentions, use find_mentions.",
    {"q": str, "date_range": list, "chart_only": bool, "limit": int, "topic_id": str},
)
async def search(args: dict) -> dict:
    s = _load()
    limit = bounded_limit(args.get("limit"))
    indices = filtered_indices(args.get("date_range"), bool(args.get("chart_only", False)), args.get("topic_id"))
    if not indices:
        await emit({"lane": "trace", "type": "tool_result", "name": "search", "summary": "0 hits (no pages in scope)"})
        return {"content": [{"type": "text", "text": json.dumps([])}]}
    q_text = distill_query(args["q"])
    # BM25 ranks: rank position 0 = best per BM25 score, restricted to the filtered scope.
    bm25_scores = np.asarray(s["bm25"].get_scores(tokenize(q_text)), dtype=np.float32)
    bm25_order = sorted(indices, key=lambda i: float(bm25_scores[i]), reverse=True)
    # Semantic ranks: same idea, cosine similarity against the query embedding.
    qv = s["model"].encode([q_text], convert_to_numpy=True)[0]
    qv = qv / (np.linalg.norm(qv) or 1.0)
    sem_scores = (s["emb"] @ qv).astype(np.float32)
    sem_order = sorted(indices, key=lambda i: float(sem_scores[i]), reverse=True)
    # Reciprocal Rank Fusion: stable, scale-free combination of two heterogeneous rankers. k=60 is
    # the canonical TREC default; the constant dampens the long tail so a page only top-100 in one
    # ranker but top-5 in the other still surfaces.
    k = 60
    rrf: dict[int, float] = {i: 0.0 for i in indices}
    for rank, i in enumerate(bm25_order):
        rrf[i] += 1.0 / (k + rank)
    for rank, i in enumerate(sem_order):
        rrf[i] += 1.0 / (k + rank)
    ordered = sorted(indices, key=lambda i: rrf[i], reverse=True)[:limit]
    hits = [format_hit(s["pages"][i], rrf[i]) for i in ordered]
    await emit({"lane": "trace", "type": "tool_result", "name": "search",
                "summary": f"{len(hits)} hits: " + ", ".join(f"p.{h['page']}" for h in hits[:8])})
    return {"content": [{"type": "text", "text": json.dumps(hits)}]}


@tool(
    "list_topics",
    "Without args: list the high-level buckets (page counts + a few sample granular ids each). "
    "Pass high_level=<bucket id> to expand the granular topic ids in that bucket — those ids are "
    "what `enumerate` takes.",
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
    "enumerate",
    "RECALL BACKBONE for \"everything about X\": given one or more granular topic_ids, commit the "
    "tagged pages to the report in a SINGLE call -- no per-page get/add_report_items needed. Unions "
    "the topics, dedups by page, ranks by similarity to your FULL query q, and emits each on-topic "
    "page as a chart (if chart-bearing) or quote (top results -> relevance=primary, rest supporting). "
    "Pages that are tagged but score below the relevance floor for q are kept in the coverage count "
    "and summarized as ONE aggregate line, not dumped individually -- so a multi-aspect query "
    "(e.g. an index plus a margin measure, a country plus a policy) does not flood the report with "
    "every page that merely mentions one aspect. Pass the user's full question as q. Optionally set "
    "min_relevance (0-1, default ~0.38) lower to surface more or higher to surface only the closest. "
    "ENTITY GUARDRAIL: when enumerating broader topics for a query about a specific named entity "
    "(country, person, company, product), pass must_contain=[entity terms] -- pages are kept only if "
    "their text contains at least one of those substrings (case-insensitive, partial match, so "
    "[\"Argentin\"] catches both \"Argentina\" and \"Argentine\"). Without this, the embedding-similarity "
    "ranking is conceptual and will surface a Greek-default chart for an Argentina query.",
    {"topic_ids": list, "q": str, "date_range": list, "chart_only": bool, "primary_k": int, "min_relevance": float, "must_contain": list},
)
async def enumerate_topics(args: dict) -> dict:
    s = _load()
    topic_ids = clean_id_list(args.get("topic_ids"))
    # Distill conversational scaffolding so a request sentence ("give me a timeline of my views on
    # china") ranks like its subject ("china") instead of near-zero against every page.
    q = distill_query(args.get("q") or "")
    date_range = args.get("date_range")
    chart_only = bool(args.get("chart_only", False))
    # Optional entity guardrail. Embedding similarity ranks by concept, so an Argentina query against
    # broad sovereign-debt topics will surface Greek/Turkish default pages too. must_contain filters
    # the union to pages whose merged text literally contains at least one of the supplied
    # substrings, case-insensitive — kept as substring (not word boundary) so "Argentin" matches both
    # "Argentina" and "Argentine".
    raw_terms = args.get("must_contain") or []
    if not isinstance(raw_terms, list):
        raw_terms = [raw_terms]
    must_terms = [str(t).lower() for t in raw_terms if isinstance(t, (str, int, float)) and str(t).strip()]
    seen: set[int] = set()
    union: list[dict] = []
    dropped_by_must: int = 0
    for tid in topic_ids:
        for p in s["pages_by_topic"].get(tid, []):
            n = int(p["page"])
            if n in seen or (chart_only and not p.get("is_chart_bearing")) or not in_date_range(p, date_range):
                continue
            if must_terms:
                text = ((p.get("content_text") or "") + " " + (p.get("card_text") or "")).lower()
                if not any(t in text for t in must_terms):
                    seen.add(n)
                    dropped_by_must += 1
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
                        "caption": snippet((p.get("card_text") or p.get("content_text") or ""), 260),
                        "relevance": relevance, "src": src})
        else:
            await emit({"lane": "report", "kind": "quote", "page": int(p["page"]),
                        "issue_date": p.get("issue_date"), "title": p.get("title"),
                        "text": query_excerpt(p, query=q, query_vec=qv if q else None, model=s["model"]) if q
                                else snippet((p.get("content_text") or p.get("card_text") or ""), MAX_TEXT_CHARS),
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
                           (f", aggregated {len(below)} below floor" if below else "") +
                           (f", dropped {dropped_by_must} by must_contain={must_terms}" if must_terms else ""),
                # Structured accounting so eval_run can score without parsing the summary string.
                "coverage_total": coverage_total, "surfaced": len(surfaced),
                "below_floor": len(below), "primary": n_primary, "floor": round(floor, 3),
                "topics": topic_ids, "must_contain": must_terms, "dropped_by_must_contain": dropped_by_must})
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


@tool(
    "get",
    "Return merged page records. Pass exactly one selector: `page` (single page), `start`+`end` "
    "(inclusive span), or `issue_id` (all pages of one issue).",
    {"page": int, "start": int, "end": int, "issue_id": str},
)
async def get(args: dict) -> dict:
    s = _load()
    issue_id = args.get("issue_id")
    if issue_id:
        issue = s["issue_by_id"].get(str(issue_id))
        pages = ([p for p in s["pages"] if int(issue["page_start"]) <= int(p["page"]) <= int(issue["page_end"])]
                 if issue else [])
        await emit({"lane": "trace", "type": "tool_result", "name": "get", "summary": f"{issue_id} -> {len(pages)} pages"})
        return {"content": [{"type": "text", "text": json.dumps({
            "issue": compact_issue(issue) if issue else None,
            "pages": [compact_page(p) for p in pages[:MAX_LIST_LIMIT]],
            "total_pages": len(pages),
        })}]}
    if args.get("start") is not None and args.get("end") is not None:
        start, end = int(args["start"]), int(args["end"])
        pages = [p for p in s["pages"] if start <= int(p["page"]) <= end]
        await emit({"lane": "trace", "type": "tool_result", "name": "get", "summary": f"p.{start}-p.{end} -> {len(pages)} pages"})
        return {"content": [{"type": "text", "text": json.dumps({
            "total": len(pages),
            "pages": [compact_page(p) for p in pages[:MAX_LIST_LIMIT]],
        })}]}
    if args.get("page") is not None:
        n = int(args["page"])
        page = s["page_by_num"].get(n)
        await emit({"lane": "trace", "type": "tool_result", "name": "get", "summary": f"p.{n}"})
        return {"content": [{"type": "text", "text": json.dumps(compact_page(page) if page else None)}]}
    await emit({"lane": "trace", "type": "tool_result", "name": "get", "summary": "no selector supplied"})
    return {"content": [{"type": "text", "text": json.dumps({"error": "pass page, start+end, or issue_id"})}]}


@tool(
    "ask_user",
    "Pause the run to ask ONE terse clarifying question. Use sparingly — only when you have "
    "concrete evidence (numbers, not vibes) that a short clarification would materially change "
    "retrieval strategy. After calling this you MUST stop: do not run further tools, do not emit "
    "an answer. The user will read the question, refine the query, and re-run. `why` should "
    "include the evidence (e.g. \"find_mentions returned 800 pages spanning 14 sub-aspects; "
    "narrowing scope would cut to ~60\").",
    {"question": str, "why": str},
)
async def ask_user(args: dict) -> dict:
    question = str(args.get("question") or "").strip()
    why = str(args.get("why") or "").strip()
    if not question:
        await emit({"lane": "trace", "type": "tool_result", "name": "ask_user",
                    "summary": "rejected: empty question"})
        return {"content": [{"type": "text", "text": "ask_user requires a non-empty question"}]}
    await emit({"lane": "ask", "question": question, "why": why})
    await emit({"lane": "trace", "type": "tool_result", "name": "ask_user",
                "summary": f"asked: {question[:80]}"})
    return {"content": [{"type": "text", "text":
        "Clarification surfaced to the user. STOP NOW: do not call further tools, do not emit "
        "an answer. The user will refine their query and re-run."}]}


@tool(
    "add_to_report",
    "Commit a single finding to the live report. Kinds: heading {text}; narrative {text}; "
    "answer {text} (the final coverage-map answer — call this exactly once at the end); "
    "quote {text, page, issue_date, title, relevance, src}; chart {page, caption, relevance, src}.",
    {"kind": str, "text": str, "page": int, "issue_date": str, "title": str, "caption": str, "relevance": str, "src": str},
)
async def add_to_report(args: dict) -> dict:
    normalized, error = normalize_report_item(args)
    if error:
        await emit({"lane": "trace", "type": "tool_result", "name": "add_to_report",
                    "summary": error["summary"]})
        return {"content": [{"type": "text", "text": json.dumps(error["payload"])}]}
    await emit({"lane": "report", **normalized})
    return {"content": [{"type": "text", "text": "added"}]}


def normalize_report_item(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    args = dict(raw)
    if args.get("kind") not in {"quote", "chart", "heading", "narrative", "answer"}:
        args["kind"] = "narrative"
    if args["kind"] in {"quote", "chart"} and args.get("relevance") not in {"primary", "supporting"}:
        args["relevance"] = "supporting"
    if args["kind"] == "quote" and args.get("page") is not None:
        page = _load()["page_by_num"].get(int(args["page"]))
        bad = unverified_quoted_spans(str(args.get("text") or ""), page)
        if bad:
            return {}, {
                "summary": f"rejected p.{args['page']} quote: quoted span not verbatim in "
                           f"content_text ({bad[0][:60]!r})",
                "payload": {
                    "rejected": True,
                    "reason": "A quoted span you wrapped in quotation marks does not appear verbatim in "
                              "this page's content_text. Quotation marks must enclose the author's actual "
                              "words copied from content_text — never the AI chart/vision summary "
                              "(card_text) or a paraphrase. Either copy the exact words from content_text, "
                              "or drop the quotation marks and present it as paraphrase.",
                    "page": int(args["page"]),
                    "unverified_spans": bad,
                },
            }
    if args["kind"] in {"heading", "narrative", "answer"}:
        for key in ("page", "issue_date", "title", "caption", "relevance", "src"):
            args.pop(key, None)
    return {k: v for k, v in args.items() if v is not None}, None


@tool(
    "add_report_items",
    "Commit one or more report items in a single tool call. Use this for all manual report commits, "
    "especially synthesis sections or several curated findings. Each item has kind in "
    "heading/narrative/answer/quote/chart; quote/chart may include "
    "page, issue_date, title, relevance, src; chart uses caption; quote/narrative/answer/heading use text.",
    {"items": list},
)
async def add_report_items(args: dict) -> dict:
    raw_items = args.get("items") or []
    if not isinstance(raw_items, list):
        raw_items = []
    added = 0
    rejected: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            rejected.append({"index": i, "reason": "item is not an object"})
            continue
        normalized, error = normalize_report_item(raw)
        if error:
            rejected.append({"index": i, **error["payload"]})
            continue
        await emit({"lane": "report", **normalized})
        added += 1
    summary = f"added {added} report item(s)" + (f", rejected {len(rejected)}" if rejected else "")
    await emit({"lane": "trace", "type": "tool_result", "name": "add_report_items", "summary": summary})
    return {"content": [{"type": "text", "text": json.dumps({"added": added, "rejected": rejected})}]}


TOOLS = [
    find_mentions,
    search,
    list_topics,
    enumerate_topics,
    get,
    ask_user,
    add_report_items,
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


def build_options(model: str | None = None, resume: str | None = None) -> ClaudeAgentOptions:
    # ClaudeAgentOptions is per-call so the UI can select the model at run time. Everything else
    # (system prompt, tool surface, lockdown) is invariant across model choice.
    # `resume` is set only on a post-failure retry: it continues an existing CLI session (the
    # model keeps the tool results from the turns that already succeeded) instead of starting over.
    return ClaudeAgentOptions(
        model=resolve_model(model),
        resume=resume,
        system_prompt=SYSTEM,
        mcp_servers={"eom": eom},
        allowed_tools=[f"mcp__eom__{t.name}" for t in TOOLS],
        # can_use_tool below is the real lockdown (it denies anything not mcp__eom__*). This list
        # is the belt-and-suspenders: it removes built-ins from the OFFERED tool set so the model
        # doesn't waste a turn calling one only to be denied. Keep it current with the CLI — 2.x
        # added ToolSearch (observed being called + denied), plus Task/TodoWrite/plan/shell built-ins.
        disallowed_tools=["Bash", "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep",
                          "WebSearch", "WebFetch", "NotebookEdit", "ToolSearch", "Task",
                          "TodoWrite", "ExitPlanMode", "BashOutput", "KillShell", "SlashCommand"],
        can_use_tool=_only_eom,
        setting_sources=[],
    )


options = build_options()  # default-model singleton kept for the __main__ smoke test


def _agent_user_message(user_query: str) -> str:
    return f"""Context for this run:
The user's input below states the DESIRED OUTPUT (scope + shape) for an Eye on the Market
authoring workflow. It is NOT query text. Do not pass the user's sentence verbatim as q to
`search` or `enumerate`: conversational scaffolding ("give me a timeline of my views on", "how my
views evolved") dilutes BM25 + embedding ranking and tanks recall ("give me a timeline of my views
on china" ranks far worse than "china"). Read the prompt for its subject and derive:
  - SUBJECT/entities + 20-year vocabulary drift (e.g. "china", "RMB / renminbi / yuan", "PBoC",
    "us-china trade") — these are your q strings and find_mentions terms.
  - SCOPE (exhaustive over the subject unless the prompt narrows it) — preserve it.
  - OUTPUT SHAPE ("timeline" => chronological; "what I got wrong" => predictions vs later
    contradictions) — this shapes the report, not the query strings.

If the prompt names a person, organization, product, law, or other entity, start with
find_mentions over the entity + distinctive alias variants (NOT the full request sentence). Skip
ambiguous surname-only variants and generic role descriptors unless the user asks for that scope.
Pass aliases as separate items in `terms` — find_mentions UNIONs them by default.

User query, verbatim:
{user_query}

Task:
Enumerate the relevant coverage. Use enumerate as the topical recall backbone; widen with `search`
for vocabulary drift; commit narrative connective tissue and any genuinely missing widening hits.
Use add_report_items for all manual report commits, batching the ordered findings and final answer
instead of spending one tool turn per item. Don't re-add enumerate's auto-committed findings. Don't
`get` pages already summarized in enumerate.primary_pages unless you need to verify a quote or
follow a cross-ref. Finish with exactly one answer item."""


# Transient = the upstream API/connection dropped (vs. a real model/tool error). These surface
# both as raised CLIConnectionError/ProcessError (CLI exited non-zero printing "API Error: ...")
# and as a clean ResultMessage(is_error) carrying the same text / an api_error_status. The canonical
# one we hit is the Node CLI's "socket connection was closed unexpectedly"; overloaded_error (529)
# and the 5xx family are the other recoverables. We retry these; everything else is fatal.
_TRANSIENT_MARKERS = (
    "socket connection was closed",
    "fetch failed",
    "econnreset", "etimedout", "epipe", "enotfound", "eai_again",
    "connection error", "connection reset", "connection closed",
    "terminated", "network error", "service unavailable", "bad gateway",
    "gateway timeout", "internal server error", "overloaded",
)
_TRANSIENT_STATUSES = {408, 425, 429, 500, 502, 503, 504, 529}
MAX_RUN_ATTEMPTS = 2  # one original + one resume; a transient drop is almost always gone by retry 2
RETRY_BACKOFF_S = 1.5

# Resume nudge: the first attempt ran via streaming-input with the full _agent_user_message; the
# retry is a single-shot string on the SAME resumed session, so the model still sees every tool
# result from before. Tell it to finish, not restart, so it doesn't re-run the search tools.
_RESUME_NUDGE = (
    "The previous turn was cut off by a transient API connection drop before you finished. "
    "Continue exactly where you left off using the search results already in your context — do "
    "NOT repeat tool calls you have already run. Commit any remaining findings and finish with "
    "exactly one answer item."
)


def _is_transient(text: object, status: object = None) -> bool:
    if isinstance(status, int) and status in _TRANSIENT_STATUSES:
        return True
    t = str(text or "").lower()
    return any(marker in t for marker in _TRANSIENT_MARKERS)


async def run_agent(prompt: str, model: str | None = None):
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    token = _emit_queue.set(q)
    selected_model = resolve_model(model)
    run_options = build_options(selected_model)

    run_start = time.perf_counter()
    cost_total = 0.0  # cumulative LIVE estimate from per-turn usage; reconciled to SDK total at done

    async def drive() -> None:
        nonlocal cost_total
        last_turn = time.perf_counter()
        counted_msg_ids: set[str] = set()  # dedupe repeated AssistantMessage streams by message_id
        session_id: str | None = None      # captured so a transient drop can resume the SAME session
        done_payload: dict[str, Any] | None = None  # final accounting; emitted once, after the last attempt

        async def consume(prompt_arg, opts) -> tuple[str, str]:
            # Drive ONE query() attempt. Returns (status, detail) where status is "ok" (clean
            # ResultMessage, success or non-transient error — done_payload is stashed) or "transient"
            # (recoverable drop — caller may resume). Raises on a fatal/non-transient exception.
            nonlocal cost_total, last_turn, session_id, done_payload
            async for msg in query(prompt=prompt_arg, options=opts):
                sid = getattr(msg, "session_id", None)
                if sid:
                    session_id = sid  # any message carries it; last one wins (stable within a session)
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            await q.put({"lane": "trace", "type": "thought", "text": block.text})
                        elif isinstance(block, ToolUseBlock):
                            await q.put({"lane": "trace", "type": "tool_call", "name": block.name, "args": block.input})
                    # Per-turn (LLM component) cost + latency. The SDK streams the SAME AssistantMessage
                    # multiple times (one emission per content block group) with the same message_id, so
                    # count a turn only ONCE per message_id. NOTE: the per-turn usage here is a
                    # message_start SNAPSHOT — input/cache tokens are real, but output_tokens is a stub
                    # (~1), because the model hasn't generated its output yet. The true output (and thus
                    # the bulk of cost on answer-heavy turns) only lands in ResultMessage at the end. So
                    # this running figure is a deliberate LOWER-BOUND estimate; we reconcile to the
                    # authoritative ResultMessage.total_cost_usd on `done`. See cost-reconcile log below.
                    if msg.usage and (msg.message_id is None or msg.message_id not in counted_msg_ids):
                        if msg.message_id is not None:
                            counted_msg_ids.add(msg.message_id)
                        now = time.perf_counter()
                        # Prefer the model id the API actually billed (msg.model); fall back to
                        # the requested model if that's missing on a particular message.
                        turn_cost = usage_cost(msg.usage, msg.model or selected_model)
                        cost_total += turn_cost
                        fresh_in = msg.usage.get("input_tokens", 0)
                        cache_r = msg.usage.get("cache_read_input_tokens", 0)
                        cache_w = msg.usage.get("cache_creation_input_tokens", 0)
                        await q.put({
                            "lane": "trace", "type": "usage", "model": msg.model,
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
                    # A clean ResultMessage can still report a transient API failure (the CLI caught it
                    # instead of crashing). Detect that BEFORE finalizing so the caller can resume.
                    if msg.is_error and _is_transient(
                        msg.result or " ".join(str(e) for e in (msg.errors or [])),
                        msg.api_error_status,
                    ):
                        return "transient", str(msg.result or msg.api_error_status or "transient API error")
                    # ResultMessage is the authoritative end-of-run accounting. total_cost_usd is the
                    # real billed cost (our running cost_total is only a lower-bound estimate — see the
                    # note in the AssistantMessage branch). model_usage carries the real cumulative
                    # token counts (incl. the output our live snapshots stubbed at ~1). Reconcile both.
                    final_cost = msg.total_cost_usd if msg.total_cost_usd is not None else cost_total
                    total_tokens = authoritative_tokens(msg.model_usage)
                    # Reconciliation log: estimate vs. authoritative. A large ratio is the signal that
                    # the live figure diverged from the billed total (stubbed output under-counts; a
                    # subscription/OAuth billing path can make the authoritative total lower instead).
                    if msg.total_cost_usd is not None and msg.total_cost_usd > 0:
                        ratio = cost_total / msg.total_cost_usd
                        print(
                            f"[cost-reconcile] estimate=${cost_total:.4f} "
                            f"authoritative=${msg.total_cost_usd:.4f} ratio={ratio:.2f}x "
                            f"turns={msg.num_turns} model_usage={msg.model_usage}",
                            file=sys.stderr, flush=True,
                        )
                    done_payload = {
                        "lane": "trace", "type": "done", "is_error": msg.is_error,
                        "cost_total": round(final_cost, 6),
                        "total_tokens": total_tokens,
                        "duration_ms": msg.duration_ms,
                        "num_turns": msg.num_turns,
                        "subtype": msg.subtype,
                        "stop_reason": msg.stop_reason,
                        "result": msg.result,
                        "errors": msg.errors,
                        "api_error_status": msg.api_error_status,
                    }
                    return "ok", ""
            return "ok", ""  # stream ended without a ResultMessage — treat as a clean (empty) finish

        async def prompts():
            yield {"type": "user", "message": {"role": "user", "content": _agent_user_message(prompt)}}

        try:
            for attempt in range(1, MAX_RUN_ATTEMPTS + 1):
                if attempt == 1:
                    prompt_arg, opts = prompts(), run_options
                else:
                    # Resume the captured session with a single-shot nudge: the model still has every
                    # prior tool result in context, so it finishes the interrupted turn rather than
                    # re-running the search tools (whose findings were already streamed to the client).
                    prompt_arg = _RESUME_NUDGE
                    opts = build_options(selected_model, resume=session_id)
                try:
                    status, detail = await consume(prompt_arg, opts)
                except (CLIConnectionError, ProcessError) as exc:
                    if not _is_transient(exc):
                        raise  # fatal — let the outer handler emit error + done
                    status, detail = "transient", str(exc)

                if status == "ok":
                    # Always emit exactly one done so the consumer loop terminates; synthesize a
                    # minimal one if the stream ended without a ResultMessage.
                    await q.put(done_payload or {
                        "lane": "trace", "type": "done", "is_error": False,
                        "cost_total": round(cost_total, 6),
                    })
                    return

                # status == "transient": resume if we have a session left and attempts to spare.
                if attempt >= MAX_RUN_ATTEMPTS or not session_id:
                    await q.put({"lane": "trace", "type": "error",
                                 "text": f"Transient API error, gave up after {attempt} attempt(s): {detail}"})
                    await q.put({"lane": "trace", "type": "done", "is_error": True,
                                 "cost_total": round(cost_total, 6)})
                    return
                await q.put({"lane": "trace", "type": "notice",
                             "text": f"⟳ Transient API drop — resuming run (attempt {attempt + 1}/{MAX_RUN_ATTEMPTS})"})
                await asyncio.sleep(RETRY_BACKOFF_S * attempt)
        except Exception as exc:  # noqa: BLE001
            await q.put({"lane": "trace", "type": "error", "text": str(exc)})
            await q.put({"lane": "trace", "type": "done", "is_error": True, "cost_total": round(cost_total, 6)})

    task = asyncio.create_task(drive())
    trace_seq = 0
    pending_call_ts: float | None = None
    report_seen = False
    answer_seen = False
    ask_seen = False
    last_thought_text = ""
    seen_report_keys: set[str] = set()  # drop findings re-emitted by a resumed attempt (U4 insurance)
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
                if ev.get("type") == "thought" and str(ev.get("text") or "").strip():
                    last_thought_text = str(ev["text"]).strip()
            elif ev.get("lane") == "report":
                # A resume shouldn't re-run the search tools, but guard anyway: if a retried attempt
                # re-commits a finding we already streamed, drop the duplicate. Keyed on identity
                # (kind + page + text/caption); the single answer is exempt — a retry's answer is the
                # first one the client sees, so it must pass through.
                if ev.get("kind") != "answer":
                    key = json.dumps(
                        [ev.get("kind"), ev.get("page"), (ev.get("text") or ev.get("caption") or "")[:160]],
                        ensure_ascii=False,
                    )
                    if key in seen_report_keys:
                        continue
                    seen_report_keys.add(key)
                report_seen = True
                if ev.get("kind") == "answer":
                    answer_seen = True
            elif ev.get("lane") == "ask":
                ask_seen = True
            ev.pop("_ts", None)
            # Guarantee a pinned answer block on every successful run. The agent emits its own
            # `answer` for TOPICAL coverage maps and for narrow yes/no / temporal / negative
            # questions; if it finishes without one (e.g. it dumped quotes for a narrow query and
            # left the verdict in its final trace thought), promote that last thought so the reader
            # gets a direct answer at the top instead of a pile of findings to infer from.
            # Guard: when findings WERE committed, only promote a thought that reads like an answer
            # (cites a page, per the output contract) — otherwise a procedural sign-off ("now let me
            # finalize…") would get pinned as the Answer. With no findings, the thought IS the answer
            # (e.g. an uncited negative like "no mentions found"), so promote it regardless.
            if (
                ev.get("type") == "done"
                and not ev.get("is_error")
                and not answer_seen
                and not ask_seen
                and last_thought_text
                and (not report_seen or "[p" in last_thought_text)
            ):
                answer_seen = True
                yield {"lane": "report", "kind": "answer", "text": last_thought_text}
            yield ev
            if ev.get("type") == "done":
                break
        await task
    finally:
        _emit_queue.reset(token)


if __name__ == "__main__":
    async def _main() -> None:
        async for event in run_agent("Find one page about solar energy and add a short report item."):
            print(json.dumps(event, ensure_ascii=False))

    asyncio.run(_main())
