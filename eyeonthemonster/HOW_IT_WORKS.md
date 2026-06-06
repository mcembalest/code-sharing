# How this app works

Agentic search over **Michael Cembalest's "Eye on the Market"** — ~20 years / 5,117 pages of
chart-heavy J.P. Morgan letters. You ask a question; an agent searches the corpus with real tools and
assembles a cited report live, where every claim links back to the source page (and its section) in
the original PDF.

This doc is the accurate, current description of the shipped system — written to be turned directly
into a "How this app works" diagram. Tuned constants are justified separately in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## 1. Offline pipeline — build the index once

The 233 MB PDF is processed once into a searchable index. Each stage consumes the previous stage's
output; none of it runs at query time.

```
Eye on the Monster.pdf
        │  (structural pass: page text, date propagation, disclaimer stripping)
        ▼
_diag/pages_clean.jsonl ── 5,117 page records {page, issue_date, content_text,
        │                    is_chart_bearing, position_in_issue, disclaimer_class}
        │
        ├──► build_issues.py  ─► index/issues.jsonl ── 632 "issues" (contiguous letters),
        │                         each {issue_id, issue_date, title, page_start, page_end}
        │                         = the SECTION boundaries + a coverage denominator
        │
        ├──► build_cards.py   ─► index/cards.jsonl ── 2,300 vision "cards": a Gemini 3.5 Flash
        │      (gemini-3.5-flash)   prose description of every chart-bearing page (titles, axes,
        │                           series, the numbers worth citing). ~$1–2 one-time.
        │
        ├──► build_topics.py  ─► index/topics.json + page_topics.jsonl
        │      (two-pass)          Pass A: Gemini text pass emits 3–7 raw topic phrases/page.
        │                          Pass B: embed + cluster the phrases → a normalized taxonomy of
        │                          14 high-level buckets / 3,933 granular topics. = the second
        │                          coverage denominator; collapses 20-year vocabulary drift
        │                          ("solar cost" 2005 ≈ "module ASP / LCOE" 2025).
        │
        └──► build_index.py   ─► index/{embeddings.npy, bm25.pkl, pages.json}
               per page, merge into one searchable surface:
               search_text = content_text + "\n\n" + card_text + "\n\n" + topic-label text
               → embeddings.npy  (static-retrieval-mrl-en-v1, 4,882 × 1024, L2-normalized)
               → bm25.pkl        (BM25Okapi over tokenized search_text)
               → pages.json      (aligned row-for-row with the embeddings)
```

235 disclaimer-only pages are dropped, leaving **4,882 kept pages** as the citation/retrieval unit.
The embedding model is `static-retrieval-mrl-en-v1` — benchmarked against transformer models and kept
because it ties/beats them on this corpus while being far faster (see DECISIONS.md).

> **Reproducibility.** The PDF→pages chain lives in `_diag/` (`scan.py` → `enrich.py` →
> `strip_disclaimers.py`); see `_diag/README.md`. Its `*.jsonl` outputs (incl. `pages_clean.jsonl`)
> are gitignored because they're large and fully regenerable — recreate them offline with
> `uv run python _diag/scan.py && uv run python _diag/enrich.py && uv run python _diag/strip_disclaimers.py`
> (deterministic; verified to reproduce the exact 5,117-record / 4,882-kept set the deployed index was
> built from). The durable artifacts are the **PDF** + `_diag/` scripts + the built **`index/`**.

---

## 2. Runtime — answering one query

```
   browser (static/index.html)
        │  GET /run?q=…&model=haiku|sonnet|opus            (server.py, FastAPI)
        ▼
   run_agent(q, model)  ── Claude Agent SDK (claude-agent-sdk → spawns the `claude` CLI)
        │   • locked to 7 in-process "eom" tools (can_use_tool allowlist; no Bash/Read/Web)
        │   • loops: classify query shape → retrieve → commit findings → synthesize answer
        ▼
   one asyncio.Queue of lane-tagged events  ──►  Server-Sent Events  ──►  two UI panes
```

### The 7 tools the agent can call

| tool | what it does |
|---|---|
| `find_mentions(terms, [date_range])` | exact regex over page text + chart cards; **auto-commits** matched pages as quote findings. First choice for named people/orgs/products/phrases. |
| `search(q, [chart_only], [limit≤25])` | hybrid **BM25 + embedding** retrieval, fused by Reciprocal Rank Fusion (k=60). Ranked hits, does **not** commit. For widening / vocabulary drift. |
| `list_topics([high_level])` | browse the topic taxonomy (buckets → granular ids). |
| `enumerate(topic_ids, q, [chart_only], [must_contain])` | union all pages tagged the given topics, rank by similarity to `q`, **auto-commit** the ones above `RELEVANCE_FLOOR` (0.23, calibrated), fold the rest into a coverage count. The backbone for "everything about X". |
| `get(page \| start,end \| issue_id)` | fetch full page/section records to read context. |
| `ask_user(question, why)` | pause for one clarifying question (high bar). |
| `add_report_items(items)` | batch-commit synthesis: headings, narrative, curated quotes/charts, and the final `answer` block. |

### How effort matches the question (query shape)
The system prompt makes the agent classify before retrieving:
- **NARROW / factual** ("did I mention X in 2007?") → one targeted lookup + a one-line verdict.
- **TOPICAL / exhaustive** ("everything about X") → enumerate over topics, group by sub-topic/chronology, finish with a coverage-map answer.
- **ANALYTICAL** ("what did I get wrong", "open questions I posed") → decompose; gather predictions, find contradicting later issues, cite both sides.

For coverage/chart queries the agent fans out **several angle-diverse reformulations** (different
framings of the subject, not just synonyms) and unions the hits — a single query's ranking is fragile,
so a valid chart indexed under one framing isn't missed under another.

---

## 3. The two-lane event stream — the seam between agent and UI

Every event carries a `lane`. One stream feeds both panes:

- **`lane: "report"`** — the report assembling live. `kind ∈ {heading, quote, chart, narrative,
  answer}`. Findings carry `page, issue_date, title, relevance (primary|supporting), src`. The
  `answer` block (emitted once at the end) is the synthesized verdict with inline `[p. N]` links.
- **`lane: "trace"`** — the machinery. `type ∈ {thought, tool_call, tool_result, done, error,
  notice}`, each with a monotonic `id` used for provenance linking.

The report **is** the accumulation of report-lane events — there is no separate final dump.

---

## 4. What the user sees (static/index.html)

- **Report pane (hero):** the `answer` pinned at top, then findings grouped into collapsible sections.
  `primary` findings show in full; `supporting` ones tuck behind a per-section disclosure.
- **Trace drawer (collapsible):** the agent's live tool calls and reasoning. Each finding shows a
  `src` chip that opens the drawer at the tool event that produced it.
- **Clickable citations:** every `[p. N]` / `[pp. N, M-K]` becomes a link that opens the rendered PDF
  page in a viewer, with its **section** (issue title, date, page range) and section-scoped navigation.
- **Progress bar** driven by tool activity; **Download PDF** exports the assembled report.

PDF pages are rendered on demand by `server.py` (`/page_image/{n}`), serialized behind a lock (PyMuPDF
isn't thread-safe) and cached immutably so repeat views are instant.

---

## 5. Deployment

`modal_app.py` packages the repo as an ASGI app (`server:app`) on **Modal**, app name
`eyeonthemonster`. Secrets (`ANTHROPIC_API_KEY` for the runtime agent, `GEMINI_API_KEY` for rebuilding
cards) come from the Modal secret `eyeonthemonster-secrets`. Deploy with
`uv run modal deploy modal_app.py`. The runtime survives transient API-socket drops by resuming the
session mid-run (see TODO.md → Design decisions).

---

## Repo map

- **Offline build:** `_diag/{scan,enrich,strip_disclaimers}.py` (PDF→`pages_clean.jsonl`, see `_diag/README.md`), then `build_issues.py`, `build_cards.py`, `build_topics.py`, `build_index.py`, `clean_text.py`
- **Runtime:** `agent.py` (7 tools + SDK + two-lane events + shape-aware prompt), `server.py` (SSE + PDF render/export), `static/index.html`
- **Eval / tuning:** `eval_run.py` (headless run; prints cost/turns/latency), `calibrate.py` (model benchmark + floor calibration → `docs/`)
- **Deploy:** `modal_app.py`
- **Index** (`index/`): `issues.jsonl`, `cards.jsonl`, `topics.json`, `page_topics*.jsonl`, `embeddings.npy`, `bm25.pkl`, `pages.json`
- **Durable data:** `Eye on the Monster.pdf` + `index/` (the index is the source of truth at runtime; per-page text lives in `index/pages.json`). Reference: `golden_queries.txt`, `baseline/` (text-only baseline)
- **Models:** vision `gemini-3.5-flash`; embeddings `static-retrieval-mrl-en-v1` (`MODEL_NAME` in `build_index.py`/`agent.py`)
