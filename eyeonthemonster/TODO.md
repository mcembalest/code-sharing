# TODO — worklist

Read this top-to-bottom, then `BUILD_BRIEF.md` (the canonical spec). The retrieval/agent
pipeline (W1–W5) is built, runs end-to-end, and has been hardened (see *Done*). The remaining
work is mostly the **product UI** (the U-series below — the user's headline goal) plus two
**Human-gated** items and a validation pass. Each item has a verify step.

## Prerequisites

- **Self-contained repo.** Everything load-bearing is here — no external memory needed.
  Structural work is baked into `_diag/pages_clean.jsonl`; consume it, never re-derive it.
- **API keys** (`.env`, gitignored, never print/commit): `GEMINI_API_KEY` and
  `ANTHROPIC_API_KEY` are both present and working.
- **You build; the runtime agent is Claude** (`claude-agent-sdk` + `claude-haiku-4-5`). Don't
  swap the runtime to another provider.
- **Env is pre-synced** (`uv sync` done). Use `uv run …`. UI: `uv run uvicorn server:app`
  (serves on `http://127.0.0.1:8000`).
- **Browser-with-vision is wired for the next session.** A project-scoped `.mcp.json` at the repo
  root registers the **Playwright MCP** (`npx @playwright/mcp@latest`). It is NOT active until
  Claude Code is restarted and the server is approved at startup; first call downloads Chromium.
  Once live, tools appear as `mcp__playwright__*` — navigate to the running app and screenshot to
  iterate on the UI visually. This is the intended workflow for finishing the U-series.
  - **Codex / non-Claude-Code agents:** this `.mcp.json` is **Claude Code-specific** and will NOT be
    picked up automatically. To get the same browser-with-vision loop, register Playwright in your
    own MCP config (Codex: `~/.codex/config.toml` → `[mcp_servers.playwright]` with
    `command = "npx"`, `args = ["@playwright/mcp@latest"]`), or work code-only and have the human
    screenshot. Separately: the **runtime eom agent** (`agent.py`) is `claude-haiku-4-5` via
    `claude-agent-sdk`, which spawns the `claude` CLI and bills whatever that login uses — currently
    the Claude **subscription** (OAuth), NOT the `.env` `ANTHROPIC_API_KEY`. So *live* agent runs
    still consume Claude credits regardless of which coding agent drives the repo. To bill the API
    key instead, move `load_dotenv(ROOT/".env")` to module-import time in `agent.py` (it's currently
    lazy, inside `_load()`, so it lands after the subprocess is spawned).
- **Index rebuild is cheap and offline.** `uv run python build_topics.py --normalize-only`
  (no API cost — raw topics in `page_topics_raw.jsonl` are cached) then
  `uv run python build_index.py`. Full rebuild ~3 min, dominated by the phrase linkage.

## Current state (updated 2026-05-25) — pipeline COMPLETE + hardened

- ✅ **W1 cards** — `build_cards.py` → `index/cards.jsonl`, ~2,300 chart-bearing pages.
- ✅ **W2 topics** — `build_topics.py` → `page_topics_raw.jsonl` (raw, cached), `page_topics.jsonl`,
  `topics.json`: **14 high-level buckets, 3,931 granular topics** over 4,882 kept pages, rolled
  up onto 686 issues. (See *Done → taxonomy* for the rebuild rationale.)
- ✅ **W3 index** — `build_index.py` → `index/{embeddings.npy, bm25.pkl, pages.json}`; rows align
  (4,882 == 4,882 == 4,882, emb dim 1024).
- ✅ **W4 agent** — `agent.py`: 11 eom tools, SDK wiring, `asyncio.Queue` → two-lane events.
  Tool lockdown (`can_use_tool=_only_eom` + `disallowed_tools`) is in and verified with the
  streaming-prompt input the SDK requires for `can_use_tool` to fire. SYSTEM prompt is the
  drafted v1 (still owner-gated — see Human-gated).
- ✅ **W5 UI** — `server.py` + `static/index.html`; `/run` streams SSE, `/page_image/{n}` opens
  the PDF once at startup. **This is the stub UI the U-series replaces.**
- ✅ **Eval harness** — `eval_run.py`: runs `run_agent` headless, prints the `coverage_status`
  trajectory + final cited `[p. N]` set as text/JSON. This is what the golden key gets scored against.

### Done this session (review of the autonomous worklist — N1–N5)

- ✅ **N1 lean tool outputs** — payloads capped/compacted/truncated; typical call a few KB.
- ✅ **N3 server hygiene** — single module-level `fitz.open`; `/page_image/{n}` bounds-checked.
- ✅ **N4 eval harness** — `eval_run.py` (above).
- ✅ **N5 taxonomy — reworked.** The first pass had two defects: (1) the granular layer had
  silently ballooned to 9,583 near-duplicate clusters (barely merged), and (2) high-level routing
  was brittle substring-on-medoid-label matching, dumping clearly-routable topics
  (`germany energiewende`, `electoral college`, `large language model`, `fannie mae`, `volcker rule`,
  `tarp`, `abenomics`, `opeb`) into a giant `other` grab-bag. Fixed:
  - Clustering switched to scipy `linkage`/`fcluster` at **cosine-sim 0.50 → 3,931 coherent
    granular clusters** (near-dups merged; threshold is `--threshold`, tunable).
  - Routing is now **medoid-first keyword matching with member-vocabulary fallback** and
    word-prefix matching (denylist for collision-prone short tokens like `coal`→coalition).
  - High-level `n_pages` is now a **true unique-page count** (was a double-counted sum).
  - Result: pages with no named bucket dropped 47 → **15 / 4,882 (0.3%)**; the listed misroutes
    are all correctly placed. Verified live: a solar-costs query cited pp.1680–1682, 2355 — all
    `energy` / `solar-photovoltaic-costs` pages.
- ✅ **agent.py bug fixes (beyond N-scope):**
  - **Coverage state is now per-run** (`_inspected` → `_inspected_var` contextvar). Was a module
    global that would leak one query's inspected issues into the next in the long-lived server and
    corrupt `coverage_status`. Verified isolated across sequential runs.
  - **`list_topics` drill-down** — added `list_topics(high_level=<bucket id>)` returning the
    bucket's top-100 granular ids (sorted by coverage). Without it only ~70 of thousands of
    granular ids were reachable, breaking the prompt's "enumerate topics" backbone. SYSTEM step 2
    updated to teach it.
  - `bounded_limit` default aligned to `MAX_SEARCH_LIMIT`.
- ✅ **U3/U4 backend pre-wired** (so a fresh session can do the UI front-end-only): `add_to_report`
  gained `relevance` + a `kind:"answer"`; SYSTEM instructs relevance tagging + a final answer
  block; trace events now carry a per-run monotonic `id`. Verified live (answer block + 13/4
  primary/supporting + ids 1..97). See U3/U4 for the remaining front-end half.

### UI session — U1–U3 front-end built (2026-05-25)

`static/index.html` was rewritten to consume the existing two-lane contract (no backend changes,
no field renames). Implemented and pending a real browser pass:
- ✅ **U1 layout** — report is now a full-width, page-scrolling hero; `#bar` is sticky; the trace
  moved into a **fixed right drawer** (slides via `transform`), **collapsed on load**, toggled by an
  "Agent trace ▸/▾" button in `#bar`, open/closed persisted in `localStorage` (`traceOpen`). Trace
  streams into the drawer even while hidden (uses the drawer's own `scrollTop`). Mobile = full-width
  drawer. *Design choice:* the drawer **overlays** the report rather than reflowing it (keeps the
  report the hero) — revisit in U-polish if it feels like it occludes content.
- ✅ **U2 sections** — `heading` opens a collapsible `<details class="section">` (styled as h2);
  `quote/chart/narrative` nest under the most recent heading in document order; a default "Findings"
  section is created lazily if findings arrive before any heading.
- ✅ **U3 relevance** — `answer` is pinned at the top in `#answer-slot` (labeled "Answer"; arrival
  scrolls to top); `primary` quotes/charts render in full; `supporting` ones are tucked behind a
  per-section "▾ N supporting findings" disclosure, supporting charts as ~240px thumbnails.
- **Zero-risk U4 hooks pre-seeded:** each finding carries `data-src`; each trace event renders with
  `id="trace-ev-{id}"`. No provenance chips yet — that's the U4 front-end.
- **Not yet verified in a browser.** No Playwright pass has run (MCP needs the restart above). The
  per-U Verify steps below are the acceptance checks for the next session.

---

## ★ ACTIVE — performance/recall pivot (see `OPTIMIZATION_PLAN.md`)

> **Direction change (owner, 2026-05-25):** optimize for **lower latency + exhaustive recall**, and
> shift the product from agent synthesis toward **result enumeration**. Plan + diagnosis (a live run
> spends 78% of its ~139 tool calls manually reading+committing pages one LLM turn at a time, and
> `search_topic` silently caps recall at 10) live in **`OPTIMIZATION_PLAN.md`**. The SYSTEM-prompt
> parts there remain owner-gated. Also fixed this session: the `in_date_range` lexical-compare bug
> that made every date-scoped search return 0 hits (`agent.py` `_norm_date`).

## ★ NEXT SESSION — make the UI *fantastic* (Playwright-driven)

> The headline goal. U1–U3 are coded but unverified; U4–U6 remain. The plan: **restart so the
> Playwright MCP is live, run the app, and iterate on the real rendered UI with screenshots** —
> don't ship UI by reading code alone.

Loop to follow:
1. Start the app (`uv run uvicorn server:app`), navigate Playwright to `http://127.0.0.1:8000`.
2. Run a real query (a multi-section one like a solar-costs / pensions query exercises headings +
   primary/supporting + the answer block). Screenshot **collapsed-on-load**, **drawer open
   mid-run**, and **finished report**.
3. **Verify U1–U3** against their Verify steps below (incl. the folded-in N2 check: a `quote` is a
   blockquote with `[p. N]`, a `chart` is an inline `/page_image` figure).
4. **Then make it genuinely great**, not just correct. Visual-quality bar to push on:
   typography/rhythm, the answer block as a confident hero, clear primary-vs-supporting hierarchy,
   chart thumbnails that read well, drawer polish (overlay vs. reflow, shadow, transitions), loading
   / empty / error states, and the mobile layout. Treat this as a design pass, screenshot-driven.
5. **Build U4 (provenance chips) and U6 (live progress)** — the `data-src` / `trace-ev-{id}` hooks
   for U4 are already in the markup; U5 export is lower priority.

## PRODUCT VISION — the two-panel report UI (U-series, the headline goal)

> **One collapsible side panel** shows the agent's action trace (its tool calls / reasoning).
> **One main panel** shows the assembled report in markdown/HTML — and makes clear **which
> quotes / charts / syntheses best answer the user's query**, not just a flat arrival-order dump.

The event contract is in place AND the relevance/answer/provenance backend is already wired
(verified live 2026-05-25) — **the remaining U-series work is almost entirely front-end
(`static/index.html`).** Current event shapes the UI consumes:
- `lane:"report"`, `kind ∈ {heading, quote, chart, narrative, answer}`. `quote`/`chart` carry
  `relevance ∈ {"primary","supporting"}` (defaults to `"supporting"`). `answer` is emitted once
  at the end — a synthesized direct answer with inline `[p. N]` (render it pinned at the top).
  Other fields: `page, issue_date, title, caption, text, src`.
- `lane:"trace"`, `type ∈ {thought, tool_call, tool_result, done, error}`, each with a per-run
  monotonic `id` (use it as the DOM anchor / provenance target).
Do not rename these fields — extend if needed.

### U1 — Layout: report is the hero, trace is a collapsible drawer
✅ **CODED (2026-05-25), pending browser verify.** See "UI session" above for what was built.
`static/index.html` was a fixed 2-column grid with the 400px trace always visible; it is now a
full-width report + collapsible right drawer.
- Make the report full-width by default; move the trace into a **collapsible right drawer**
  (toggle button in `#bar`, e.g. "Agent trace ▸"; collapsed on load). Persist the open/closed
  choice (localStorage). Keep streaming the trace into the (possibly hidden) drawer so opening it
  mid-run shows full history. Keep the existing mobile breakpoint behaving.
- **Verify:** page loads report-only, full width; toggle reveals/hides the trace without losing
  events; a live run streams into the report while the drawer stays collapsed. (This also closes
  the never-done **N2** browser check: confirm a `quote` renders as a blockquote with `[p. N]`
  and a `chart` renders an inline `/page_image` figure.)

### U2 — Report structure: group findings under their sections
✅ **CODED (2026-05-25), pending browser verify.** `heading` now opens a collapsible section and
findings nest under it.
Was append-only in arrival order; `heading` events didn't open a
container. The renderer now **nests** `quote/chart/narrative` under the most recent `heading`
so the finished document reads as a grouped report (the SYSTEM prompt already tells the agent to
group by sub-topic / chronology and to emit headings).
- **Verify:** a multi-section query renders collapsible `<section>`s, each with its heading and
  the findings committed under it, in document order.

### U3 — Relevance: surface which findings best answer the query  ★ core of the vision
**Decided (owner, 2026-05-25): agent-tagged relevance + a synthesized answer block.**
- ✅ **BACKEND DONE (agent.py + SYSTEM, verified live):** `add_to_report` takes `relevance`
  (`primary`|`supporting`, default `supporting`) and a `kind:"answer"`; SYSTEM instructs the agent
  to tag findings and to FINISH with one `answer` block (direct answer + inline `[p. N]` to the
  primary findings). A live solar-costs run emitted the answer block + 13 primary / 4 supporting
  tags. *Tuning note:* the agent currently leans toward `primary` (13/17) — if the rendered report
  feels under-differentiated, tighten the "reserve primary for the strongest" wording in SYSTEM.
- ✅ **FRONT-END CODED (2026-05-25), pending browser verify:** `answer` is pinned at the top of the
  report; `primary` quotes/charts render in full; `supporting` ones are tucked behind a per-section
  "▾ N supporting findings" disclosure; `supporting` charts render as ~240px thumbnails. *Tuning
  note still open:* watch whether primary-vs-supporting reads as under-differentiated in the browser
  (the agent leans `primary`) — may want a stronger visual treatment and/or the SYSTEM-prompt tweak.
- *Optional future complement* (no extra LLM, skip for v1): within each tier, order findings by
  similarity of their text to the query embedding (`static-retrieval-mrl-en-v1`). Weak on
  analytical queries, so never the primary signal.
- **Verify:** the report opens with the pinned `answer`; primary vs. supporting are visually
  distinct; the disclosure expands/collapses supporting findings.

### U4 — Provenance: link a finding back to the trace that surfaced it  (closes a known gap)
- ✅ **BACKEND DONE:** every trace event now carries a per-run monotonic `id` (verified 1..97).
- ⬜ **FRONT-END:** render each finding's `src` as a small chip; on click, open the trace drawer
  scrolled to the matching event. Correlate by `src` string (e.g. `"search_topic:solar-pv-costs"`)
  against trace `tool_call` name+args; use the trace `id` as the DOM anchor to scroll to.
- **Verify:** clicking a finding's provenance chip reveals the originating tool call in the drawer.

### U5 — Export the assembled report
The report is markdown-native already. Add a "Copy as Markdown" / "Download .md" (and/or `.html`)
control that serializes the assembled report — answer block, sections, quotes with citations,
chart references (`[chart: p. N]` or embedded `/page_image` for HTML).
- **Verify:** export round-trips to a clean, citation-complete document.

### U6 — Live progress in the hero panel
Surface liveness without opening the trace: a thin status line / progress bar driven by
`coverage_status` events ("inspected X/Y issues") and the current tool name, plus a running/done
spinner.
- **Verify:** progress advances as `coverage_status` fires and resolves on `done`.

---

## Human-gated — needs the owner (do NOT fabricate)

- **Real SYSTEM prompt** — v1 is drafted and wired into `agent.py` (drives: list_topics →
  drill-down → granular `search_topic` → widen with semantic/keyword → `add_to_report` live →
  `mark_inspected` → poll `coverage_status` until `remaining==[]`). **Owner to review/refine**,
  especially the new relevance-tagging + answer-block instructions once U3's contract is set.
- **Golden answer key** — hand-built true answer set for ONE `golden_queries.txt` query; needs
  corpus-owner knowledge to judge true recall.

## Validation (after the prompt + key exist)

Run the 5 golden queries through the UI; confirm the agent enumerates topics/issues on exhaustive
queries and decomposes the analytical ones (got-wrong, open-questions). Score cited pages (via
`eval_run.py`) against the golden key. Compare against the text-only baseline in
`baseline/golden_results.md`.

## Smaller follow-ups / smells

- **`other` bucket tail** — routing fixed, but ~1,246 small/idiosyncratic granular clusters still
  land in `other` (Thucydides trap, IRS circular 230, HFRI index, …). Only 15 pages have *no*
  named bucket, so this is low priority; a light LLM-naming pass over the largest `other` clusters
  could reclaim a few (`pumped hydro storage`→energy, `troubled asset relief program`→banks).
- **"equities" morphology** — the prefix keyword `equity` doesn't match the plural "equities"
  (different stem), so a few equity-only clusters route by their region instead. Harmless; add an
  `equit` stem if it ever matters.
- **Issue-boundary spot-checks** — eyeball `index/issues.jsonl` around **2015-09-08, 2018-12-10,
  2026-03-03** (likely legit multi-page special reports, but verify boundaries).
- **Embedding model swap** — `MODEL_NAME` is a deliberately swappable axis; trying a stronger
  embedding model and measuring recall lift is an intended experiment, not a defect.

## Design decisions — DO NOT silently undo

These were deliberate and hard-won; a fresh agent may be tempted to "simplify" them. Don't.
- **Agentic-iterative, not one-pass retrieval.** The agent calls search as a tool and loops.
  Exhaustiveness comes from **enumerable denominators** (the issue table AND the topic index),
  not from telling the model "be thorough." Two denominators on purpose.
- **Claude Agent SDK, not the raw Client SDK.** The SDK **compacts context**, so coverage state
  MUST live in tools (`mark_inspected`/`coverage_status`), never in the transcript. And that state
  is **per-run** (`_inspected_var` contextvar) so the long-lived server can't leak coverage across
  queries — do not revert it to a module global.
- **Tool lockdown via `can_use_tool` is mandatory**, and `can_use_tool` only fires when `query()`
  is fed a **streaming async-iterable prompt** (see `run_agent`). A plain string prompt silently
  bypasses the lockdown and the agent shells out via Bash. Keep the streaming prompt.
- **Topics are a separate TEXT pass over `content_text + card_text`** (build_topics.py), NOT folded
  into the vision call. Do not re-add a `topics:` field to the card prompt.
- **Taxonomy: fixed high-level buckets + medoid-first keyword routing.** High-level is a curated
  ~14-bucket list; each granular cluster routes by its medoid (then member vocabulary) through
  word-prefix keyword matching. Granular granularity is the clustering threshold (`--threshold`,
  currently sim 0.50, ~3,931 clusters) — tune it, don't hand-merge. Cluster coherence was chosen
  over matching any particular topic count.
- **The embedding model (`static-retrieval-mrl-en-v1`) is a deliberately SWAPPABLE axis.** Keep
  `MODEL_NAME` a constant.
- **The `[p. N]` citation contract is non-negotiable** — every claim the reader sees is cited.
- **The two-lane event contract** (`lane:"report"` kinds vs `lane:"trace"` types) is the seam
  between agent and UI. The U-series extends it (`relevance`, `kind:"answer"`, trace `id`) — extend,
  don't rename.

## Project context

- The PDF is the user's father's work: **Michael Cembalest, J.P. Morgan "Eye on the Market"**,
  ~20 years. The **golden answer key (human-gated) needs that owner knowledge**.
- **Budget is not a constraint** (~$1–2 total for the full vision pass). Optimize for quality and
  engineering, not API cost.
- **Two query archetypes** (the 5 in `golden_queries.txt` split ~3/2):
  - *Quantitative-topical* (solar costs, state pensions, polarization) — vision cards + topic
    enumeration carry these; the chart numbers live in `card_text`.
  - *Analytical* (what did I get wrong, what open questions did I pose) — NOT similarity-findable;
    require decomposition. The text-only baseline was weakest here (see `baseline/golden_results.md`).

## Map of the repo

- **Build scripts:** `build_issues.py`, `build_cards.py`, `build_topics.py`, `build_index.py`.
- **Runtime:** `agent.py` (11 eom tools + SDK + lanes; drafted SYSTEM), `server.py`,
  `static/index.html`, `eval_run.py` (headless eval).
- **Indexes** (`index/`, amortized): `issues.jsonl`, `cards.jsonl`, `page_topics_raw.jsonl`,
  `page_topics.jsonl`, `topics.json`, `embeddings.npy`, `bm25.pkl`, `pages.json`.
- **Inputs/reference:** `_diag/pages_clean.jsonl` (canonical page data), `golden_queries.txt`,
  `baseline/golden_results.md` (text-only baseline), `review_cards.py` (card QA tool).
- Vision: `gemini-3.5-flash`. Embeddings: `static-retrieval-mrl-en-v1` (`MODEL_NAME`, swappable).
</content>
</invoke>
