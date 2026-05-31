# TODO — worklist

Read this top-to-bottom, then `BUILD_BRIEF.md` for original project context. The retrieval/agent
pipeline (W1-W5) is built, runs end-to-end, and has been hardened (see *Done*). The product UI
v1 is now built and browser-verified. Remaining work is mostly **latency measurement/optimization**
and **validation by testing + judgment** (run representative queries through the UI / `eval_run.py`
and judge output quality directly). A formal golden key is **optional** (owner decision 2026-05-31)
— useful for a recall number later, NOT a release gate.

## Prerequisites

- **Self-contained repo.** Everything load-bearing is here — no external memory needed.
  Structural work is baked into `_diag/pages_clean.jsonl`; consume it, never re-derive it.
- **API keys** (`.env`, gitignored, never print/commit): `GEMINI_API_KEY` and
  `ANTHROPIC_API_KEY` are both present and working.
- **You build; the runtime agent is Claude** (`claude-agent-sdk`; current model is
  `claude-sonnet-4-6`). Don't swap the runtime to another provider.
- **Env is pre-synced** (`uv sync` done). Use `uv run …`. UI: `uv run uvicorn server:app`
  (serves on `http://127.0.0.1:8000`).
- **Browser-with-vision is wired for the next session.** A project-scoped `.mcp.json` at the repo
  root registers the **Playwright MCP** (`npx @playwright/mcp@latest`). It is NOT active until
  Claude Code is restarted and the server is approved at startup; first call downloads Chromium.
  Once live, tools appear as `mcp__playwright__*` — navigate to the running app and screenshot to
  iterate on UI changes visually.
  - **Codex / non-Claude-Code agents:** this `.mcp.json` is **Claude Code-specific** and will NOT be
    picked up automatically. To get the same browser-with-vision loop, register Playwright in your
    own MCP config (Codex: `~/.codex/config.toml` → `[mcp_servers.playwright]` with
    `command = "npx"`, `args = ["@playwright/mcp@latest"]`), or work code-only and have the human
    screenshot. Separately: the **runtime eom agent** (`agent.py`) uses `claude-agent-sdk`, which
    spawns the `claude` CLI. `.env` is loaded at module-import time before the subprocess starts, so
    `ANTHROPIC_API_KEY` is available to the runtime process.
- **Index rebuild is cheap and offline.** `uv run python build_topics.py --normalize-only`
  (no API cost — raw topics in `page_topics_raw.jsonl` are cached) then
  `uv run python build_index.py`. Full rebuild ~3 min, dominated by the phrase linkage.

## Current state (updated 2026-05-30) — pipeline COMPLETE + hardened

- ✅ **W1 cards** — `build_cards.py` → `index/cards.jsonl`, ~2,300 chart-bearing pages.
- ✅ **W2 topics** — `build_topics.py` → `page_topics_raw.jsonl` (raw, cached), `page_topics.jsonl`,
  `topics.json`: **14 high-level buckets, 3,931 granular topics** over 4,882 kept pages, rolled
  up onto **632 issues** (down from 686 after the date-propagation fix below merged spuriously-
  split issues). (See *Done → taxonomy* for the rebuild rationale.)
- ✅ **W3 index** — `build_index.py` → `index/{embeddings.npy, bm25.pkl, pages.json}`; rows align
  (4,882 == 4,882 == 4,882, emb dim 1024).
- ✅ **W4 agent** — `agent.py`: 7 eom tools, SDK wiring, `asyncio.Queue` → two-lane events.
  Tool lockdown (`can_use_tool=_only_eom` + `disallowed_tools`) is in and verified with the
  streaming-prompt input the SDK requires for `can_use_tool` to fire. SYSTEM prompt is shape-aware
  and no longer broadly owner-gated.
- ✅ **W5 UI** — `server.py` + `static/index.html`; `/run` streams SSE, `/page_image/{n}` opens
  the PDF once at startup. Product UI v1 is in place.
- ✅ **Eval harness** — `eval_run.py`: runs `run_agent` headless and prints tool path,
  enumeration accounting, committed pages, cited pages, and optional golden-key scores.

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
  - **Per-run state isolation** was fixed during the earlier coverage-tool implementation; current
    runtime no longer exposes `mark_inspected` / `coverage_status`, but the long-lived server still
    keeps run-local event state isolated.
  - **`list_topics` drill-down** — added `list_topics(high_level=<bucket id>)` returning the
    bucket's top-100 granular ids (sorted by coverage). Without it only ~70 of thousands of
    granular ids were reachable, breaking the prompt's "enumerate topics" backbone. SYSTEM step 2
    updated to teach it.
  - `bounded_limit` default aligned to `MAX_SEARCH_LIMIT`.
- ✅ **U3/U4 backend pre-wired** (so a fresh session can do the UI front-end-only): `add_to_report`
  gained `relevance` + a `kind:"answer"`; SYSTEM instructs relevance tagging + a final answer
  block; trace events now carry a per-run monotonic `id`. Verified live (answer block + 13/4
  primary/supporting + ids 1..97). See U3/U4 for the remaining front-end half.

### Done 2026-05-30 — date + title fixes + full re-scan

- ✅ **Date-inference hardening** (`_diag/enrich.py`): added monotonicity check during forward
  propagation. A detected date that jumps backward by >365 days from the rolling date is rejected
  as a body-text reference (citation, chart legend, historical event), not a header date. **57
  spurious dates rejected** in the full run — including the originally-reported `Jan 1, 2017`
  poisoning pp. 5042–5044, plus clusters like the 2008 Pearl Harbor reference (`Dec 7, 1941`)
  and the p.4579–4595 historical-events chart-legend cluster inside the June 2025 energy paper.
- ✅ **Title extraction hardening** (`clean_text.py` + `build_issues.py`): per-issue sub-banner
  lines (e.g. `Online Trump Tracker`, `2025 Energy Paper / Trump Tracker`, `2024 Outlook / …
  monitor / … monitor`) were being picked as issue titles. Added narrow strip rules in
  `clean_text.py` (sub-banner, cross-promo `Access our X here`, JPM private-banking preamble
  + continuations, institutional-use disclaimer, OCR-mangled header tail) and wired
  `clean_content` into `build_issues.py::title_from_page`. Repeat-title sweep over 632 issue-
  starts: **0 banner titles, 0 Untitled** (only `Executive Summary` × 2, legitimate).
- ✅ **Full pipeline re-scan** (5117 pages): scan → enrich → strip → build_issues → build_cards
  (2,300 cards in 58 min, 0 errors) → build_topics → build_index (4,882 pages, dim 1024).
  `index/cards.jsonl.bak` kept as a safety net — delete when satisfied.
- ✅ **End-to-end verified in the running app** via Playwright: p.5044 now resolves to
  *"Software is suddenly unloved: positioning and short interest, March 6, 2026"* (previously
  *"Software vs semiconductors, high bandwidth memory etc, Jan 1, 2017"*).
- ✅ **`enumerate` entity guardrail (`must_contain` param + SYSTEM nudge).** Stress-testing the
  app surfaced a precision bug: an Argentina query enumerated 7 broad topics
  (`sovereign-debt`, `currency-pegs`, `emerging-market-vulnerabilities`, …), unioned 242 tagged
  pages, and committed many Greece/Turkey/Mexico/Spain pages that share the concepts but never
  mention Argentina. **Embedding-similarity ranking is conceptual, not lexical** — Greek-default
  charts score high against "Argentina peso default". Fix in `agent.py`:
  - New `enumerate(..., must_contain: list[str])` param: if supplied, pages are kept only when
    their merged `content_text + card_text` contains at least one of the substrings
    (case-insensitive, partial — `["Argentin"]` matches `Argentina` + `Argentine`). Trace summary
    reports `dropped_by_must_contain` for diagnosis.
  - SYSTEM prompt adds an **ENTITY GUARDRAIL** section instructing the agent to pass
    `must_contain=[entity terms]` whenever enumerating broad topics for a named-entity query.
  - Headless verify on the live index: broad-topic union (242 pages) → 27 with
    `must_contain=["argentin"]` — **88.8 % off-target drop**, and net recall of Argentina-mentioning
    pages is HIGHER than narrow Argentina-only topics (27 vs 11), because Argentina is often
    discussed on pages primarily tagged with sovereign-debt themes.
- ✅ **SYSTEM prompt revision — shape branching + optional answer block (owner-decided 2026-05-30).**
  Stress tests showed two real defects: (a) narrow factual queries ("did I write about the iPhone
  in 2007?") opened with "Coverage map:" scaffolding before delivering a one-line answer, and
  (b) the Bitcoin "earliest mention + evolution" query burned ~3M tokens / $1.68 / 10 min
  partly because the agent treated the mandatory answer block as a forcing function to keep
  widening. Revision in `agent.py::SYSTEM`:
  - New **QUERY SHAPE** section up top: classify NARROW vs TOPICAL vs ANALYTICAL before
    retrieval; match effort to shape. Replaces "Your default job is EXHAUSTIVE recall" with
    shape-aware guidance. Ambiguous → default TOPICAL.
  - **ANSWER BLOCK** rule made shape-conditional: TOPICAL queries still finish with one
    `add_to_report(kind="answer", ...)` coverage map; NARROW queries can skip it when the
    auto-committed findings self-suffice. Don't pad a one-line answer with scaffolding.
  - **Stop condition** rewritten: stop when the answer is in hand at the chosen shape, not when
    "find_mentions / enumerate / widening are complete" mechanically.
  - Relevance tagging kept LLM-driven for manual commits; `enumerate` assigns primary/supporting
    by query similarity for bulk findings.

### Done 2026-05-31 — release-gate quick wins (narrow-answer + denylist)

Go/no-go gate pass. **Release is gated on testing + judgment, not a formal golden key / validation
set** (owner decision 2026-05-31): run representative queries through the UI / `eval_run.py` and
judge output quality directly. Remaining gate items are now just **latency on coverage queries** and
a **judgment pass over the analytical archetype** (got-wrong / open-questions). Two shippable defects
found and fixed this session:

- ✅ **Narrow/temporal answer gap.** Live "did I write about the iPhone in 2007?" dumped 7 all-time
  iPhone quotes with NO verdict — the user had to infer the temporal answer. Root causes: (a) the
  SYSTEM answer-block rule made the block OPTIONAL for narrow queries, (b) the `run_agent` promotion
  net only fired when the report was *empty*, so the quotes suppressed it. Fixed in `agent.py`:
  SYSTEM now REQUIRES a one-/two-sentence verdict block for yes/no / temporal / superlative /
  negative narrow queries (+ a date_range nudge so "in 2007" actually scopes), and the promotion
  net now fires whenever no `answer` block was emitted (tracked via `answer_seen`, not report
  presence). Re-run verdict is now correct: *"No iPhone mention in 2007 … earliest is October 2009
  [p. 505] … later mentions Apple–FBI 2017 [pp. 2430–2435], CHIPS Act 2023 [p. 3908], antitrust
  2024 [pp. 4151, 4215, 4326]."*
- ✅ **Stale `disallowed_tools` vs CLI 2.1.158.** The agent wasted a turn calling `ToolSearch`
  (a new 2.x built-in not in the denylist) before `can_use_tool` denied it. The `can_use_tool`
  allowlist is the real lockdown and held — but the offered-tool denylist now also lists
  `ToolSearch, Task, TodoWrite, ExitPlanMode, BashOutput, KillShell, SlashCommand, MultiEdit`,
  so the model no longer reaches for them. Verified: clean tool path, no ToolSearch.

- ✅ **Analytical archetype VALIDATED (judgment pass, gate item #2).** Ran both analytical golden
  queries live and judged output as the author. Both clear the bar **decisively** — this was the
  highest-risk archetype (not similarity-findable) and it works:
  - *"What did I get wrong"*: 533s / $2.45 / 65 turns. Found real verbatim retractions — Sept-2008
    "we were wrong" [p. 263], the 2009→2011 housing-attribution retraction "maybe the biggest
    estimation miss ever" [pp. 829–843], the decade of wrong rate forecasts (the "oops chart"),
    Bitcoin "wrong on Bitcoin" (2025), Argentina/Milei — plus a correct meta-pattern synthesis.
  - *"Open questions"*: 444s / $1.40 / 46 turns. 9 thematic clusters WITH per-question resolution
    status (⟨unresolved⟩ / ⟨answered by shale boom⟩ / ⟨answered catastrophically in 2022⟩) — exactly
    the resolved-vs-unresolved decomposition the query asked for.
- ✅ **find_mentions auto-commit flood fixed (volume-gated relevance).** The testing surfaced a real
  defect: find_mentions marked EVERY exact-match page `primary`, so phrase-heavy analytical queries
  (10–20 find_mentions calls on fuzzy phrases like "watch for", "we don't know") buried the synthesis
  under hundreds of raw quotes (14 "Exact mentions:" dump headings before the curated sections) and
  even marked off-target pages primary (the iPod page p.505 surfaced as "primary" for an open-questions
  query). Fix in `agent.py`: `FIND_MENTIONS_PRIMARY_MAX=12` — a call returning ≤12 hits (distinctive
  entity) stays `primary`; >12 (common phrase) commits as `supporting` so the U3 UI tucks it behind the
  per-section disclosure. Verified directly (no LLM): iPhone (7)→primary, "remains to be seen" (22) and
  "watch for" (13)→supporting. The answer block was always clean; this declutters the report *body*.
- ⬜ **Remaining gate item: latency/cost on deep coverage queries.** 7–9 min and $1.4–$2.5 per
  analytical run (the shape-aware prompt did NOT tame this; got-wrong is pricier than the $1.68 Bitcoin
  baseline). ~half the tool calls are trailing manual `add_to_report`. Acceptable for once-in-a-while
  deep dives given the UI progress bar, but it's the one open product bottleneck. Also: **0 charts**
  surfaced on either analytical run over a chart-heavy corpus — worth a look (enumerate/chart_only on
  the curated synthesis, or the agent isn't reaching for charts on analytical shapes).

### Open observations from 2026-05-30 stress test (not yet acted on)

Five Cembalest-voice queries through the live UI exposed these still-open issues:

- **Token cost on coverage queries is high.** "Earliest Bitcoin mention + evolution" burned
  ~3M tokens / $1.68 / 51 turns. The 2026-05-30 SYSTEM revision should reduce this by softening
  the "be exhaustive" forcing function, but it's worth re-measuring on the same query.
- **Latency is the real product bottleneck.** Coverage queries can still take multiple minutes
  wall-clock. Most of that is sequential LLM turns, not retrieval. Next optimization target:
  re-measure representative coverage queries after the shape-aware prompt, then reduce avoidable
  turns and parallelize independent tool calls where the SDK/runtime permits.

### UI session — U1–U6 product UI v1 built and verified (2026-05-30)

`static/index.html` consumes the existing two-lane contract (no field renames). Verified in the
in-app browser against live local runs:
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
- ✅ **U4 provenance** — each finding renders a `src` chip; clicking it opens the trace drawer,
  scrolls to the best matching tool event, and flashes it. Matching handles both colon-delimited
  sources (`find_mentions:iPhone`) and loose sources (`search widening`).
- ✅ **U5 export** — `Download PDF` is the only export control retained by owner preference.
  `Copy MD` / `Download MD` were removed. PDF export posts the assembled report to `/export_pdf`
  and returns a valid PDF.
- ✅ **U6 live progress** — hero status shows current tool, running/done state, and a progress bar.
  It advances from tool activity and structured `enumerate` trace fields, then resolves to 100% on
  `done`.
- ✅ **Narrow-query hardening** — if a factual/negative query produces no report-lane events, the
  final assistant prose is promoted into a pinned `answer` block so the report pane never finishes
  empty.

---

## Performance/Recall Pivot

> **Direction change (owner, 2026-05-25):** optimize for **lower latency + exhaustive recall**, and
> shift the product from agent synthesis toward **result enumeration**.

The historical `OPTIMIZATION_PLAN.md` has been deleted because its levers have either shipped or
become stale. Current state:
- ✅ Bulk `enumerate` tool ships and is active: it unions granular topics, dedups pages, ranks by
  query similarity, auto-commits quote/chart findings, and emits structured coverage accounting.
- ✅ Topic enumeration is uncapped except for the `MAX_ENUMERATE=500` safety ceiling.
- ✅ SYSTEM prompt now branches by query shape: NARROW, TOPICAL, ANALYTICAL.
- ✅ Entity guardrail (`must_contain`) prevents broad-topic enumeration from committing pages that
  are conceptually similar but lack the named entity.
- ⬜ Remaining optimization work: measure the same expensive coverage query before/after the prompt
  changes, then reduce avoidable LLM turns and parallelize independent tool calls where possible.

## ★ ACTIVE / NEXT SESSION — latency + testing

1. Re-measure "Earliest Bitcoin mention + evolution" after the 2026-05-30 shape-aware prompt and
   current UI/runtime changes. Record wall-clock, tool-call count, token/cost total, and cited pages.
2. Optimize the remaining avoidable LLM turns without weakening recall. Candidate areas:
   redundant widening searches, unnecessary manual `add_to_report` calls after `enumerate`, and
   independent tool calls that could run concurrently if the SDK/runtime supports it.
3. Run the 5 `golden_queries.txt` queries (+ ad-hoc Cembalest-voice queries) through the UI and/or
   `eval_run.py` and **judge output quality directly** — coverage breadth, citation fidelity, and
   whether the analytical ones (got-wrong, open-questions) actually decompose. Compare against the
   text-only baseline in `baseline/golden_results.md`. No formal scored recall required to ship.
4. (Optional, NOT a gate) If a recall number is later wanted, hand-build a golden key for one query
   with the owner and score via `eval_run.py --golden`.

## PRODUCT VISION — the two-panel report UI (U-series)

> **One collapsible side panel** shows the agent's action trace (its tool calls / reasoning).
> **One main panel** shows the assembled report in markdown/HTML — and makes clear **which
> quotes / charts / syntheses best answer the user's query**, not just a flat arrival-order dump.

The event contract is in place and the U-series UI is implemented. Current event shapes the UI consumes:
- `lane:"report"`, `kind ∈ {heading, quote, chart, narrative, answer}`. `quote`/`chart` carry
  `relevance ∈ {"primary","supporting"}` (defaults to `"supporting"`). `answer` is emitted once
  at the end — a synthesized direct answer with inline `[p. N]` (render it pinned at the top).
  Other fields: `page, issue_date, title, caption, text, src`.
- `lane:"trace"`, `type ∈ {thought, tool_call, tool_result, done, error}`, each with a per-run
  monotonic `id` (use it as the DOM anchor / provenance target).
Do not rename these fields — extend if needed.

### U1 — Layout: report is the hero, trace is a collapsible drawer
✅ **DONE + browser verified.** See "UI session" above for what was built.
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
✅ **DONE + browser verified.** `heading` now opens a collapsible section and
findings nest under it.
Was append-only in arrival order; `heading` events didn't open a
container. The renderer now **nests** `quote/chart/narrative` under the most recent `heading`
so the finished document reads as a grouped report (the SYSTEM prompt already tells the agent to
group by sub-topic / chronology and to emit headings).
- **Verify:** a multi-section query renders collapsible `<section>`s, each with its heading and
  the findings committed under it, in document order.

### U3 — Relevance: surface which findings best answer the query  ★ core of the vision
✅ **DONE + browser verified.**
**Decided (owner, 2026-05-25): agent-tagged relevance + a synthesized answer block.**
- ✅ **BACKEND DONE (agent.py + SYSTEM, verified live):** `add_to_report` takes `relevance`
  (`primary`|`supporting`, default `supporting`) and a `kind:"answer"`; SYSTEM instructs the agent
  to tag findings and to FINISH with one `answer` block (direct answer + inline `[p. N]` to the
  primary findings). A live solar-costs run emitted the answer block + 13 primary / 4 supporting
  tags. *Tuning note:* the agent currently leans toward `primary` (13/17) — if the rendered report
  feels under-differentiated, tighten the "reserve primary for the strongest" wording in SYSTEM.
- ✅ **FRONT-END:** `answer` is pinned at the top of the
  report; `primary` quotes/charts render in full; `supporting` ones are tucked behind a per-section
  "▾ N supporting findings" disclosure; `supporting` charts render as ~240px thumbnails. *Tuning
  note still open:* watch whether primary-vs-supporting reads as under-differentiated in the browser
  (the agent leans `primary`) — may want a stronger visual treatment and/or the SYSTEM-prompt tweak.
- *Optional future complement* (no extra LLM, skip for v1): within each tier, order findings by
  similarity of their text to the query embedding (`static-retrieval-mrl-en-v1`). Weak on
  analytical queries, so never the primary signal.
- **Verify:** the report opens with the pinned `answer`; primary vs. supporting are visually
  distinct; the disclosure expands/collapses supporting findings.

### U4 — Provenance: link a finding back to the trace that surfaced it
✅ **DONE + browser verified.**
- ✅ **BACKEND DONE:** every trace event now carries a per-run monotonic `id` (verified 1..97).
- ✅ **FRONT-END:** render each finding's `src` as a small chip; on click, open the trace drawer
  scrolled to the matching event. Correlate by `src` string (e.g. `"search_topic:solar-pv-costs"`)
  against trace `tool_call` name+args; use the trace `id` as the DOM anchor to scroll to.
- **Verify:** clicking a finding's provenance chip reveals the originating tool call in the drawer.

### U5 — Export the assembled report
✅ **DONE + browser/API verified for PDF export.** Owner preference: keep only `Download PDF`; remove
`Copy MD` and `Download MD`. `/export_pdf` renders a citation-complete PDF from the assembled report.
- **Verify:** `Download PDF` click has no console errors; `/export_pdf` returns a valid PDF.

### U6 — Live progress in the hero panel
✅ **DONE + browser verified.** Surface liveness without opening the trace: a thin status line /
progress bar driven by tool activity and structured `enumerate` trace fields, plus current tool
name and running/done state.
- **Verify:** progress advances during a live run and resolves on `done`.

---

## Human-gated — needs the owner (do NOT fabricate)

- **Golden answer key (OPTIONAL — not a release gate).** A hand-built true answer set would let
  `eval_run.py --golden` print a recall/precision number; it needs corpus-owner knowledge. Per the
  2026-05-31 owner decision, release is judged by testing + eyeballing output, not by this key — so
  it is a nice-to-have for later quantification, not a blocker.

> **SYSTEM prompt is no longer broadly owner-gated.** The 2026-05-30 revision below resolved the
> shape-branching, answer-block, and entity-guardrail questions explicitly. Future SYSTEM tweaks
> are normal engineering — only changes that re-target the agent's core stance (e.g. dropping
> coverage-map mode entirely, switching to deterministic relevance) need owner sign-off.

## Validation by testing + judgment

Run the 5 golden queries (+ ad-hoc Cembalest-voice queries) through the UI; confirm the agent
enumerates topics/issues on exhaustive queries and decomposes the analytical ones (got-wrong,
open-questions). Judge output directly: coverage breadth, citation fidelity (`[p. N]` correctness,
verbatim quotes from content_text), and whether the answer block reads as a useful verdict. Compare
against the text-only baseline in `baseline/golden_results.md`. Scoring against a golden key
(`eval_run.py --golden`) is optional — for a later recall number, not required to ship.

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
- **Claude Agent SDK, not the raw Client SDK.** The SDK **compacts context**, so durable run state
  must live outside the model transcript. Current active coverage accounting comes from structured
  tool events such as `enumerate`, not from model memory.
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
- **Runtime:** `agent.py` (7 eom tools + SDK + lanes; shape-aware SYSTEM), `server.py`,
  `static/index.html`, `eval_run.py` (headless eval).
- **Indexes** (`index/`, amortized): `issues.jsonl`, `cards.jsonl`, `page_topics_raw.jsonl`,
  `page_topics.jsonl`, `topics.json`, `embeddings.npy`, `bm25.pkl`, `pages.json`.
- **Inputs/reference:** `_diag/pages_clean.jsonl` (canonical page data), `golden_queries.txt`,
  `baseline/golden_results.md` (text-only baseline), `review_cards.py` (card QA tool).
- Vision: `gemini-3.5-flash`. Embeddings: `static-retrieval-mrl-en-v1` (`MODEL_NAME`, swappable).
</content>
</invoke>
