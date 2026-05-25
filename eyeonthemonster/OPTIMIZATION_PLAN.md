# Optimization plan — enumeration-first, low-latency recall

**Direction (owner, 2026-05-25):** re-target the product toward **exhaustive enumeration of every
relevant result, fast**. Put **less weight on agent synthesis** (narratives, a written answer essay,
per-finding relevance prose) and **more on result enumeration** (just surface every relevant
page/quote/chart, cited). This doc is the plan; the SYSTEM-prompt parts stay **owner-gated** (see
TODO.md → Human-gated) and are called out below.

## Diagnosis — grounded in a live run (solar-cost query, 2026-05-25)

Measured from the trace of one full run (`static/index.html` trace lane):

- **139 eom tool calls total**, of which:
  - **67 `add_to_report` (48%)** — the agent commits **one finding per LLM round-trip**.
  - **41 page reads** (`get_page` 25, `get_issue` 12, `get_pages` 4) — reading individual pages to
    decide relevance and extract quotes.
  - 12 `coverage_status` (polling), 12 retrieval (`search_topic` 8, `semantic` 2, `keyword` 2),
    7 scoping (`list_topics` 2, `mark_inspected` 5).
- So **~108 of 139 calls (78%) are the agent manually enumerating** — read a page, decide, commit,
  repeat — each a `claude-haiku-4-5` round-trip. **That is the latency.** (Observed wall-clock for
  the run: ~5 min.) Tool execution itself is local and fast; the cost is the *number of LLM turns*.
- **Synthesis load:** of the 67 commits — 19 `heading`, 43 `quote`, **3 `narrative`, 2 `answer`**
  (the SYSTEM prompt mandates exactly one answer; it emitted two). Plus per-finding `relevance`
  tagging. This is the layer to shrink.
- **Recall ceiling (bug-class):** `search_topic` reports a topic's true `total` but returns only
  `pages[:MAX_LIST_LIMIT]` = **top 10** (`agent.py:309`). `search_semantic`/`search_keyword` cap at
  `MAX_SEARCH_LIMIT` = 10 too. **A topic with 50 tagged pages only ever shows the agent 10** —
  exhaustive recall is impossible through these tools as written.

**Conclusion:** the two asks are one fix. Enumeration is *deterministic* (the topic index already
knows every tagged page); doing it through per-item LLM turns is what makes the system both slow and
synthesis-heavy. Move enumeration out of the LLM loop.

## Strategy — the agent scopes; the server enumerates

Keep the agent for **decomposition and scope selection** (which the analytical queries genuinely
need — see *Preserve* below). Take **enumeration and per-finding commits** away from it and do them
deterministically server-side. The agent's job shrinks to: decompose query → pick topic ids +
date range → call **one bulk enumeration tool** → (optionally) a short framing line.

### Lever 1 — Server-side bulk enumeration  ★ biggest latency + recall win
Add a tool (e.g. `enumerate`) that takes `topic_ids: list`, `date_range`, `chart_only`, and on the
**server**:
1. unions all pages tagged with any of `topic_ids`, dedups by page number,
2. ranks them (default: chronological; option: by query-embedding similarity — see Lever 3),
3. **emits every one to the `report` lane in a single tool call** as `quote`/`chart` events
   (page, issue_date, title, snippet/caption, `src="enumerate:<topic_id>"`) — **no per-page LLM
   turn**.

This collapses the ~108 read+commit round-trips into **one call per topic group** (a handful).
The UI already renders `quote`/`chart` report events; it just receives a burst instead of a drip.
*Contract note:* this is an **extension** (a new tool + bulk emission), not a rename — consistent
with TODO.md's two-lane rule.

### Lever 2 — Remove the recall cap on the enumeration path
`enumerate` commits the **full** tagged set (guard with a high cap, e.g. 500, configurable), not
top-10. Keep the small 10-cap only on the *exploratory* `search_semantic`/`search_keyword` the agent
reads while widening — there, small is correct. Recall on the enumeration path then equals the topic
index itself: **exhaustive by construction** over the chosen topics ∪ date range.

### Lever 3 — De-emphasize synthesis  ⚠️ touches SYSTEM (owner-gated)
- **Drop the mandatory answer essay** (`agent.py:51–53`) → make it an optional one- or two-sentence
  header, or remove. Drop the `narrative` connective tissue from the default path.
- **Replace per-finding LLM `relevance` tagging** (`agent.py:46–48`) with **deterministic ranking**:
  cosine of each page's embedding to the query embedding (`static-retrieval-mrl-en-v1`, already
  loaded). Top-k per group = `primary`, the rest = `supporting`. No LLM relevance calls. (This is the
  "optional future complement" already noted in TODO U3 — promote it to the default.)
- **Generate headings server-side**, one per topic group (use the topic label), so the agent isn't
  spending turns on structure either.
- Net: the agent makes ~10–20 turns (decompose → list_topics → a few `enumerate` calls) instead of
  ~139. Latency should drop by roughly the same ratio.

### Lever 4 — Latency mechanics (secondary, after 1–3)
- The turn-count reduction from Levers 1–3 is the headline. Then:
- **Stop polling `coverage_status` 12×** — with bulk enumeration the server knows exactly which
  issues the enumerated set covers; emit coverage once, derived, at the end (also feeds U6).
- Allow independent tool calls to run in parallel where the agent issues them together.
- Keep `claude-haiku-4-5` (already the fast model); the win is fewer turns, not a smaller model.

## Preserve — do NOT undo (per TODO.md → Design decisions)

- **Agentic-iterative over two denominators stays.** We are removing the *per-item LLM tax*, not the
  loop: the agent still drives scope over the issue table AND the topic index. This is **not** a
  reversion to one-pass retrieval — it's bulk enumeration *within* an agent-chosen scope.
- **Analytical queries still need synthesis.** "What did I get wrong," "what open questions did I
  pose" are not answerable by enumeration alone (TODO.md → Two query archetypes). Keep the
  read+reason path available; let the agent pick **enumeration-first (fast path)** for
  quantitative/topical queries and the **synthesis path** for analytical ones. Default to
  enumeration; don't delete the synthesis tools.
- **`[p. N]` citation contract** and the **two-lane event contract** (extend, never rename).
- The embedding model as a swappable axis (`MODEL_NAME`).

## Trade-off to confirm with owner

Enumeration-first makes quantitative/topical queries fast and exhaustive, but a pure enumeration
dump reads as a *list*, not an *answer*. For analytical archetypes that's a regression. Proposed
resolution: **branch by query type** — enumeration is the default and the fast path; the agent opts
into light synthesis only when the query is analytical. Owner to confirm this branching vs. "always
enumerate, never synthesize."

## Sequencing & verification

1. **Lever 2 first** (trivial, no prompt change): give the enumeration path the full tagged set.
   *Verify:* a topic with >10 tagged pages enumerates all of them.
2. **Lever 1** (`enumerate` tool, bulk report-lane emission). *Verify:* a solar-cost query commits
   the full topic set in a handful of tool calls; report-lane event count ≫ tool-call count.
3. **Lever 3** (owner-gated SYSTEM rewrite + deterministic ranking). *Verify:* tool-call count per
   run drops from ~139 to ~10–20; wall-clock drops proportionally; primary/supporting split is
   produced server-side without `relevance` LLM calls.
4. **Lever 4** (coverage once, parallelism). *Verify:* `coverage_status` called ≤1×; U6 progress
   still advances.
5. Re-run the eval harness (`eval_run.py`) and the golden queries; confirm cited-page recall is
   **≥** the current run on topical queries and unchanged on analytical ones. Compare wall-clock.

## What's already done toward this

- ✅ The `in_date_range` bug fix (this session) — date-scoped search now actually returns hits;
  without it the enumeration path would still be empty. (`agent.py`, `_norm_date` + `in_date_range`.)
- ✅ **Lever 1 + Lever 2 built & verified (2026-05-25, owner picked "Levers 1+2 now").**
  - New `enumerate` tool (`agent.py`): unions `topic_ids`, dedups by page, ranks by cosine of each
    page to `q` (top `primary_k` → `primary`, rest → `supporting`), emits one server-side `heading`
    + every page as a `chart` (if chart-bearing) or `quote` to the report lane in a **single call**,
    and returns a tiny `{committed, primary}` ack so the agent's context stays lean. Uncapped except
    a `MAX_ENUMERATE=500` safety ceiling. `_load` gained `row_by_page` (page→embedding row) and
    `label_by_topic`. `search_topic`'s description/return now honestly mark it a **preview** (top-10
    sample + true total) and point to `enumerate` for the full set.
  - **Headless verify:** 4 solar topics (60 tagged) → **41 deduped pages committed in one call**
    (19 charts + 22 quotes + 1 heading); ranking put pp.1680/1682 (the known-correct solar-cost
    pages) at #1–2 as `primary`; 6 primary / 35 supporting.
  - **UI verify:** an enumerate burst renders correctly — primary findings in full, supporting tucked
    behind the disclosure, charts as `/page_image` figures, `[p. N]` on every item. (Also closes the
    long-open **N2** chart-render check.)
  - **NOT yet active in a live agent run** — the agent keeps using its old `search_topic` +
    per-page-`add_to_report` flow until the SYSTEM prompt routes it onto `enumerate` (Lever 3 below).

## Lever 3 — exact owner-gated change to ACTIVATE the above

The capability is built; flipping the agent onto it is a SYSTEM-prompt edit (`agent.py` `SYSTEM`),
which is owner-gated. Minimal version (replaces the per-page enumeration instruction at step 3):

> 3. **ENUMERATE in bulk:** once you've chosen the relevant granular topic ids, call `enumerate`
>    with all of them at once (pass the user's question as `q`). It commits *every* tagged page to
>    the report, ranked, in one call — do NOT loop `search_topic` + `add_to_report` page by page.
>    Use `search_topic` only to preview a topic's size before enumerating.

And, for the **branch-by-query-type** decision (owner picked this): add a line up top —

> If the question is quantitative/topical ("everything about X", costs, trends), make `enumerate`
> the backbone and keep prose to a one-line framing. If it is analytical ("what did I get wrong",
> "what open questions did I pose"), use the read+reason path (get_page/get_pages + narrative) since
> enumeration alone won't answer it.

Also trim: drop the mandatory `answer` essay to optional/short, and stop the per-finding `relevance`
instruction (enumerate now assigns relevance by similarity). Owner to review wording before it lands.
