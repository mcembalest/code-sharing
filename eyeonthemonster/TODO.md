# TODO — open work

The pipeline + agent + UI are built, hardened, and deployed to Modal. For how the whole thing works,
see [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md); for the tuned constants and their justification, see
[`docs/DECISIONS.md`](docs/DECISIONS.md). This file is only what's still open. Completed work lives in
git history, not here.

## Open

1. **Latency / cost on deep coverage queries.** Analytical runs take ~7–9 min and $1.4–$2.5 (e.g.
   got-wrong 533s / $2.45 / 65 turns; open-questions 444s / $1.40 / 46 turns). Most of it is
   sequential LLM turns, not retrieval. Modal `web` timeout is 600s — got-wrong at 533s is near the
   ceiling, so cutting turns also de-risks deploy timeouts. Levers, in order: (a) compose the answer
   block from `enumerate`/`find_mentions` accounting instead of re-committing one `add_report_items`
   per page (≈half the calls on analytical runs are trailing manual commits); (b) drop redundant
   widening `search` passes; (c) parallelize independent tool calls if the SDK permits. Measure each
   change with `uv run python eval_run.py "<query>"` (prints cost/tokens/turns/sdk_wall); the bar is
   fewer turns at equal cited-page coverage.

2. **Charts on analytical queries.** Both analytical golden runs surfaced **0 charts** over a
   chart-heavy corpus. Worth checking whether the agent should reach for `enumerate(chart_only=True)`
   when synthesizing got-wrong / open-questions answers.

3. **Verify the transient-drop resume fix against a real drop.** The session-resume retry
   (`agent.py::run_agent`) is unit-checked but never exercised end-to-end. To force one: temporarily
   `raise CLIConnectionError("socket connection was closed unexpectedly")` once inside `consume()`
   after the first tool turn, run `eval_run.py "<any query>"`, and confirm the `⟳ … resuming` notice
   appears, turns 1–2 are NOT re-streamed, and exactly one `done` arrives. Remove the injection after.

4. **"How this app works" visual (owner goal).** Turn `HOW_IT_WORKS.md` into an on-site diagram
   (offline pipeline → index → runtime query flow → two-lane UI). The doc is written diagram-ready;
   keep it the single accurate source and update it alongside any architecture change.

## Smaller smells (low priority)

- **`other` topic tail** — ~1,246 small idiosyncratic granular clusters land in `other`; only 15
  pages have no named bucket. A light LLM-naming pass over the largest `other` clusters could reclaim
  a few (`pumped hydro storage`→energy, `troubled asset relief program`→banks). Low priority.
- **"equities" morphology** — the `equity` prefix doesn't match the plural "equities"; a few clusters
  route by region instead. Harmless; add an `equit` stem if it ever matters.
- **Issue-boundary spot-checks** — eyeball `index/issues.jsonl` around 2015-09-08, 2018-12-10,
  2026-03-03 (likely legit multi-page special reports, but confirm boundaries).
- **Cost display on the retry path** — the displayed cost is the SDK's `total_cost_usd`, and a
  resumed run's figure covers only the continuation, so it can undercount after a transient drop.
  Rare, cosmetic. (We report only the SDK number — there is no self-computed estimate to fall back on.)

## Optional (not a release gate)

- **Golden answer key.** A hand-built true answer set would let `eval_run.py --golden` print
  recall/precision; it needs corpus-owner knowledge. Release is judged by testing + eyeballing output
  (owner decision 2026-05-31), so this is for a later quantified recall number, not a blocker.
  `calibrate.py` already gives an objective retrieval-quality proxy from the topic tags.

---

## Design decisions — DO NOT silently undo

Deliberate and hard-won; a fresh agent may be tempted to "simplify" them. Don't.

- **Agentic-iterative, not one-pass retrieval.** The agent calls search as a tool and loops.
  Exhaustiveness comes from **enumerable denominators** (the issue table AND the topic index), not
  from telling the model "be thorough." Two denominators on purpose.
- **Claude Agent SDK, not the raw Client SDK.** The SDK **compacts context**, so durable run state
  must live outside the model transcript (structured tool events such as `enumerate`, not model memory).
- **Tool lockdown via `can_use_tool` is mandatory**, and it only fires when `query()` is fed a
  **streaming async-iterable prompt** (see `run_agent`). A plain string prompt silently bypasses the
  lockdown and the agent shells out via Bash. Keep the streaming prompt.
- **Embedding model `static-retrieval-mrl-en-v1` is VALIDATED — keep it.** Benchmarked against
  gte-small/bge-small (`calibrate.py --benchmark`): static ties/beats them on this corpus and is far
  faster. Don't "upgrade" without re-running the benchmark. See `docs/DECISIONS.md`.
- **`RELEVANCE_FLOOR = 0.23` is calibrated, not guessed** (`calibrate.py --calibrate`, Youden's J on
  topic-tag distributions). The old 0.38 sat above the on-topic mean and cut recall to ~44%. Re-run
  the calibration if the model or corpus changes. See `docs/DECISIONS.md` + the chart there.
- **Angle-diverse multi-query fan-out** is the fix for recall misses, NOT a model change: a chart is
  indexed under one framing and ranks far down under a differently-framed query.
- **Topics are a separate TEXT pass over `content_text + card_text`** (`build_topics.py`), NOT folded
  into the vision call. Don't re-add a `topics:` field to the card prompt.
- **Taxonomy: fixed ~14 high-level buckets + medoid-first keyword routing**, granular granularity set
  by the clustering threshold (`--threshold`, sim 0.50 → ~3,933 clusters). Tune the threshold; don't
  hand-merge.
- **The `[p. N]` citation contract is non-negotiable** — every claim the reader sees is cited, and the
  UI turns each citation into a clickable link to the source page + its section.
- **The two-lane event contract** (`lane:"report"` kinds vs `lane:"trace"` types) is the agent↔UI
  seam. Extend it (`relevance`, `kind:"answer"`, trace `id`) — don't rename.
- **Transient-drop retry uses session RESUME, not a fresh re-run** (`agent.py::run_agent`). Resuming
  the captured `session_id` keeps prior tool results in context so the model finishes the interrupted
  turn without replaying turns or re-running search. Keep `done` emitted exactly once across every
  path, and only retry errors that pass `_is_transient`.

## Project context

The PDF is the user's father's work: **Michael Cembalest, J.P. Morgan "Eye on the Market"**, ~20
years. Budget is not a constraint (~$1–2 total for the full vision pass) — optimize for quality and
engineering, not API cost. Two query archetypes: *quantitative-topical* (solar costs, pensions,
polarization — carried by vision cards + topic enumeration) and *analytical* (got-wrong,
open-questions — not similarity-findable; require decomposition).
