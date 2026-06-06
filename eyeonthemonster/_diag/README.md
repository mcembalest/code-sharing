# `_diag/` — source-data pipeline (NOT throwaway diagnostics)

Despite the name, this folder holds the **first stage of the index pipeline**: turning the raw PDF
into the canonical per-page records that every `build_*.py` script then consumes. Don't delete it
expecting it to be scratch — `build_issues.py`, `build_cards.py`, `build_topics.py`, and
`build_index.py` all read `_diag/pages_clean.jsonl`.

## The chain (all offline — no API, no cost)

```
Eye on the Monster.pdf
   └─ scan.py              → _diag/pages.jsonl           (PyMuPDF text extraction, per page)
        └─ enrich.py       → _diag/pages_enriched.jsonl  (date propagation, is_chart_bearing,
        │                                                  position_in_issue, disclaimer_class)
        └─ strip_disclaimers.py → _diag/pages_clean.jsonl (disclaimer/footer stripping)
                                   = 5,117 records, the canonical input to build_*.py
```

Regenerate from scratch any time (deterministic, ~1–2 min):

```bash
uv run python _diag/scan.py && uv run python _diag/enrich.py && uv run python _diag/strip_disclaimers.py
```

## What's tracked vs. regenerable

- **Tracked (in git):** the three scripts above — the durable source of the pipeline.
- **Gitignored (regenerable, large):** `pages.jsonl`, `pages_enriched.jsonl`, `pages_clean.jsonl`.
  These are intermediate data; recreate them with the command above rather than committing ~80 MB.

The genuinely diagnostic one-offs that used to live here (sampling/prototype scripts) were removed —
`build_cards.py` is the production successor to the old `gemini_sample.py` card prototype.
