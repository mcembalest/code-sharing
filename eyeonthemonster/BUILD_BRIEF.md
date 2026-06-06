# Build Brief — Agentic Search + Live Trace UI

Self-contained handoff. Implements the confirmed direction: an iterative agent that
calls search as a tool, instructed to be exhaustive, with coverage grounded in an
enumerable issue table; plus a browser UI that renders the agent's actions as markdown
in realtime.

Honor the existing structural work — do **not** redo date propagation or disclaimer
stripping; consume their output.

## Output contract (non-negotiable)

Every cited source includes a page citation from this PDF: `[p. N]` or `[pp. N-M]`.
Title + date desirable: `Oil Boil, March 14, 2009 [p. 3456]`.

## Existing assets to reuse

- `_diag/pages_clean.jsonl` — **5117 records**, one per page. Fields:
  `page` (int-as-str), `issue_date` (propagated; 5113/5117 populated), `content_text`
  (disclaimer-stripped), `is_chart_bearing` (bool-as-str), `position_in_issue`,
  `disclaimer_class` (`none` | `strip_footer` | `skip`).
  Drop `disclaimer_class == "skip"` (235 pages) when indexing.
- `_diag/gemini_sample.py` — working Gemini 3.5 Flash card harness; **scale this up** in Step 2.
- `baseline/preprocess.py`, `baseline/query.py` — working BM25 + static-embedding index
  (`static-retrieval-mrl-en-v1`). Adapt, don't rewrite.
- `.env` — `GEMINI_API_KEY` (project root; never log it; already gitignored).
- Vision model: `gemini-3.5-flash` via `google-genai`. Embeddings:
  `sentence-transformers/static-retrieval-mrl-en-v1` (keep swappable via a `MODEL_NAME` const).

Use `uv` for everything. New deps: `fastapi`, `uvicorn`, `claude-agent-sdk` (the Agent
SDK — runs the tool loop + context management; NOT the raw `anthropic` Client SDK).

---

## Step 1 — Issue table (no API spend)

Derive contiguous issues from `pages_clean.jsonl`. Boundary = `position_in_issue == 1`
(fall back to a change in `issue_date` if that field is unreliable). This table is the
**coverage denominator** the agent counts against.

Write `index/issues.jsonl`, one record per issue:

```json
{"issue_id": "issue-0042", "issue_date": "2014-10-28", "title": "...",
 "page_start": 1665, "page_end": 1690, "n_pages": 26}
```

`title`: first non-empty, non-boilerplate line of the first page's `content_text`
(strip the "EYE ON THE MARKET • … • <date>" header). Best-effort is fine.

Sanity-check: print issue count, pages-per-issue distribution, and any issue spanning
> 60 pages (likely a missed boundary) for manual review.

---

## Step 2 — Bulk vision cards (~$1–2, one-time)

Run `gemini-3.5-flash` over the **~2,300 chart-bearing kept pages**
(`is_chart_bearing == "True"` and `disclaimer_class != "skip"`). Reuse the prompt from
`_diag/gemini_sample.py` (it already produces good chart descriptions — see
`_diag/sample_cards.md`).

Write `index/cards.jsonl`: `{"page": 228, "card_text": "..."}`.

Requirements:
- **Idempotent / resumable**: skip pages already in `cards.jsonl`. A crash at page 1500
  must not redo 1–1499.
- **Concurrency**: bounded parallelism (e.g. 8 workers) — this is the slow step, not the
  expensive one.
- Render pages with PyMuPDF (`fitz`) at ~150 DPI; reuse the render code in the sample script.
- Log progress (page N of 2300). Do not print the API key.

---

## Step 2b — Topics (two-pass hybrid, pre-computed per page + issue)

A discrete, **enumerable, normalized** topic layer: a second coverage denominator
alongside the issue table, and the thing that collapses 20-year vocabulary drift at index
time ("solar cost" 2005 and "module ASP / LCOE" 2025 → one canonical granular topic).

**Pass A — extract (free-form, text-only; runs AFTER Step 2 cards).** Per kept page, emit
3–7 short granular topic phrases from one cheap `gemini-3.5-flash` **text** pass over
`content_text + card_text`, **batched** (~15 pages/call). `card_text` already renders the
chart content as prose (numbers + entities), so chart pages need **no image re-visit** —
all kept pages are treated uniformly. (Supersedes the earlier "fold topics into the vision
call" idea: `build_cards.py` keeps card output as clean prose, and topics ride on
`card_text` here instead — no doubled vision cost.)
Write `index/page_topics_raw.jsonl`: `{"page": 228, "topics_raw": ["solar PV cost", ...]}`.

**Pass B — normalize (this is what makes topics enumerable).**
1. Collect all distinct raw phrases; embed them with the SAME static model (`MODEL_NAME`).
2. Cluster by cosine (agglomerative / threshold union-find) → canonical **granular**
   topics; label each cluster by its medoid phrase (or one cheap LLM call to name clusters).
3. Cluster the granular topics again into ~10–20 **high-level** parents (or one LLM pass).
Write `index/topics.json` (the taxonomy):
`{high_level: [{id, label, granular: [{id, label, n_pages}]}]}`.
Rewrite tags → `index/page_topics.jsonl`:
`{"page": 228, "high_level": ["energy"], "granular": ["solar-pv-cost", "wind-subsidy-cycle"]}`.

**Issue rollup:** union each issue's page tags → add `high_level`/`granular` to the issue
records. Gives topic-scoped issue routing (esp. for the analytical queries).

> Why two passes: free-form alone won't *join* (one idea gets five spellings → not
> countable); a fixed taxonomy up front is brittle for a 20-year corpus. Extract free, then
> normalize via the embedding model we already have. Re-runnable; the cluster threshold is
> the one knob to tune.

---

## Step 3 — Merged re-index

Adapt `baseline/preprocess.py`. Searchable surface per page is **merged**:
`search_text = content_text + "\n\n" + card_text + "\n\n" + " ".join(granular topic labels)`
(card/topic parts empty where absent). Page stays the citation unit.

Outputs in `index/`:
- `embeddings.npy` — normalized static embeddings over `search_text`
- `bm25.pkl` — BM25Okapi over tokenized `search_text`
- `pages.json` — list of `{page, issue_id, issue_date, title, content_text, card_text,
  is_chart_bearing, high_level, granular}` aligned by row to the embeddings/bm25 order

Keep `MODEL_NAME` a module constant so the embedding model stays a swappable axis.

---

## Step 4 — Agent (Claude Agent SDK)

Use the **Claude Agent SDK** (`claude-agent-sdk`), not the raw Client SDK. `query()`
runs the tool loop + context management for us; we supply custom in-process tools and a
system prompt, and lock it down so it does **not** behave like a coding agent.

### Tools — custom in-process MCP tools

Define each with `@tool` and bundle via `create_sdk_mcp_server`. All search hits:
`{page, issue_id, issue_date, title, score, snippet}` (snippet ≤ 300 chars). Load
embeddings/bm25/model **once** at module import (reuse the `_load()` singleton in
`baseline/query.py`), not per call.

| Tool | Args | Notes |
|---|---|---|
| `search_keyword` | `q, date_range?, chart_only?, limit=20` | BM25 over merged text |
| `search_semantic` | `q, date_range?, chart_only?, limit=20` | embeddings over merged text |
| `list_issues` | `date_range?` | from `issues.jsonl` — coverage denominator (by date) |
| `list_topics` | `date_range?` | the topic taxonomy + per-topic page counts — coverage denominator (by topic) |
| `search_topic` | `topic_id, date_range?` | ALL pages tagged a canonical topic — the enumerable set for "everything about X" |
| `get_issue` | `issue_id` | full pages+cards for one issue |
| `get_page` | `page` | one merged page record |
| `get_pages` | `start, end` | inclusive span |
| `add_to_report` | `kind, **fields` | **report sink** — `kind` ∈ `quote`/`chart`/`heading`/`narrative`. The agent calls this as it *confirms* findings; the accumulation **is** the report (no final dump). |
| `mark_inspected` | `issue_ids` | **externalized coverage** — record issues examined |
| `coverage_status` | `date_range?, topic_id?` | returns `{inspected, total, remaining:[ids]}` for the scope (by date and/or topic) |

`date_range` = `["YYYY-MM-DD","YYYY-MM-DD"]` or null. `chart_only` filters
`is_chart_bearing`. `search_keyword`/`search_semantic` also accept an optional `topic_id`
filter.

> **Why `mark_inspected` / `coverage_status` exist (the SDK gotcha):** the Agent SDK
> manages context *by compaction*. If earlier `tool_result`s get compacted away, the agent
> forgets which issues it already saw — silently breaking the coverage guarantee. So
> coverage state lives in **durable process memory behind a tool**, never in the transcript.
> The agent closes the loop by polling `coverage_status` until `remaining == []`, not by
> "feeling done."

### Wiring

```python
from claude_agent_sdk import tool, create_sdk_mcp_server, query, ClaudeAgentOptions

@tool("search_semantic", "Semantic search over merged page text + chart cards",
      {"q": str, "date_range": list, "chart_only": bool, "limit": int})
async def search_semantic(args):
    hits = semantic(args["q"], args.get("date_range"), args.get("chart_only", False),
                    args.get("limit", 20))
    await emit({"lane": "trace", "type": "tool_result", "name": "search_semantic",
                "summary": f"{len(hits)} hits: " + ", ".join(f'p.{h[\"page\"]}' for h in hits[:8])})
    return {"content": [{"type": "text", "text": json.dumps(hits)}]}

@tool("add_to_report", "Commit a confirmed finding to the live report",
      {"kind": str, "text": str, "page": int, "issue_date": str, "title": str,
       "caption": str, "src": str})
async def add_to_report(args):
    await emit({"lane": "report", **{k: v for k, v in args.items() if v is not None}})
    return {"content": [{"type": "text", "text": "added"}]}

eom = create_sdk_mcp_server(name="eom", version="1.0.0",
        tools=[search_keyword, search_semantic, list_issues, get_issue,
               get_page, get_pages, add_to_report, mark_inspected, coverage_status])

options = ClaudeAgentOptions(
    model="claude-haiku-4-5-20251001",          # fast/cheap search agent; swappable
    system_prompt=SYSTEM,                        # the lever — see TODO.md
    mcp_servers={"eom": eom},
    allowed_tools=[f"mcp__eom__{t.name}" for t in TOOLS],   # all eom tools
    disallowed_tools=["Bash","Read","Write","Edit","Glob",  # built-ins stay loaded —
                      "Grep","WebSearch","WebFetch","NotebookEdit"],  # block them
    can_use_tool=only_eom,                       # HARD allowlist, enforced at execution
    setting_sources=[],                          # don't load .claude config; not Claude Code
)
```

> **Lockdown is NOT just `allowed_tools` + `setting_sources`.** `allowed_tools` only governs
> *auto-approval*; the SDK's built-in tools (Bash, Read, Grep, …) remain loaded and the agent
> WILL shell out — e.g. grep the SDK's own on-disk tool-result cache — if they're reachable.
> Enforce a real allowlist with a `can_use_tool` callback that returns `PermissionResultAllow`
> for `mcp__eom__*` and `PermissionResultDeny` otherwise (plus `disallowed_tools` as backup).
> Without this the whole enumerable-coverage design is bypassed.

The SDK handles prompt caching of the static system+tools prefix — no manual `cache_control`.

### Two lanes over one stream

Every event carries a `lane`; one stream feeds both UI panes (and coverage analysis):

- **`lane:"trace"`** (machinery / eval substrate): `thought` (from `TextBlock`s in the
  message stream), `tool_call` + `tool_result` (from the tools, via `emit`).
- **`lane:"report"`** (the report assembling live, from `add_to_report`):
  `{"lane":"report","kind":"quote","text":"...","page":1670,"issue_date":"2014-10-28","title":"...","src":"tr-7"}`
  `{"lane":"report","kind":"chart","page":228,"caption":"Wind additions grind to a halt...","src":"tr-7"}`
  plus `heading` / `narrative`. `src` links a fragment back to the trace tool_result that
  produced it. **No final `answer` block** — the report *is* the accumulation.

Bridge the SDK stream + tool emissions through one `asyncio.Queue`. The tools `emit`
(push results/report fragments); a driver task runs `query()` and pushes `thought`s from
`AssistantMessage` `TextBlock`s, then a `done` sentinel on `ResultMessage`:

```python
import asyncio, json
from claude_agent_sdk import query, AssistantMessage, TextBlock, ResultMessage

def run_agent(prompt: str):                      # async generator of lane-tagged events
    q = asyncio.Queue()
    async def emit(ev): await q.put(ev)          # tools close over this
    async def drive():
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        await q.put({"lane": "trace", "type": "thought", "text": b.text})
            elif isinstance(msg, ResultMessage):
                await q.put({"lane": "trace", "type": "done"})
    async def gen():
        task = asyncio.create_task(drive())
        while True:
            ev = await q.get()
            yield ev
            if ev.get("type") == "done": break
        await task
    return gen()
```

The system prompt must tell the agent: decompose; **map the query to canonical topics**
via `list_topics`, pull the tagged set with `search_topic` (the enumerable backbone for
"everything about X"), then widen with `search_semantic`/`search_keyword` to catch
stragglers the tagging missed; follow cross-references; **commit findings live** via
`add_to_report` (with `src` = the originating tool_result); and **be exhaustive by
enumeration** — `mark_inspected` as you go, poll `coverage_status` (scoped by topic and/or
date) until `remaining == []` before finishing. Citations are `[p. N]`.

---

## Step 5 — Live trace UI (SSE)

`server.py` (FastAPI). Loads indexes once at startup; streams agent events as SSE.

```python
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from agent import run_agent  # the async generator from Step 4

app = FastAPI()

@app.get("/")
def index():
    return HTMLResponse(open("static/index.html").read())

@app.get("/run")
async def run(q: str):
    async def stream():
        async for event in run_agent(q):       # async: SDK query() is async
            yield f"data: {json.dumps(event)}\n\n"
        yield "event: done\ndata: {}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})  # disable proxy buffering

@app.get("/page_image/{n}")
def page_image(n: int):
    import fitz                                  # render the chart page on demand
    doc = fitz.open("Eye on the Monster.pdf")    # open once at startup in real code
    pix = doc[n - 1].get_pixmap(dpi=130)         # pages are 1-indexed in our citations
    return Response(pix.tobytes("png"), media_type="image/png")
```

Run: `uv run uvicorn server:app --reload`. (Add `Response` to the fastapi imports;
open the PDF once at module load, not per request.)

`static/index.html` — **two panes over one stream**: the report assembles on the left,
the debug trace scrolls on the right (collapsible). Events route by `lane`. No build step;
`marked` from CDN; each fragment renders as a complete markdown block (no half-parsed flicker).

```html
<!doctype html><meta charset="utf-8"><title>Eye on the Monster</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  body{font:14px/1.5 system-ui;margin:0}
  #bar{padding:.6rem 1rem;border-bottom:1px solid #ddd}
  #cols{display:grid;grid-template-columns:1fr 380px;height:calc(100vh - 52px)}
  #report{padding:1.5rem 2rem;overflow:auto;max-width:760px}
  #trace{border-left:1px solid #eee;background:#fafafa;padding:1rem;overflow:auto;font-size:12px}
  blockquote{border-left:3px solid #7c3aed;margin:.6rem 0;padding:.2rem .9rem;color:#333}
  blockquote .cite{display:block;color:#7c3aed;font-size:12px;margin-top:.3rem}
  figure{margin:1rem 0}figure img{max-width:100%;border:1px solid #ddd}
  figcaption{font-size:12px;color:#666;margin-top:.3rem}
  .ev{border-left:3px solid #ddd;padding:.3rem .6rem;margin:.4rem 0}
  .tool_call{border-color:#3b82f6}.tool_result{border-color:#16a34a}.thought{color:#888}
</style>
<div id="bar">
  <input id="q" size="70" placeholder="what is everything I've ever said about…">
  <button onclick="go()">Run</button>
</div>
<div id="cols"><div id="report"></div><div id="trace"></div></div>
<script>
const $=id=>document.getElementById(id);
function append(pane,html,cls){const d=document.createElement('div');if(cls)d.className=cls;
  d.innerHTML=html;$(pane).appendChild(d);d.scrollIntoView({block:'end'});}
function go(){
  $('report').innerHTML='';$('trace').innerHTML='';
  const es=new EventSource('/run?q='+encodeURIComponent($('q').value));
  es.onmessage=e=>{
    const ev=JSON.parse(e.data);
    if(ev.lane==='report'){
      if(ev.kind==='heading')   append('report',marked.parse('## '+ev.text));
      else if(ev.kind==='narrative') append('report',marked.parse(ev.text));
      else if(ev.kind==='quote') append('report',
        '<blockquote>'+marked.parse(ev.text)+
        '<span class="cite">'+(ev.title||'')+(ev.issue_date?', '+ev.issue_date:'')+
        ' [p. '+ev.page+']</span></blockquote>');
      else if(ev.kind==='chart') append('report',
        '<figure><img src="/page_image/'+ev.page+'">'+
        '<figcaption>'+(ev.caption||'')+' [p. '+ev.page+']</figcaption></figure>');
    } else {  // lane === "trace"
      let md = ev.type==='tool_call'   ? '**→ '+ev.name+'** `'+JSON.stringify(ev.args)+'`'
             : ev.type==='tool_result' ? '**← '+ev.name+'** — '+ev.summary
             : ev.text||'';
      if(md) append('trace',marked.parse(md),'ev '+ev.type);
    }
  };
  es.addEventListener('done',()=>es.close());
}
</script>
```

> **Tier 0 fallback** (validate the loop before any UI): have `run_agent` also append each
> event as a markdown block to `trace.md` and open it in VS Code's markdown preview. Same
> event stream, zero UI code.

---

## Validation

1. Run the 5 golden queries (`golden_queries.txt`) through the agent; eyeball traces in
   the UI — confirm it enumerates issues on the exhaustive ones and decomposes the
   analytical ones (got-wrong, open-questions).
2. **Measure recall**: hand-build a true answer set for ONE exhaustive query (the corpus
   owner knows it) and score the agent's cited pages against it. This settles whether
   "instructed-exhaustive" actually achieves coverage — turn the design assumption into a
   number.

## Build order & spend

1 (free) → 2 (~$1–2, slow not expensive) → 3 (free, ~minutes) → 4 → 5.
Cost is not the constraint; engineering is.
