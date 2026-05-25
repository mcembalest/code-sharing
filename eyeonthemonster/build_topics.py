from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from scipy.cluster.hierarchy import fcluster, linkage
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
PAGES_PATH = ROOT / "_diag" / "pages_clean.jsonl"
CARDS_PATH = ROOT / "index" / "cards.jsonl"
ISSUES_PATH = ROOT / "index" / "issues.jsonl"
RAW_OUT = ROOT / "index" / "page_topics_raw.jsonl"
TOPICS_OUT = ROOT / "index" / "topics.json"
PAGE_TOPICS_OUT = ROOT / "index" / "page_topics.jsonl"
MODEL = "gemini-3.5-flash"
MODEL_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"

PROMPT = """Extract granular topical tags from these Eye on the Market pages.

For each page, return 3 to 7 short noun phrases. Prefer specific economic,
market, sector, policy, company, country, technology, risk, or asset-allocation
topics that would help route search coverage. Avoid generic phrases like
"chart analysis", "market commentary", or "investment outlook".

Return ONLY valid JSON in this exact shape:
{{"pages":[{{"page":123,"topics":["solar PV costs","European banks"]}}]}}

Pages:
{pages}
"""

_SLUG = re.compile(r"[^a-z0-9]+")
# Each granular cluster is routed to the FIRST-scoring high-level bucket whose keywords appear
# (as whole words / word-prefixes) across the cluster's member phrases. Order is the tie-break
# priority. "other" is the catch-all for clusters that match no keyword.
HIGH_LEVEL_TOPICS = [
    ("energy", "Energy, Power & Climate", ("energy", "energiewende", "oil", "gas", "lng", "shale", "fracking", "hydraulic fracturing", "coal", "solar", "photovoltaic", "wind", "nuclear", "electric", "power", "grid", "battery", "hydrogen", "carbon", "climate", "emissions", "renewable", "biofuel", "decarbon", "lcoe", "utility", "fuel")),
    ("politics-policy", "Politics, Elections & Public Policy", ("politic", "elect", "voting", "congress", "senate", "president", "polarization", "policy", "regulation", "regulatory", "tax", "tariff", "sanction", "immigration", "defense", "military", "foreign policy", "geopolitical", "public-private partnership", "government shutdown")),
    ("fiscal-debt", "Fiscal, Debt & Entitlements", ("debt", "deficit", "fiscal", "entitlement", "medicare", "medicaid", "social security", "pension", "opeb", "municipal", "sovereign", "budget", "cbo", "treasury", "public finance", "underfunded", "unfunded")),
    ("markets-investing", "Markets, Investing & Asset Allocation", ("equity", "s&p", "stock", "bond", "yield", "credit", "spread", "valuation", "p/e", "return", "hedge", "asset allocation", "portfolio", "fund", "etf", "invest", "volatility", "liquidity", "dividend", "buyback", "small cap", "large cap")),
    ("banks-credit", "Banking, Credit & Financial System", ("bank", "lending", "mortgage", "loan", "credit", "default", "leverage", "securit", "financial regulation", "underwriting", "private credit", "monoline", "fannie", "freddie", "gse", "tarp", "talf", "lehman", "volcker", "systemic risk", "basel", "fdic", "deposit")),
    ("macro-growth-inflation", "Macro Growth, Inflation & Rates", ("inflation", "disinflation", "reflation", "cpi", "ppi", "gdp", "growth", "recession", "monetary", "federal reserve", "fed", "interest rate", "rate", "yield curve", "pmi", "purchasing managers", "productivity", "output gap", "stagflation", "deflation", "quantitative easing", "economic leading", "stimulus", "economic recovery", "m2 money", "money supply")),
    ("labor-consumer-housing", "Labor, Consumers & Housing", ("labor", "employment", "unemployment", "wage", "consumer", "household", "housing", "home", "real estate", "commercial real estate", "demographic", "retirement")),
    ("healthcare-pandemic", "Healthcare, Biotech & Pandemic", ("health", "healthcare", "pharma", "drug", "clinical", "vaccine", "covid", "pandemic", "medical", "biotech", "hospital", "aca", "affordable care act", "obamacare")),
    ("technology-innovation", "Technology, AI & Innovation", ("technology", "software", "ai", "artificial intelligence", "machine learning", "llm", "language model", "chatgpt", "neural", "semiconductor", "cyber", "internet", "ipo", "spac", "venture", "innovation", "data center", "crypto", "bitcoin", "cloud", "digital", "robot", "automation")),
    ("global-regions-trade", "Global Regions, Trade & Supply Chains", ("china", "chinese", "euro", "brexit", "japan", "abenomics", "emerging", "country", "trade", "nafta", "supply-chain", "supply chain", "exports", "imports", "currency", "foreign", "cross-border", "msci", "asia", "latin america", "bric")),
    ("corporates-earnings", "Corporates, Earnings & Capital Spending", ("corporate", "earnings", "profit", "ebitda", "free cash flow", "capital spending", "capital expenditure", "capex", "manufacturing", "industrial", "m&a", "merger", "acquisition", "margins", "ism")),
    ("commodities-resources", "Commodities, Food & Natural Resources", ("commodity", "metal", "copper", "gold", "agriculture", "food", "water", "mining", "resource", "crop")),
    ("legal-governance", "Legal, Governance & Disclosures", ("legal", "court", "litigation", "antitrust", "governance", "disclosure", "compliance", "fiduciary")),
    ("other", "Other Corpus Topics", ()),
]

# Keywords match at a word prefix so "bank" catches banks/banking, "loan" catches loans, and
# "elect" catches election/electoral. A few short tokens are too collision-prone for prefix
# matching (e.g. "coal"->"coalition", "ai"->"aid", "gold"->"goldman", "aca"->"academic"), so
# they are pinned to whole-word matches instead.
_WHOLE_WORD = {"coal", "wind", "ai", "home", "aca", "ism", "fed", "gold"}


def _compile_keywords() -> list[tuple[str, list[re.Pattern]]]:
    compiled = []
    for topic_id, _name, keywords in HIGH_LEVEL_TOPICS:
        pats = []
        for kw in keywords:
            kw = kw.lower()
            tail = r"\b" if kw in _WHOLE_WORD else ""
            pats.append(re.compile(rf"\b{re.escape(kw)}{tail}"))
        compiled.append((topic_id, pats))
    return compiled


_KEYWORD_PATTERNS = _compile_keywords()


def is_true(value: object) -> bool:
    return value is True or str(value).lower() == "true"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def kept_pages() -> list[dict]:
    cards = {int(r["page"]): r.get("card_text") or "" for r in load_jsonl(CARDS_PATH)}
    out = []
    for rec in load_jsonl(PAGES_PATH):
        if rec.get("disclaimer_class") == "skip":
            continue
        page = int(rec["page"])
        rec = dict(rec)
        rec["page"] = page
        rec["card_text"] = cards.get(page, "")
        out.append(rec)
    return out


def load_done_raw() -> set[int]:
    done = set()
    for rec in load_jsonl(RAW_OUT):
        if rec.get("topics_raw"):
            done.add(int(rec["page"]))
    return done


def chunked(items: list[dict], n: int) -> list[list[dict]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def compact_page(page: dict) -> dict:
    text = ((page.get("content_text") or "") + "\n\n" + (page.get("card_text") or "")).strip()
    return {"page": int(page["page"]), "text": text[:5000]}


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def extract_batch(client: genai.Client, batch: list[dict]) -> list[dict]:
    prompt = PROMPT.format(pages=json.dumps([compact_page(p) for p in batch], ensure_ascii=False))
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    data = parse_json(resp.text or "{}")
    by_page = {int(p["page"]): p for p in batch}
    out = []
    for item in data.get("pages", []):
        page = int(item.get("page"))
        if page not in by_page:
            continue
        topics = [str(t).strip().lower() for t in item.get("topics", []) if str(t).strip()]
        seen = []
        for t in topics:
            if t not in seen:
                seen.append(t)
        out.append({"page": page, "topics_raw": seen[:7]})
    seen_pages = {r["page"] for r in out}
    for page in by_page:
        if page not in seen_pages:
            out.append({"page": page, "topics_raw": []})
    return out


async def run_extract(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("GEMINI_API_KEY not loaded")
    done = load_done_raw()
    pending = [p for p in kept_pages() if int(p["page"]) not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    batches = chunked(pending, args.batch_size)
    print(f"raw topic pages already present: {len(done)}")
    print(f"pending kept pages selected: {len(pending)} in {len(batches)} batches")
    if args.dry_run:
        return
    client = genai.Client(api_key=api_key)
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()
    queue: asyncio.Queue[tuple[int, list[dict]] | None] = asyncio.Queue()
    for i, batch in enumerate(batches, 1):
        queue.put_nowait((i, batch))

    async def worker(name: int) -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            ordinal, batch = item
            try:
                rows = await asyncio.wait_for(
                    asyncio.to_thread(extract_batch, client, batch),
                    timeout=args.batch_timeout,
                )
                async with lock:
                    with RAW_OUT.open("a") as f:
                        for row in rows:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    pages = f"p.{rows[0]['page']}-p.{rows[-1]['page']}" if rows else "empty"
                    print(f"[{ordinal}/{len(batches)}] worker={name} {pages} -> {len(rows)} pages", flush=True)
            except Exception as exc:
                async with lock:
                    print(f"[{ordinal}/{len(batches)}] worker={name} ERROR: {exc}", file=sys.stderr, flush=True)
            finally:
                queue.task_done()

    tasks = [asyncio.create_task(worker(i + 1)) for i in range(max(1, args.workers))]
    for _ in tasks:
        queue.put_nowait(None)
    t0 = time.time()
    await queue.join()
    await asyncio.gather(*tasks)
    print(f"extract finished in {time.time() - t0:.1f}s")


def slugify(label: str, prefix: str, used: set[str]) -> str:
    base = _SLUG.sub("-", label.lower()).strip("-")[:48] or prefix
    value = base
    i = 2
    while value in used:
        value = f"{base}-{i}"
        i += 1
    used.add(value)
    return value


_HL_ORDER = {topic_id: i for i, (topic_id, _name, _kw) in enumerate(HIGH_LEVEL_TOPICS)}


def _route_text(text: str) -> str:
    text = text.lower()
    scores: dict[str, int] = {}
    for topic_id, pats in _KEYWORD_PATTERNS:
        if topic_id == "other":
            continue
        hits = sum(1 for pat in pats if pat.search(text))
        if hits:
            scores[topic_id] = hits
    if not scores:
        return "other"
    return min(scores, key=lambda t: (-scores[t], _HL_ORDER[t]))


def high_level_for(member_phrases: list[str], medoid: str) -> str:
    # Route on the medoid label first — it is the cluster's cleanest representative, so it
    # avoids a keyword-rich bucket winning on incidental member vocabulary (e.g. a Volcker-rule
    # cluster mentioning the Fed routing to macro instead of banks). Fall back to the full
    # member vocabulary only when the medoid itself matches no keyword (e.g. "germany
    # energiewende" -> energy via its members' power/renewable phrases).
    routed = _route_text(medoid)
    if routed != "other":
        return routed
    return _route_text(" ".join(member_phrases))


def normalize_rows(rows: list[dict], threshold: float) -> tuple[dict, list[dict]]:
    page_phrases = {int(r["page"]): [str(t).lower().strip() for t in r.get("topics_raw", []) if str(t).strip()] for r in rows}
    phrase_counts = Counter(t for topics in page_phrases.values() for t in set(topics))
    phrases = sorted(phrase_counts)
    if not phrases:
        return {"high_level": []}, [{"page": p, "high_level": [], "granular": []} for p in sorted(page_phrases)]
    pages_for_phrase: dict[str, set[int]] = defaultdict(set)
    for page, topics in page_phrases.items():
        for t in set(topics):
            pages_for_phrase[t].add(page)
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(phrases, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    labels = fcluster(linkage(emb, method="average", metric="cosine"), t=1.0 - threshold, criterion="distance")
    clusters: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[int(label)].append(i)
    used_granular: set[str] = set()
    granular = []
    phrase_to_gid = {}
    gid_pages: dict[str, set[int]] = {}
    gid_to_high: dict[str, str] = {}
    for idxs in clusters.values():
        sub = emb[idxs]
        sim = sub @ sub.T
        medoid_idx = idxs[int(np.argmax(sim.mean(axis=1)))]
        label = phrases[medoid_idx]
        gid = slugify(label, "topic", used_granular)
        members = [phrases[i] for i in idxs]
        pages = set().union(*(pages_for_phrase[m] for m in members))
        for i in idxs:
            phrase_to_gid[phrases[i]] = gid
        gid_pages[gid] = pages
        gid_to_high[gid] = high_level_for(members, label)
        granular.append({"id": gid, "label": label, "n_pages": len(pages)})
    names = {topic_id: name for topic_id, name, _keywords in HIGH_LEVEL_TOPICS}
    parents: dict[str, list[dict]] = defaultdict(list)
    for g in granular:
        parents[gid_to_high[g["id"]]].append(g)
    high = []
    for hid, members in parents.items():
        members.sort(key=lambda g: (-int(g["n_pages"]), g["label"]))
        pages = set().union(*(gid_pages[g["id"]] for g in members)) if members else set()
        high.append({
            "id": hid,
            "label": names.get(hid, hid.replace("-", " ").title()),
            "n_pages": len(pages),
            "granular": [{"id": g["id"], "label": g["label"], "n_pages": int(g["n_pages"])} for g in members],
        })
    high.sort(key=lambda h: (_HL_ORDER.get(h["id"], 999), h["label"]))
    page_rows = []
    for page, phrases_for_page in sorted(page_phrases.items()):
        gids = sorted({phrase_to_gid[p] for p in phrases_for_page if p in phrase_to_gid})
        hids = sorted({gid_to_high[g] for g in gids if g in gid_to_high})
        page_rows.append({"page": page, "high_level": hids, "granular": gids})
    return {"high_level": high}, page_rows


def issue_for_page(issues: list[dict], page: int) -> dict | None:
    for issue in issues:
        if int(issue["page_start"]) <= page <= int(issue["page_end"]):
            return issue
    return None


def rollup_issues(page_rows: list[dict]) -> None:
    issues = load_jsonl(ISSUES_PATH)
    by_issue = {r["issue_id"]: {"high_level": set(), "granular": set()} for r in issues}
    for row in page_rows:
        issue = issue_for_page(issues, int(row["page"]))
        if not issue:
            continue
        bucket = by_issue[issue["issue_id"]]
        bucket["high_level"].update(row.get("high_level", []))
        bucket["granular"].update(row.get("granular", []))
    with ISSUES_PATH.open("w") as f:
        for issue in issues:
            tags = by_issue[issue["issue_id"]]
            issue["high_level"] = sorted(tags["high_level"])
            issue["granular"] = sorted(tags["granular"])
            f.write(json.dumps(issue, ensure_ascii=False) + "\n")


def run_normalize(args: argparse.Namespace) -> None:
    rows = load_jsonl(RAW_OUT)
    taxonomy, page_rows = normalize_rows(rows, args.threshold)
    TOPICS_OUT.write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2))
    with PAGE_TOPICS_OUT.open("w") as f:
        for row in page_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rollup_issues(page_rows)
    print(f"wrote {TOPICS_OUT} with {len(taxonomy['high_level'])} high-level topics")
    print(f"wrote {PAGE_TOPICS_OUT} with {len(page_rows)} page rows")
    print(f"updated {ISSUES_PATH} with topic rollups")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract and normalize page topics.")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--normalize-only", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.50, help="Cosine-similarity merge floor for granular clustering (higher = finer/more topics).")
    ap.add_argument("--batch-timeout", type=float, default=90.0, help="Seconds before abandoning one Gemini batch.")
    return ap.parse_args()


async def main() -> None:
    args = parse_args()
    if not args.normalize_only:
        await run_extract(args)
    if not args.extract_only and not args.dry_run:
        run_normalize(args)


if __name__ == "__main__":
    asyncio.run(main())
