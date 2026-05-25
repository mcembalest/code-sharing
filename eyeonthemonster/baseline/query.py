import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).resolve().parent / "_index"
MODEL_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_WS = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


_state = {}


def _load():
    if _state:
        return _state
    emb = np.load(INDEX_DIR / "embeddings.npy")
    with (INDEX_DIR / "page_ids.json").open() as f:
        records = json.load(f)
    with (INDEX_DIR / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)["bm25"]
    model = SentenceTransformer(MODEL_NAME)
    _state.update(emb=emb, records=records, bm25=bm25, model=model)
    return _state


def score(query: str, bm25_weight: float = 0.5) -> np.ndarray:
    s = _load()
    bm = np.asarray(s["bm25"].get_scores(tokenize(query)), dtype=np.float32)
    q = s["model"].encode([query], convert_to_numpy=True)[0]
    qn = np.linalg.norm(q) or 1.0
    cos = (s["emb"] @ (q / qn)).astype(np.float32)
    return bm25_weight * _minmax(bm) + (1.0 - bm25_weight) * _minmax(cos)


def top_k(query: str, limit: int, bm25_weight: float) -> list[dict]:
    s = _load()
    scores = score(query, bm25_weight)
    idx = np.argsort(-scores)[:limit]
    out = []
    for i in idx:
        r = s["records"][int(i)]
        out.append({"score": float(scores[int(i)]), "page": r["page"], "issue_date": r["issue_date"], "content_text": r["content_text"]})
    return out


def format_hits(hits: list[dict]) -> str:
    lines = []
    for n, h in enumerate(hits, 1):
        snippet = _WS.sub(" ", h["content_text"]).strip()[:300]
        date = h["issue_date"] or "unknown date"
        lines.append(f"{n}. score={h['score']:.3f} | {date} [p. {h['page']}] | id=page-{h['page']}\n   {snippet}\n")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--bm25-weight", type=float, default=0.5)
    args = ap.parse_args()
    print(format_hits(top_k(args.query, args.limit, args.bm25_weight)))


if __name__ == "__main__":
    main()
