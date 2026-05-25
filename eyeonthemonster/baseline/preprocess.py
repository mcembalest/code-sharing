import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "_diag" / "pages_clean.jsonl"
INDEX_DIR = Path(__file__).resolve().parent / "_index"
MODEL_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def main() -> None:
    t0 = time.time()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    with SRC.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("disclaimer_class") == "skip":
                continue
            records.append(
                {
                    "page": r["page"],
                    "issue_date": r.get("issue_date"),
                    "content_text": r.get("content_text") or "",
                }
            )

    tokenized = [tokenize(r["content_text"]) for r in records]
    bm25 = BM25Okapi(tokenized)

    model = SentenceTransformer(MODEL_NAME)
    texts = [r["content_text"] for r in records]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = (emb / norms).astype(np.float32)

    np.save(INDEX_DIR / "embeddings.npy", emb)
    with (INDEX_DIR / "page_ids.json").open("w") as f:
        json.dump(records, f)
    with (INDEX_DIR / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_len": [len(t) for t in tokenized]}, f)

    elapsed = time.time() - t0
    print(f"indexed {len(records)} pages | embedding dim {emb.shape[1]} | {elapsed:.1f}s")


if __name__ == "__main__":
    main()
