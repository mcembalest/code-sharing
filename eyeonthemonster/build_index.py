from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from clean_text import clean_content


ROOT = Path(__file__).resolve().parent
PAGES_PATH = ROOT / "_diag" / "pages_clean.jsonl"
CARDS_PATH = ROOT / "index" / "cards.jsonl"
ISSUES_PATH = ROOT / "index" / "issues.jsonl"
TOPICS_PATH = ROOT / "index" / "topics.json"
PAGE_TOPICS_PATH = ROOT / "index" / "page_topics.jsonl"
INDEX_DIR = ROOT / "index"
MODEL_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def is_true(value: object) -> bool:
    return value is True or str(value).lower() == "true"


def issue_for_page(issues: list[dict], page: int) -> dict | None:
    for issue in issues:
        if int(issue["page_start"]) <= page <= int(issue["page_end"]):
            return issue
    return None


def main() -> None:
    t0 = time.time()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cards = {int(r["page"]): r.get("card_text") or "" for r in load_jsonl(CARDS_PATH)}
    page_topics = {int(r["page"]): r for r in load_jsonl(PAGE_TOPICS_PATH)}
    issues = load_jsonl(ISSUES_PATH)
    topic_labels = {}
    if TOPICS_PATH.exists():
        taxonomy = json.loads(TOPICS_PATH.read_text())
        for parent in taxonomy.get("high_level", []):
            topic_labels[parent["id"]] = parent["label"]
            for granular in parent.get("granular", []):
                topic_labels[granular["id"]] = granular["label"]
    records = []
    texts = []
    for src in load_jsonl(PAGES_PATH):
        if src.get("disclaimer_class") == "skip":
            continue
        page = int(src["page"])
        tags = page_topics.get(page, {"high_level": [], "granular": []})
        granular = list(tags.get("granular", []))
        issue = issue_for_page(issues, page)
        card_text = cards.get(page, "")
        topic_text = " ".join(topic_labels.get(t, t) for t in granular)
        # Strip running-header/footer/disclaimer boilerplate so neither the report quote bodies nor
        # the embeddings carry the J.P. Morgan letterhead + FDIC block that sat on every page.
        content_text = clean_content(src.get("content_text") or "")
        search_text = "\n\n".join(part for part in [content_text, card_text, topic_text] if part)
        records.append(
            {
                "page": page,
                "issue_id": issue.get("issue_id") if issue else None,
                "issue_date": src.get("issue_date"),
                "title": issue.get("title") if issue else None,
                "content_text": content_text,
                "card_text": card_text,
                "is_chart_bearing": is_true(src.get("is_chart_bearing")),
                "high_level": list(tags.get("high_level", [])),
                "granular": granular,
                "search_text": search_text,
            }
        )
        texts.append(search_text)
    tokenized = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized)
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    np.save(INDEX_DIR / "embeddings.npy", emb.astype(np.float32))
    with (INDEX_DIR / "pages.json").open("w") as f:
        json.dump(records, f)
    with (INDEX_DIR / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_len": [len(t) for t in tokenized]}, f)
    print(f"indexed {len(records)} pages | embedding dim {emb.shape[1]} | {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
