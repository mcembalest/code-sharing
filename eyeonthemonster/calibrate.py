"""Scrutinize the retrieval pipeline's two model-dependent magic numbers — the embedding MODEL and
the RELEVANCE_FLOOR — against evidence instead of guesswork.

Ground truth without a hand-labeled golden set: the granular topic tags. A page tagged with topic T
is, by construction, about T's label; a page not tagged T is (mostly) not. So for each topic we have
positives (tagged pages) and negatives (random untagged pages), and a natural query (the topic
label). That lets us measure, objectively and at scale:

  --benchmark   ROC-AUC of (on-topic vs off-topic) cosine, per candidate embedding model. The model
                that best separates relevant from irrelevant pages wins. Picks MODEL_NAME.

  --calibrate   Using the deployed embeddings.npy, build the on-topic vs off-topic cosine
                distributions, plot them, and set RELEVANCE_FLOOR at the threshold that best
                separates the two (Youden's J / max-F1), with the chart saved for the docs.

Run:  uv run python calibrate.py --benchmark
      uv run python calibrate.py --calibrate            # after re-embedding with the chosen model
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index"
DOCS = ROOT / "docs"
SEED = 13
# Topics in a sane size band: big enough to have stable positives, small enough that the label is a
# specific concept (a 1300-page mega-bucket label is too generic to be a meaningful "query").
TOPIC_MIN_PAGES = 12
TOPIC_MAX_PAGES = 250
N_TOPICS = 80
POS_PER_TOPIC = 20
NEG_PER_TOPIC = 40

CANDIDATES = {
    "static-retrieval-mrl-en-v1": "sentence-transformers/static-retrieval-mrl-en-v1",
    "gte-small": "thenlper/gte-small",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}
# bge is asymmetric: queries need an instruction prefix, documents do not. Everything else is symmetric.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_corpus() -> tuple[list[dict], dict[str, str]]:
    pages = json.loads((INDEX / "pages.json").read_text())
    tax = json.loads((INDEX / "topics.json").read_text())
    labels = {}
    for b in tax["high_level"]:
        labels[b["id"]] = b["label"]
        for g in b.get("granular", []):
            labels[g["id"]] = g["label"]
    return pages, labels


def sample_topics(pages: list[dict], labels: dict[str, str]) -> list[dict]:
    rng = random.Random(SEED)
    by_topic: dict[str, list[int]] = {}
    for i, p in enumerate(pages):
        for t in p.get("granular", []):
            by_topic.setdefault(t, []).append(i)
    eligible = [t for t, idxs in by_topic.items()
                if TOPIC_MIN_PAGES <= len(idxs) <= TOPIC_MAX_PAGES and t in labels]
    rng.shuffle(eligible)
    chosen = eligible[:N_TOPICS]
    all_idx = set(range(len(pages)))
    tasks = []
    for t in chosen:
        pos = by_topic[t]
        neg_pool = list(all_idx - set(pos))
        tasks.append({
            "topic": t,
            "label": labels[t],
            "pos": rng.sample(pos, min(POS_PER_TOPIC, len(pos))),
            "neg": rng.sample(neg_pool, NEG_PER_TOPIC),
        })
    return tasks


def encode(model_name: str, texts: list[str], is_query: bool) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    if "bge" in model_name and is_query:
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    return m.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)


def doc_text(p: dict, condition: str) -> str:
    if condition == "as_deployed":
        return p.get("search_text", "") or ""
    # content_only: strip the injected topic-label echo so we measure real semantic matching, not the
    # label words being copied into the page text by build_index.
    return "\n\n".join(x for x in [p.get("content_text", ""), p.get("card_text", "")] if x)


def benchmark(hard: bool = False) -> None:
    pages, labels = load_corpus()
    tasks = sample_topics(pages, labels)
    if hard:
        # Hard positives: pages tagged with the topic whose text does NOT contain the label verbatim,
        # so ranking them requires real semantic matching (the abstract-query failure mode), not lexical
        # echo. This is the test that actually probes "is a transformer worth it for our hard queries".
        kept = []
        for t in tasks:
            lab = t["label"].lower()
            hp = [i for i in t["pos"]
                  if lab not in ((pages[i].get("content_text", "") or "") + " "
                                 + (pages[i].get("card_text", "") or "")).lower()]
            if len(hp) >= 5:
                t["pos"] = hp
                kept.append(t)
        tasks = kept
        print(f"HARD mode: {len(tasks)} topics with >=5 non-lexical (semantic-only) positives\n")
    needed = sorted({i for t in tasks for i in (t["pos"] + t["neg"])})
    row_of = {i: r for r, i in enumerate(needed)}
    qlabels = [t["label"] for t in tasks]
    print(f"{len(tasks)} topics | {len(needed)} unique pages encoded per model")
    print("AUC = separating on-topic (tagged) from off-topic (random) pages by cosine to the topic label.\n")
    print("'as_deployed' = page text incl. injected topic-label echo; 'content_only' = real semantic match.\n")
    print(f"{'model':28} {'as_deployed':>12} {'content_only':>13}")
    for short, name in CANDIDATES.items():
        qvecs = encode(name, qlabels, is_query=True)
        row = f"{short:28}"
        for cond in ("as_deployed", "content_only"):
            texts = [doc_text(pages[i], cond) for i in needed]
            doc_vecs = encode(name, texts, is_query=False)
            all_scores, all_truth = [], []
            for ti, t in enumerate(tasks):
                qv = qvecs[ti]
                all_scores += [float(doc_vecs[row_of[i]] @ qv) for i in t["pos"]]
                all_truth += [1] * len(t["pos"])
                all_scores += [float(doc_vecs[row_of[i]] @ qv) for i in t["neg"]]
                all_truth += [0] * len(t["neg"])
            row += f"{roc_auc_score(all_truth, all_scores):>12.3f} " + " " * 1
        print(row)


DEPLOYED_MODEL = "sentence-transformers/static-retrieval-mrl-en-v1"
OLD_FLOOR = 0.38  # the previous, never-calibrated guess — shown on the plot for comparison
CALIB_TOPICS = 200
CALIB_POS = 25
CALIB_NEG = 50


def calibrate() -> None:
    """Set RELEVANCE_FLOOR from data: build on-topic vs off-topic cosine distributions on the DEPLOYED
    embeddings, pick the threshold that best separates them (Youden's J), and plot it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sentence_transformers import SentenceTransformer

    pages, labels = load_corpus()
    emb = np.load(INDEX / "embeddings.npy")  # row i aligns with pages[i] (build_index appends in order)
    assert emb.shape[0] == len(pages), f"emb {emb.shape[0]} != pages {len(pages)}"

    rng = random.Random(SEED)
    by_topic: dict[str, list[int]] = {}
    for i, p in enumerate(pages):
        for t in p.get("granular", []):
            by_topic.setdefault(t, []).append(i)
    # granular topic -> set of pages in its parent high-level bucket (for realistic "sibling" negatives)
    tax = json.loads((INDEX / "topics.json").read_text())
    parent_of = {g["id"]: b["id"] for b in tax["high_level"] for g in b.get("granular", [])}
    by_bucket: dict[str, set[int]] = {}
    for i, p in enumerate(pages):
        for b in p.get("high_level", []):
            by_bucket.setdefault(b, set()).add(i)
    eligible = [t for t, idxs in by_topic.items()
                if TOPIC_MIN_PAGES <= len(idxs) <= TOPIC_MAX_PAGES and t in labels and t in parent_of]
    rng.shuffle(eligible)
    eligible = eligible[:CALIB_TOPICS]

    model = SentenceTransformer(DEPLOYED_MODEL)
    qvecs = model.encode([labels[t] for t in eligible], normalize_embeddings=True, convert_to_numpy=True)

    # Realistic negatives = pages in the same high-level bucket but NOT this granular topic — topically
    # adjacent but off-point, which is exactly what the floor must reject inside an enumerate union.
    pos_scores, neg_scores = [], []
    for ti, t in enumerate(eligible):
        qv = qvecs[ti]
        pos = rng.sample(by_topic[t], min(CALIB_POS, len(by_topic[t])))
        sibling_pool = list(by_bucket.get(parent_of[t], set()) - set(by_topic[t]))
        if len(sibling_pool) < 10:
            continue
        neg = rng.sample(sibling_pool, min(CALIB_NEG, len(sibling_pool)))
        pos_scores += [float(emb[i] @ qv) for i in pos]
        neg_scores += [float(emb[i] @ qv) for i in neg]
    pos_scores = np.array(pos_scores); neg_scores = np.array(neg_scores)

    scores = np.concatenate([pos_scores, neg_scores])
    truth = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thr = roc_curve(truth, scores)
    auc = roc_auc_score(truth, scores)
    youden = thr[np.argmax(tpr - fpr)]          # max separation: TPR - FPR
    # max-F1 operating point (precision-leaning alternative), reported for context
    f1s = []
    for c in thr:
        tp = (pos_scores >= c).sum(); fp = (neg_scores >= c).sum(); fn = (pos_scores < c).sum()
        prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
    f1_thr = thr[int(np.argmax(f1s))]
    floor = round(float(youden), 2)

    def stats_at(c):
        tp = (pos_scores >= c).sum(); fp = (neg_scores >= c).sum()
        return tp / len(pos_scores), fp / len(neg_scores)  # recall, false-positive rate

    print(f"calibration: {len(eligible)} topics | {len(pos_scores)} on-topic, {len(neg_scores)} off-topic samples")
    print(f"  AUC={auc:.3f}  on-topic mean={pos_scores.mean():.3f}  off-topic mean={neg_scores.mean():.3f}")
    print(f"  Youden's-J floor = {youden:.3f}  -> RELEVANCE_FLOOR = {floor}")
    print(f"  max-F1 floor     = {f1_thr:.3f}")
    for name, c in [("OLD 0.38", OLD_FLOOR), (f"NEW {floor}", floor)]:
        r, f = stats_at(c)
        print(f"  at {name}: on-topic recall={r:.1%}  off-topic kept(FPR)={f:.1%}")

    DOCS.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(-0.1, 0.9, 60)
    ax1.hist(neg_scores, bins=bins, alpha=0.6, color="#c1543b", label=f"off-topic (sibling bucket)  μ={neg_scores.mean():.2f}")
    ax1.hist(pos_scores, bins=bins, alpha=0.6, color="#127083", label=f"on-topic (tagged)  μ={pos_scores.mean():.2f}")
    ax1.axvline(OLD_FLOOR, color="#888", ls="--", lw=2, label=f"old guess {OLD_FLOOR}")
    ax1.axvline(floor, color="#1a7f37", ls="-", lw=2.5, label=f"calibrated {floor} (Youden J)")
    ax1.set_xlabel("cosine similarity to query (topic label)"); ax1.set_ylabel("page count")
    ax1.set_title(f"RELEVANCE_FLOOR calibration — static model (AUC {auc:.3f})")
    ax1.legend(fontsize=9)
    ax2.plot(fpr, tpr, color="#127083", lw=2)
    j = np.argmax(tpr - fpr)
    ax2.scatter([fpr[j]], [tpr[j]], color="#1a7f37", zorder=5, s=70,
                label=f"floor {floor}: recall {tpr[j]:.0%}, FPR {fpr[j]:.0%}")
    ax2.plot([0, 1], [0, 1], color="#bbb", ls=":")
    ax2.set_xlabel("false-positive rate (off-topic kept)"); ax2.set_ylabel("true-positive rate (on-topic kept)")
    ax2.set_title("ROC — separating on-topic from off-topic"); ax2.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out = DOCS / "relevance_floor_calibration.png"
    fig.savefig(out, dpi=130)
    print(f"  wrote {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--hard", action="store_true", help="benchmark on semantic-only (non-lexical) positives")
    ap.add_argument("--calibrate", action="store_true", help="calibrate + plot RELEVANCE_FLOOR")
    args = ap.parse_args()
    if args.benchmark:
        benchmark(hard=args.hard)
    if args.calibrate:
        calibrate()


if __name__ == "__main__":
    main()
