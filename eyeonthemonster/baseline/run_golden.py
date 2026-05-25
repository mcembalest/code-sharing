from pathlib import Path

from query import format_hits, top_k

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "golden_queries.txt"
OUT = Path(__file__).resolve().parent / "golden_results.md"


def main() -> None:
    queries = [ln.strip() for ln in QUERIES.read_text().splitlines() if ln.strip()]
    sections = []
    for q in queries:
        hits = top_k(q, limit=12, bm25_weight=0.5)
        sections.append(f"## {q}\n\n{format_hits(hits)}")
    OUT.write_text("\n".join(sections))
    print(f"wrote {OUT} | {len(queries)} queries")


if __name__ == "__main__":
    main()
