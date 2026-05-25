import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "_diag" / "pages_clean.jsonl"
OUT = ROOT / "index" / "issues.jsonl"

HEADER_RE = re.compile(r"^(?:eye on the market|eotm|michael cembalest)\b", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
DATE_LINE_RE = re.compile(
    r"^(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2},\s+\d{4}$",
    re.IGNORECASE,
)
BOILERPLATE_RE = re.compile(
    r"^(?:please see important disclaimers|important disclosures|table of contents)\b",
    re.IGNORECASE,
)


def norm_date(value: str | None) -> str | None:
    if not value:
        return None
    for fmt in ("%B %d, %Y", "%B %e, %Y", "%b %d, %Y", "%b %e, %Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return value


def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" \t\r\n-–—•")


def title_from_page(text: str) -> str:
    for raw_line in text.splitlines():
        line = clean_line(raw_line)
        if not line:
            continue
        if PAGE_NUMBER_RE.match(line):
            continue
        if DATE_LINE_RE.match(line):
            continue
        if HEADER_RE.match(line):
            continue
        if BOILERPLATE_RE.match(line):
            continue
        return line[:180]
    return "Untitled"


def is_boundary(page: dict, prev: dict | None) -> bool:
    if str(page.get("position_in_issue")) == "1":
        return True
    return bool(prev and page.get("issue_date") and page.get("issue_date") != prev.get("issue_date"))


def load_pages() -> list[dict]:
    with SRC.open() as f:
        return [json.loads(line) for line in f]


def build_issues(pages: list[dict]) -> list[dict]:
    issues: list[dict] = []
    current: dict | None = None
    prev: dict | None = None

    for page in pages:
        page_num = int(page["page"])
        if current is None or is_boundary(page, prev):
            if current is not None:
                current["page_end"] = int(prev["page"])
                current["n_pages"] = current["page_end"] - current["page_start"] + 1
                issues.append(current)

            current = {
                "issue_id": f"issue-{len(issues) + 1:04d}",
                "issue_date": norm_date(page.get("issue_date")),
                "title": title_from_page(page.get("content_text") or ""),
                "page_start": page_num,
            }
        prev = page

    if current is not None and prev is not None:
        current["page_end"] = int(prev["page"])
        current["n_pages"] = current["page_end"] - current["page_start"] + 1
        issues.append(current)

    return issues


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_sanity(issues: list[dict]) -> None:
    lengths = [i["n_pages"] for i in issues]
    long_issues = [i for i in issues if i["n_pages"] > 60]

    print(f"wrote {len(issues)} issues -> {OUT.relative_to(ROOT)}")
    print(
        "pages/issue: "
        f"min={min(lengths)} p25={sorted(lengths)[len(lengths)//4]} "
        f"median={median(lengths):.1f} mean={mean(lengths):.1f} "
        f"p75={sorted(lengths)[(len(lengths)*3)//4]} max={max(lengths)}"
    )

    if long_issues:
        print("issues spanning >60 pages:")
        for issue in long_issues:
            print(
                f"  {issue['issue_id']} {issue['issue_date'] or 'unknown-date'} "
                f"pp.{issue['page_start']}-{issue['page_end']} "
                f"({issue['n_pages']} pages): {issue['title']}"
            )
    else:
        print("issues spanning >60 pages: none")


def main() -> None:
    issues = build_issues(load_pages())
    write_jsonl(OUT, issues)
    print_sanity(issues)


if __name__ == "__main__":
    main()
