from __future__ import annotations

import json
import os
import re
import secrets as _secrets
import threading
from io import BytesIO
from pathlib import Path

import fitz
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from agent import run_agent


ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "Eye on the Monster.pdf"
PDF = fitz.open(PDF_PATH)
# PyMuPDF documents are NOT thread-safe, and FastAPI runs sync endpoints (page_image, export_pdf's
# figure render) in an anyio threadpool. Two users rendering pages at once would hit the shared
# document concurrently and corrupt the render — the "PDF occasionally fails to render" bug. Serialize
# every page access through one lock so only one render touches the document at a time.
_PDF_LOCK = threading.Lock()


def _render_page(page_num: int, dpi: int) -> tuple[bytes, int, int]:
    """Render a 1-based monster-PDF page under the document lock; returns (png_bytes, width, height).

    On the rare chance the shared handle has been left in a bad state, reopen the document once and
    retry rather than surfacing a broken image to the viewer."""
    global PDF
    with _PDF_LOCK:
        try:
            pix = PDF.load_page(page_num - 1).get_pixmap(dpi=dpi, alpha=False)
        except Exception:
            PDF = fitz.open(PDF_PATH)
            pix = PDF.load_page(page_num - 1).get_pixmap(dpi=dpi, alpha=False)
        return pix.tobytes("png"), pix.width, pix.height


def _load_sections() -> list[dict]:
    """Issue (section) boundaries, sorted by start page, so a citation can open the whole section the
    cited page belongs to — not just the lone page."""
    rows: list[dict] = []
    path = ROOT / "index" / "issues.jsonl"
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("page_start") and row.get("page_end"):
                rows.append(row)
    rows.sort(key=lambda r: int(r["page_start"]))
    return rows


_SECTIONS = _load_sections()


def _section_for_page(n: int) -> dict | None:
    for row in _SECTIONS:
        if int(row["page_start"]) <= n <= int(row["page_end"]):
            return row
    return None

_basic = HTTPBasic(auto_error=False)


def _require_password(creds: HTTPBasicCredentials | None = Depends(_basic)) -> None:
    expected = os.environ.get("APP_PASSWORD")
    if not expected:
        return
    if creds is None or not _secrets.compare_digest(creds.password, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


app = FastAPI(dependencies=[Depends(_require_password)])

PAGE_W = 612
PAGE_H = 792
MARGIN = 48
BOTTOM = PAGE_H - 48
CONTENT_W = PAGE_W - (2 * MARGIN)
TEAL = (18 / 255, 111 / 255, 131 / 255)
INK = (23 / 255, 26 / 255, 29 / 255)
MUTED = (93 / 255, 102 / 255, 112 / 255)
LINE = (207 / 255, 213 / 255, 216 / 255)

_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_MARK = re.compile(r"[*_`#>]+")
_WS = re.compile(r"\s+")


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((ROOT / "static" / "index.html").read_text())


@app.get("/run")
async def run(q: str, model: str | None = None) -> StreamingResponse:
    async def stream():
        async for event in run_agent(q, model=model):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/favicon.ico")
def favicon() -> Response:
    # Silence the browser's default favicon request (otherwise a console 404 on every load).
    return Response(status_code=204)


@app.get("/meta")
def meta() -> dict:
    # Total monster-PDF page count, so the page viewer can bound its prev/next navigation.
    return {"pages": len(PDF)}


@app.get("/page_image/{n}")
def page_image(n: int, dpi: int = 130) -> Response:
    if n < 1 or n > len(PDF):
        return Response("page not found", status_code=404)
    # dpi is the viewer's zoom control; clamp so a stray value can't ask for a giant render.
    dpi = max(72, min(int(dpi), 220))
    png, _, _ = _render_page(n, dpi)
    # Page renders never change — let the browser cache them so paging back/forth and re-opening
    # citations is instant and doesn't re-acquire the render lock.
    return Response(
        png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/page_section/{n}")
def page_section(n: int) -> dict:
    # The section (issue) a cited page belongs to, so the viewer can present the whole section the
    # citation refers to and scope navigation to it.
    section = _section_for_page(n) if 1 <= n <= len(PDF) else None
    if not section:
        return {"page": n, "section": None}
    return {
        "page": n,
        "section": {
            "issue_id": section.get("issue_id"),
            "title": section.get("title"),
            "issue_date": section.get("issue_date"),
            "page_start": int(section["page_start"]),
            "page_end": int(section["page_end"]),
            "n_pages": int(section.get("n_pages") or (int(section["page_end"]) - int(section["page_start"]) + 1)),
        },
    }


def _plain(value: object) -> str:
    text = str(value or "")
    text = (
        text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2022", "·")
    )
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_MARK.sub("", text)
    return _WS.sub(" ", text).strip()


def _wrap(text: str, font_size: float, width: float) -> list[str]:
    lines: list[str] = []
    for para in (text or "").splitlines() or [""]:
        words = para.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if fitz.get_text_length(candidate, fontname="helv", fontsize=font_size) <= width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


class ReportPdf:
    def __init__(self) -> None:
        self.doc = fitz.open()
        self.page: fitz.Page | None = None
        self.y = MARGIN
        self.new_page()

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.page.insert_text(
            (MARGIN, 30),
            "EYE ON THE MONSTER • GENERATED REPORT",
            fontsize=9,
            fontname="helv",
            color=INK,
        )
        self.page.draw_line((MARGIN, 42), (PAGE_W - MARGIN, 42), color=TEAL, width=1.8)
        self.y = 66

    def ensure(self, height: float) -> None:
        if self.y + height > BOTTOM:
            self.new_page()

    def rule(self, color=LINE, width: float = 0.7, pad: float = 8) -> None:
        self.ensure(pad + 2)
        assert self.page is not None
        self.page.draw_line((MARGIN, self.y), (PAGE_W - MARGIN, self.y), color=color, width=width)
        self.y += pad

    def text(
        self,
        value: object,
        *,
        size: float = 10.5,
        color=INK,
        width: float = CONTENT_W,
        gap: float = 4,
    ) -> None:
        text = _plain(value)
        if not text:
            return
        line_h = size * 1.34
        lines = _wrap(text, size, width)
        self.ensure((len(lines) * line_h) + gap)
        assert self.page is not None
        for line in lines:
            self.page.insert_text((MARGIN, self.y + size), line, fontsize=size, fontname="helv", color=color)
            self.y += line_h
        self.y += gap

    def label(self, value: str) -> None:
        self.ensure(22)
        assert self.page is not None
        self.page.insert_text(
            (MARGIN, self.y + 8),
            value.upper(),
            fontsize=8.5,
            fontname="helv",
            color=TEAL,
        )
        self.y += 20

    def title(self, value: object) -> None:
        self.rule(color=INK, width=1.1, pad=12)
        self.text(value, size=16, gap=8)
        self.rule(color=LINE, width=0.6, pad=10)

    def figure(self, ev: dict) -> None:
        try:
            page_num = int(ev.get("page"))
        except (TypeError, ValueError):
            return
        if page_num < 1 or page_num > len(PDF):
            return

        png, pix_w, pix_h = _render_page(page_num, 115)
        img_w = CONTENT_W
        img_h = img_w * (pix_h / pix_w)
        if img_h > 575:
            img_h = 575
            img_w = img_h * (pix_w / pix_h)
        self.ensure(img_h + 58)
        assert self.page is not None
        rect = fitz.Rect(MARGIN, self.y, MARGIN + img_w, self.y + img_h)
        self.page.insert_image(rect, stream=png)
        self.y += img_h + 8
        caption = _plain(ev.get("caption") or "Chart")
        self.text(f"{caption} [p. {page_num}]", size=9, color=MUTED, gap=8)

    def finding(self, ev: dict) -> None:
        self.rule(color=LINE, width=0.45, pad=9)
        if ev.get("kind") == "chart":
            self.figure(ev)
            return
        text = _plain(ev.get("text"))
        page = ev.get("page")
        if text:
            self.text(text, size=10.5, gap=2)
        cite = _plain(
            f"{ev.get('title') or ''}{', ' + ev.get('issue_date') if ev.get('issue_date') else ''}"
        )
        if page:
            cite = f"{cite} [p. {page}]".strip()
        self.text(cite, size=8.8, color=TEAL, gap=7)


def _build_report_pdf(payload: dict) -> bytes:
    pdf = ReportPdf()
    pdf.label("Query")
    pdf.text(payload.get("query") or "Eye on the Monster report", size=12, gap=12)

    answer = payload.get("answer")
    if answer:
        pdf.rule(color=TEAL, width=1.2, pad=12)
        pdf.label("Answer")
        pdf.text(answer, size=12.5, gap=14)

    sections = payload.get("sections") or []
    for section in sections:
        title = section.get("title") or "Findings"
        pdf.title(title)
        primary = section.get("primary") or []
        supporting = section.get("supporting") or []
        if primary:
            pdf.label("Primary findings")
            for ev in primary:
                pdf.finding(ev)
        if supporting:
            pdf.label("Supporting findings")
            for ev in supporting:
                pdf.finding(ev)

    out = pdf.doc.tobytes(garbage=4, deflate=True)
    pdf.doc.close()
    return out


@app.post("/export_pdf")
async def export_pdf(request: Request) -> StreamingResponse:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="expected report object")

    data = _build_report_pdf(payload)
    return StreamingResponse(
        BytesIO(data),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="eye-on-the-monster-report.pdf"'},
    )
