from __future__ import annotations

import json
from pathlib import Path

import fitz
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from agent import run_agent


ROOT = Path(__file__).resolve().parent
PDF = fitz.open(ROOT / "Eye on the Monster.pdf")
app = FastAPI()


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((ROOT / "static" / "index.html").read_text())


@app.get("/run")
async def run(q: str) -> StreamingResponse:
    async def stream():
        async for event in run_agent(q):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/page_image/{n}")
def page_image(n: int) -> Response:
    if n < 1 or n > len(PDF):
        return Response("page not found", status_code=404)
    pix = PDF.load_page(n - 1).get_pixmap(dpi=130, alpha=False)
    return Response(pix.tobytes("png"), media_type="image/png")
