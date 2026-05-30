"""Strip running-header / footer / disclaimer boilerplate from a page's content_text.

The PDF extraction leaves the J.P. Morgan running header ("EYE ON THE MARKET ... MICHAEL CEMBALEST
... J.P. MORGAN"), a covid-era "Access our ... web portal here" banner, the standalone date and page
number, the FDIC investment-products disclaimer block, and wingding bullet glyphs (private-use-area
chars) embedded at the top of essentially every page. Those polluted both the report's quote bodies
and the embedding text. This strips them line-by-line (strip-not-skip: the page is kept, only the
boilerplate lines are removed) so what remains is the author's actual prose.
"""

from __future__ import annotations

import re

# Wingding/symbol bullets land in the Unicode private-use area (U+F000-U+F0FF, e.g. U+F0B7); drop them.
_PUA = re.compile("[\uf000-\uf0ff]")
# C0 control chars (except tab/newline) show up as OCR mojibake inside the running header
# ("MICHAEL CEMBALEST" -> "MI\x12H!EL ...") and as stray bytes in prose; drop them.
_CTRL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")
# Match the running header on "EYE ON THE MARKET ... MORGAN" rather than the author's name: the name
# is frequently OCR-corrupted, but the J.P. MORGAN bracket is stable. Covers the "... OUTLOOK ..."
# and energy-paper header variants too.
_HEADER = re.compile(r"eye on the market\b.*\bmorgan\b", re.I)
_PORTAL = re.compile(r"web portal here", re.I)
_DISCLAIMER = re.compile(
    r"not fdic insured|investment products are|investment products:"
    r"|not a deposit|guaranteed by,?\s*jpmorgan|subject to investment risk"
    r"|possible loss of the principal|or guaranteed by",
    re.I,
)
# A standalone running-header date line. OCR sometimes splits the digits ("January 1 , 202 3"), so
# after the month name we allow any run of digits/spaces/commas/periods (must contain a digit) to EOL
# rather than a rigid "<day>, <year>" shape.
_DATE_LINE = re.compile(
    r"^(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)[\s.,]*\d[\s\d.,]*$",
    re.I,
)
_PAGENUM = re.compile(r"^\d{1,4}$")
# Lines that are only bullet/dash glyphs (ascii + a few common unicode bullets) and whitespace.
_BULLET_ONLY = re.compile("^[" + re.escape("-–—•·∙◦●○▪♦") + r"\s]*$")
_BLANKS = re.compile(r"\n{2,}")

# Sometimes only the "J.P. MORGAN" tail of the running header survives OCR
# (the "EYE ON THE MARKET ... MICHAEL CEMBALEST" portion is mangled into
# wingdings/control chars and stripped before this line is matched). Catch
# that residual as its own line type.
_HEADER_TAIL = re.compile(r"^\W*j\.?\s*p\.?\s*morgan\W*$", re.I)

# "Access our X here" cross-promo banner pointing the reader at a sibling
# publication (web portal, outlook, energy paper). Existing _PORTAL only
# caught the covid-era "web portal here"; widen to all "access our ... here".
_CROSS_PROMO = re.compile(r"^access our\b.*\bhere\b\.?$", re.I)

# Per-issue preamble paragraph that opens certain modern pieces. PyMuPDF
# splits it into ~3 lines, so we match each distinctive fragment of the
# paragraph individually — all of them are stable boilerplate that never
# appears in body prose.
_PB_PREAMBLE = re.compile(
    r"the following commentary has been developed exclusively for j\.?\s*p\.?\s*morgan"
    r"|preserve the integrity of our(?:\s+ongoing)?"
    r"|integrity of our ongoing dialogue"
    r"|ongoing dialogue with you,?\s+it is important that this information remain private"
    r"|it is important that this information remain private",
    re.I,
)

# International distribution-restriction line that lands by itself on certain
# pages: "FOR INSTITUTIONAL/WHOLESALE/PROFESSIONAL CLIENTS AND QUALIFIED ..."
_INST_USE = re.compile(
    r"\bfor (?:institutional|wholesale|professional)[\s/]+"
    r"(?:institutional|wholesale|professional)",
    re.I,
)

# Per-issue sub-banner. Appears immediately below the JPM running header on
# every page of an issue, naming the publication family. Two shapes observed:
#   (a) "/"-separated nav listing concurrent publications, e.g.
#       "2025 Energy Paper / Trump Tracker"
#       "2024 Outlook / 2024 energy paper / US inflation monitor / US Federal debt monitor"
#   (b) a single short banner naming one publication, e.g.
#       "Online Trump Tracker", "2025 Eye on the Market Outlook", "2026 Energy Paper"
# Body prose can mention any of these phrases, so we don't strip on keyword
# alone — we require the whole line to be short, terminal-punctuation-free,
# and either slash-delimited with short segments or a single brief banner.
_SUBBANNER_KEYWORD = re.compile(
    r"\b(?:trump tracker|energy paper|"
    r"eye on the market outlook|"
    r"(?:annual\s+)?outlook|"
    r"(?:inflation|federal\s+debt|economic|equity|labor)\s+monitor)\b",
    re.I,
)


def _is_sub_banner(line: str) -> bool:
    if not line or len(line) > 120:
        return False
    if line[-1] in ".?!":
        return False
    if not _SUBBANNER_KEYWORD.search(line):
        return False
    if " / " in line:
        segs = [s.strip() for s in line.split(" / ")]
        return bool(segs) and all(0 < len(s) <= 50 for s in segs)
    return len(line.split()) <= 8


def _is_boilerplate(line: str) -> bool:
    # Order-independent: these line types are running headers/footers/disclaimers wherever they sit.
    if _HEADER.search(line) or _PORTAL.search(line) or _DISCLAIMER.search(line):
        return True
    if _HEADER_TAIL.match(line) or _CROSS_PROMO.match(line):
        return True
    if _PB_PREAMBLE.search(line) or _INST_USE.search(line):
        return True
    if _DATE_LINE.match(line) or _PAGENUM.match(line):
        return True
    if _is_sub_banner(line):
        return True
    return bool(_BULLET_ONLY.match(line))


def clean_content(text: str) -> str:
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = _CTRL.sub("", _PUA.sub("", raw)).strip()
        if not line:
            out.append("")
            continue
        if _is_boilerplate(line):
            continue
        out.append(line)
    return _BLANKS.sub("\n\n", "\n".join(out)).strip()
