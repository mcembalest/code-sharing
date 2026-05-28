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


def _is_boilerplate(line: str) -> bool:
    # Order-independent: these line types are running headers/footers/disclaimers wherever they sit.
    if _HEADER.search(line) or _PORTAL.search(line) or _DISCLAIMER.search(line):
        return True
    if _DATE_LINE.match(line) or _PAGENUM.match(line):
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
