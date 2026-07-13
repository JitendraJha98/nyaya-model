"""Statute-DB extraction: gazette/India Code PDF text -> StatuteSection rows.

Pipeline pieces used by scripts/03_build_corpus.py:
  extract_pdf_text()  PDF -> text, font-aware: drops page-bottom footnotes
                      (small type that would otherwise collide with section
                      numbering — amendment footnotes literally start "1. ")
                      and superscript markers; strips page-number lines.
  slice_act_body()    trims the ARRANGEMENT OF SECTIONS ToC (before the
                      enacting formula) and everything from THE SCHEDULE on.
  split_sections()    splits on the gazette section pattern
                      "<n>. <Title>.——<body>" and attaches chapters.
  validate_sections() the >=98%-clean gate: count vs expected, numbering
                      monotonicity/gaps, empty bodies.

Verbatim discipline: section text is reproduced as extracted (whitespace
normalised); nothing is paraphrased.
"""

import re

# "103. Title.——(1) body" — title ends with "." followed by em/en dashes.
# ToC lines ("103. Title.") lack the dash separator and thus never match.
# Long titles wrap across lines in the PDFs, so the title may span newlines;
# the lazy quantifier stops at the first ".—", which titles never contain.
# Gazette typo tolerance (both occur in the official BNS PDF): a missing
# period before the dash (s.217 "…person—Whoever") — accepted only for
# em/en dashes so hyphens like "co-operative" never terminate a title —
# and a stray dash right after the number (s.255 "255.—Title.—").
_SECTION_START = re.compile(
    r"^\s*(\d{1,3}[A-Z]{0,2})\.\s*[—–]{0,2}\s*([\s\S]{3,300}?)(?:\.\s*[—–-]{1,2}|\s*[—–]{1,2})",
    re.MULTILINE,
)
_CHAPTER = re.compile(r"^\s*(CHAPTER\s+[IVXLC]+[A-Z]?)\s*$", re.MULTILINE)
_ENACTING = re.compile(r"BE it enacted by Parliament", re.IGNORECASE)
_TERMINATORS = re.compile(
    r"^\s*(THE\s+(FIRST\s+|SECOND\s+|THIRD\s+)?SCHEDULE|APPENDIX)\b", re.MULTILINE
)


def extract_pdf_text(pdf_path) -> str:
    """PDF -> text with footnotes, superscripts and page numbers removed."""
    import fitz

    parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_height = page.rect.height
            footnote_y = page_height * 0.78
            lines_out = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    kept = []
                    for span in line["spans"]:
                        size = span["size"]
                        y0 = span["bbox"][1]
                        if size <= 7.5:
                            continue  # superscript footnote markers
                        if size <= 9.5 and y0 >= footnote_y:
                            continue  # footnote text at page bottom
                        kept.append(span["text"])
                    text = "".join(kept)
                    if text.strip():
                        lines_out.append(text)
            # first line of each page is the page number
            if lines_out and re.fullmatch(r"\s*\d+\s*", lines_out[0]):
                lines_out = lines_out[1:]
            parts.append("\n".join(lines_out))
    return "\n".join(parts)


def slice_act_body(text: str) -> str:
    """Trim ToC (everything before the enacting formula) and schedules."""
    m = _ENACTING.search(text)
    if m:
        text = text[m.start():]
    t = _TERMINATORS.search(text)
    if t:
        text = text[: t.start()]
    return text


def split_sections(text: str) -> list[dict]:
    """Split act body into sections with chapter attribution."""
    chapters = []  # (position, "CHAPTER I — HEADING")
    for m in _CHAPTER.finditer(text):
        label = m.group(1).strip()
        # the chapter heading is the next non-empty line if it is upper-case
        rest = text[m.end():].lstrip("\n")
        heading = rest.split("\n", 1)[0].strip()
        if heading and heading == heading.upper() and not _SECTION_START.match(rest):
            label = f"{label} — {heading.title()}"
        chapters.append((m.start(), label))

    def chapter_at(pos):
        current = None
        for cpos, label in chapters:
            if cpos <= pos:
                current = label
            else:
                break
        return current

    sections = []
    matches = [
        m
        for m in _SECTION_START.finditer(text)
        # A title containing its own section-start line means the match
        # swallowed ToC entries (no-dash lines) — not a real section.
        if not re.search(r"^\s*\d{1,3}[A-Z]{0,2}\.\s", m.group(2), re.MULTILINE)
    ]
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end]
        # drop trailing chapter headings that belong to the next section
        body = _CHAPTER.split(body)[0]
        body = re.sub(r"\s+", " ", body).strip()
        sections.append(
            {
                "section": m.group(1),
                "title": re.sub(r"\s+", " ", m.group(2)).strip(),
                "text": body,
                "chapter": chapter_at(m.start()),
            }
        )
    return sections


def _numeric(section: str) -> int:
    return int(re.match(r"\d+", section).group(0))


def validate_sections(sections: list[dict], expected_count: int | None = None) -> dict:
    """Extraction-quality report backing the >=98%-clean go/no-go gate."""
    numbers = [s["section"] for s in sections]
    numeric = [_numeric(n) for n in numbers]
    monotonic = all(b >= a for a, b in zip(numeric, numeric[1:]))

    gaps = []
    seen = {n for n in numeric}
    if numeric:
        for expected in range(min(numeric), max(numeric) + 1):
            if expected not in seen:
                gaps.append(str(expected))

    empty_or_short = [s["section"] for s in sections if len(s["text"]) < 40]

    problems = len(gaps) + len(empty_or_short) + (0 if monotonic else 1)
    denominator = max(len(sections), expected_count or 0, 1)
    clean_fraction = round(max(0.0, 1 - problems / denominator), 4)

    return {
        "extracted": len(sections),
        "expected": expected_count,
        "monotonic": monotonic,
        "numbering_gaps": gaps,
        "empty_or_short": empty_or_short,
        "clean_fraction": clean_fraction,
    }
