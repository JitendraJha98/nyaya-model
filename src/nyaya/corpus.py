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
# U+2015 HORIZONTAL BAR is included: India Code's re-typeset "As on" PDFs of
# older acts (SMA 1954, HMA 1955) use it instead of an em dash after titles.
# The optional "[" before the number is a wholly-substituted section
# ("[16. Title.—…") whose footnote superscript the extractor stripped.
# The title is tempered so it can never run across the next section's own
# start line — otherwise a dash-less line (e.g. an omitted section) makes the
# lazy title swallow its successor — nor across a blank line (real wrapped
# titles never contain one; trailing ToC entries otherwise bleed into the
# act header when fed unsliced text).
# "\s?\." tolerates the stray space in Code on Wages s."39 .Time limit…".
_NEXT_SECTION_GUARD = r"(?!\n\s*\n)(?!\n\s*\[?\d{1,3}[A-Z]{0,2}\s?\.\s)"
_SECTION_START = re.compile(
    r"^\s*\[?(\d{1,3}[A-Z]{0,2})\s?\.\s*[—–―]{0,2}\s*"
    r"((?:" + _NEXT_SECTION_GUARD + r"[\s\S]){3,300}?)"
    r"(?:\.\s*[—–―-]{1,2}|\s*[—–―]{1,2})",
    re.MULTILINE,
)
# Omitted/repealed sections have no dash separator at all:
# "6. [Guardianship in marriage.] Omitted by the Child Marriage …".
_OMITTED_SECTION = re.compile(
    r"^\s*\[?(\d{1,3}[A-Z]{0,2})\s?\.\s*\[([^\]\n]{3,200}?)\.?\]\s*(?=Omitted|Rep)",
    re.MULTILINE,
)
_CHAPTER = re.compile(r"^\s*(CHAPTER\s+[IVXLC]+[A-Z]?)\s*$", re.MULTILINE)
# Post-1950 acts: "BE it enacted by Parliament…"; pre-Constitution acts
# (NI Act 1881): "It is hereby enacted as follows".
_ENACTING = re.compile(
    r"BE it enacted by Parliament|It is hereby enacted", re.IGNORECASE
)
# "\[?" tolerates schedules inserted by amendment ("[THE FIRST SCHEDULE").
_TERMINATORS = re.compile(
    r"^\s*\[?(THE\s+(FIRST\s+|SECOND\s+|THIRD\s+)?SCHEDULE|APPENDIX"
    r"|STATEMENT OF OBJECTS AND REASONS)\b",
    re.MULTILINE,
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
    candidates = sorted(
        list(_SECTION_START.finditer(text)) + list(_OMITTED_SECTION.finditer(text)),
        key=lambda m: (m.start(), m.end()),
    )
    matches, last_end, seen = [], -1, set()
    for m in candidates:
        if m.start() < last_end:  # overlapping duplicate from the other regex
            continue
        # A title containing its own section-start line means the match
        # swallowed ToC entries (no-dash lines) — not a real section.
        if re.search(r"^\s*\d{1,3}[A-Z]{0,2}\.\s", m.group(2), re.MULTILINE):
            continue
        # A repeated section number is footnote noise (commencement
        # notifications print as "1. 18 December, 2020.—…" mid-act).
        if m.group(1) in seen:
            continue
        matches.append(m)
        last_end = m.end()
        seen.add(m.group(1))
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


# "(?=\s|$)" not "\s": a right-column number whose title wrapped to the next
# line is a bare "498A." with nothing after the period. Letter-suffixed
# sections ("41A Notice…") sometimes omit the period entirely — distinctive
# enough to accept; bare digits without a period are NOT (prose risk).
_PLAIN_SECTION = re.compile(r"^[ \t]*(\d{1,3})\.(?=\s|$)")
_SUFFIXED_SECTION = re.compile(r"^[ \t]*(\d{1,3}[A-Z]{1,2})\.?(?=\s|$)")
_CELL_SECTION = re.compile(r"^[ \t]*(\d{1,3}[A-Z]{0,2})\.(?:\s|$)", re.MULTILINE)
_SUBSECTION_MARKER = re.compile(r"^\s*(\d{1,3})\s*\(\d+\)")


def _cell_sections(cell: str) -> list[str]:
    """Line-leading section numbers in a table cell."""
    out = []
    for line in cell.splitlines():
        m = _PLAIN_SECTION.match(line) or _SUFFIXED_SECTION.match(line)
        if m:
            out.append(m.group(1))
    return out


def extract_mapping_pairs(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """(new-act cell, old-act cell) rows -> (new_section, old_section) pairs.

    Semantics from the NCRB Sankalan corresponding-section tables: a
    line-leading "N." in the left cell sets the current new-act section, and
    so does a subsection marker "N (k)" / "N(k)" (consolidating sections
    appear only as subsection-level lines). Every line-leading "M." in the
    right cell pairs with the current section. Header/chapter rows and
    "New Section" markers yield no pairs.
    """
    pairs = []
    current_new = None
    for left, right in rows:
        left, right = left or "", right or ""
        if "CHAPTER" in left.upper() or not (left.strip() or right.strip()):
            continue
        left_sections = _cell_sections(left)
        if left_sections:
            current_new = left_sections[0]
        else:
            marker = _SUBSECTION_MARKER.match(left)
            if marker:
                current_new = marker.group(1)
        if current_new is None:
            continue
        for old in _cell_sections(right):
            pairs.append((current_new, old))
    return pairs


def _column_lines(page) -> list[tuple[float, int, str]]:
    """Reconstruct (y, column, text) lines from word positions.

    fitz's find_tables drops/mis-merges rows of the NCRB tables (borderless,
    irregular), so columns are split positionally instead: a line belongs to
    the right column when it starts past ~48% of the page width.
    """
    mid = page.rect.width * 0.48
    lines: dict[tuple[int, int, int], list[tuple[int, float, float, str]]] = {}
    for x0, y0, _x1, _y1, word, block, line, word_no in page.get_text("words"):
        # partition each physical line at the column boundary: fitz sometimes
        # merges both columns' words into one line object
        column = 0 if x0 < mid else 1
        lines.setdefault((block, line, column), []).append((word_no, x0, y0, word))
    out = []
    for words in lines.values():
        words.sort()
        x_start = min(w[1] for w in words)
        y = min(w[2] for w in words)
        text = " ".join(w[3] for w in words)
        out.append((y, 0 if x_start < mid else 1, text))
    # Band y into 3pt buckets before sorting: cells on the same visual row can
    # differ by sub-point y, and a right cell sorting before its left setter
    # attaches the old section to the wrong new section. Row pitch is >=12pt,
    # so banding never merges adjacent rows.
    return sorted(out, key=lambda item: (round(item[0] / 3), item[1], item[0]))


def parse_ncrb_mapping(pdf_path) -> list[tuple[str, str]]:
    """Extract (new_section, old_section) pairs from an NCRB Sankalan PDF.

    The corresponding-section table is a contiguous block of two-column pages
    beginning at the "CORRESPONDING SECTION TABLE" header; the block ends at
    the first page with no numbered right-column lines (the act text)."""
    import fitz

    rows: list[tuple[str, str]] = []
    started = False
    with fitz.open(pdf_path) as doc:
        for page in doc:
            has_header = "CORRESPONDING SECTION TABLE" in page.get_text().upper()
            lines = _column_lines(page)
            right_numbered = any(
                col == 1 and _CELL_SECTION.match(text) for _y, col, text in lines
            )
            if not started:
                # the index/cover pages mention the header too — the block
                # starts at the first page that also has two-column content
                if has_header and right_numbered:
                    started = True
                else:
                    continue
            elif not right_numbered:
                break
            for _y, col, text in lines:
                rows.append((text, "") if col == 0 else ("", text))
    return extract_mapping_pairs(rows)


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
