"""Step 3 — Build the raw legal corpus: the statute DB (single source of truth).

Assets, in order (see docs/ROADMAP.md, "Core data assets"):
  1. Statute DB — 13 priority acts from indiacode.nic.in / legislative.gov.in,
     one JSONL row per section (schema: src/nyaya/schemas.py::StatuteSection).
     Pipeline: PDF -> pymupdf text extraction -> regex section splitter ->
     manual spot-check 5% -> JSONL. Watch for Devanagari mojibake; prefer HTML for Hindi.
     ** THIS SCRIPT IMPLEMENTS ONLY THIS ASSET. **
  2. IPC<->BNS / CrPC<->BNSS mapping tables — from official MHA comparison tables (~1,100 rows).
  3. Procedure KB — 60–80 hand-written "how do I..." docs verified against act text.
  4. Landmark judgments (~200 SC cases) — from HF datasets, not scraping.

TODO: implement extractors for assets 2-4.

Sources are listed in configs/statute_sources.yaml (verified 2026-07-13).

Outputs:
  data/raw/<act_id>.pdf              downloaded source PDF
  data/canonical/<act_id>.jsonl      extracted StatuteSection rows
  reports/corpus_spotcheck_<act_id>.jsonl   5% human-review sample

Go/no-go gate: docs/ROADMAP.md requires the manual spot-check to come back
>=98% clean before any synthetic generation. The splitter here is a
best-effort regex heuristic (bare acts are not perfectly uniform in
formatting) — that gate is not automatable and is why the spot-check sample
exists; a human fills in `_review_clean` on each sampled row.

KNOWN ISSUE: indiacode.nic.in returns HTTP 403 to plain (non-browser)
requests. download_pdf() sends browser-like headers to work around this;
if it still 403s, the site may require a session cookie obtained by first
visiting the handle page in a real browser.

Usage:
    python scripts/03_build_corpus.py                  # all acts in configs/statute_sources.yaml
    python scripts/03_build_corpus.py --act bns         # one act only
    python scripts/03_build_corpus.py --skip-download   # reuse PDFs already in data/raw/
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.schemas import StatuteSection  # noqa: E402

SOURCES_CONFIG = ROOT / "configs" / "statute_sources.yaml"
RAW_DIR = ROOT / "data" / "raw"
CANONICAL_DIR = ROOT / "data" / "canonical"
REPORTS_DIR = ROOT / "reports"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}

SPOT_CHECK_FRACTION = 0.05
MIN_CLEAN_FRACTION = 0.98  # go/no-go gate — see docs/ROADMAP.md
DOWNLOAD_RETRIES = 3

# Matches a section header at the start of a candidate span: a number
# (optionally letter-suffixed, e.g. "66A") followed by ". " and a capital
# letter. Excludes numbers preceded by a digit/period/open-paren so
# sub-clause numbering like "(1)" or dates like "2018." never match.
SECTION_MARKER = re.compile(r"(?<![\d.\(])\b(\d{1,3}[A-Z]{0,2})\.\s+(?=[A-Z])")
# Constitution uses "Article N." instead of a bare number.
ARTICLE_MARKER = re.compile(r"\bArticle\s+(\d{1,3}[A-Z]{0,2})\.\s+(?=[A-Z])")
# Indian bare acts conventionally separate a section's title from its body
# with an em-dash (or hyphen) after the title: "Title.—Body...". The dash
# is optional so sections without one still split at the first sentence.
TITLE_BODY_SPLIT = re.compile(r"^(?P<title>[^.]{1,150}?)\.\s*[—\-]?\s*(?P<body>.+)$", re.DOTALL)

GARBLE_CHARS = "�"


def load_sources(act_filter: str | None = None) -> list[dict]:
    config = yaml.safe_load(SOURCES_CONFIG.read_text(encoding="utf-8"))
    acts = config["acts"]
    if act_filter:
        acts = [a for a in acts if a["id"] == act_filter]
        if not acts:
            raise SystemExit(f"unknown act id: {act_filter}")
    return acts


def download_pdf(url: str, dest: Path, referer: str, session) -> bool:
    """Download `url` to `dest` with browser-like headers. Skips if already present."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already downloaded")
        return True
    headers = {**BROWSER_HEADERS, "Referer": referer}
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            resp = session.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            return True
        except requests.RequestException as e:
            print(f"[retry {attempt + 1}/{DOWNLOAD_RETRIES}] {url}: {e}")
            if attempt < DOWNLOAD_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    print(f"[FAILED] could not download {url}")
    return False


def extract_pages(pdf_path: Path) -> list[str]:
    """Return raw per-page text via PyMuPDF."""
    import pymupdf

    doc = pymupdf.open(pdf_path)
    try:
        return [page.get_text() for page in doc]
    finally:
        doc.close()


def _is_page_number(line: str) -> bool:
    stripped = line.strip(".- ")
    return len(stripped) <= 4 and stripped.isdigit()


def clean_pages(pages: list[str]) -> str:
    """Strip repeated running headers/footers and bare page numbers, then
    collapse everything into one whitespace-normalized string.

    Lines that repeat on more than half the pages are almost always running
    headers/footers ("THE CONSUMER PROTECTION ACT, 2019" on every page) —
    dropping them by frequency, rather than by act-specific pattern, keeps
    this general across all 13 acts.
    """
    line_lists = [[ln.strip() for ln in p.split("\n") if ln.strip()] for p in pages]
    n_pages = len(pages)
    noisy: set[str] = set()
    if n_pages > 3:
        counts = Counter(ln for lines in line_lists for ln in set(lines))
        noisy = {ln for ln, c in counts.items() if c > n_pages * 0.5}

    kept = [
        ln
        for lines in line_lists
        for ln in lines
        if ln not in noisy and not _is_page_number(ln)
    ]
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def garble_ratio(text: str) -> float:
    """Fraction of characters that are Unicode replacement chars — a cheap
    signal for PDF-encoding mojibake (see docs/ROADMAP.md warning)."""
    if not text:
        return 0.0
    return sum(1 for c in text if c in GARBLE_CHARS) / len(text)


def split_sections(text: str, use_articles: bool = False) -> list[dict]:
    """Split normalized full-act text into {section, title, text} candidates.

    Best-effort regex split — Indian bare acts are not perfectly uniform,
    hence the mandatory 5% manual spot-check + >=98% clean gate before this
    feeds synthetic generation. The "ARRANGEMENT OF SECTIONS" table of
    contents also matches the header pattern (each entry looks like
    "1. Short title, extent and commencement."); it is filtered out by
    keeping, per section number, only the candidate with the longest body —
    the TOC entry is always the shorter of the two.

    Known false-positive risk: a sentence that happens to end with a section
    cross-reference immediately followed by a capitalized sentence (e.g.
    "...as provided in section 34. This includes...") can be misread as a
    new header. This is exactly what the spot-check gate exists to catch.
    """
    marker = ARTICLE_MARKER if use_articles else SECTION_MARKER
    matches = list(marker.finditer(text))
    if not matches:
        return []

    candidates: dict[str, dict] = {}
    first_seen_at: dict[str, int] = {}
    for i, m in enumerate(matches):
        num = m.group(1)
        span_start = m.start()
        span_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        span = text[span_start:span_end]
        rest = span[m.end() - span_start :].strip()

        title_match = TITLE_BODY_SPLIT.match(rest)
        if title_match:
            title = title_match.group("title").strip()
            body = title_match.group("body").strip()
        else:
            title = None
            body = rest
        if not body:
            continue

        first_seen_at.setdefault(num, span_start)
        existing = candidates.get(num)
        if existing is None or len(body) > len(existing["text"]):
            candidates[num] = {"section": num, "title": title, "text": body}

    return sorted(candidates.values(), key=lambda c: first_seen_at[c["section"]])


def build_statute_sections(act: dict, split: list[dict]) -> list[StatuteSection]:
    return [
        StatuteSection(
            act_id=act["id"],
            act_name=act["name"],
            section=c["section"],
            title=c["title"] or "",
            text=c["text"],
            source_url=act.get("pdf_url"),
        )
        for c in split
    ]


def write_jsonl(records: list[StatuteSection], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_spot_check(records: list[StatuteSection], path: Path, fraction: float = SPOT_CHECK_FRACTION) -> int:
    """Sample `fraction` of records into a human-review JSONL file.

    Each row gets an extra `_review_clean` field (null) for the reviewer to
    fill in with true/false. Returns the sample size (0 if `records` is empty
    — no file is written in that case).
    """
    if not records:
        return 0
    n = max(1, round(len(records) * fraction))
    sample = random.sample(records, min(n, len(records)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in sample:
            row = asdict(r)
            row["_review_clean"] = None
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(sample)


def process_act(act: dict, session, skip_download: bool = False) -> dict:
    pdf_path = RAW_DIR / f"{act['id']}.pdf"
    if not skip_download:
        ok = download_pdf(act["pdf_url"], pdf_path, referer=act["handle_url"], session=session)
        if not ok:
            return {"act": act["id"], "status": "download_failed"}
    if not pdf_path.exists():
        return {"act": act["id"], "status": "missing_pdf"}

    pages = extract_pages(pdf_path)
    text = clean_pages(pages)
    if len(text) < 500:
        return {"act": act["id"], "status": "empty_extraction", "chars": len(text)}

    ratio = garble_ratio(text)
    if ratio > 0.01:
        print(f"[warn] {act['id']}: {ratio:.1%} replacement characters — possible encoding/mojibake issue")

    split = split_sections(text, use_articles=(act["id"] == "constitution"))
    records = build_statute_sections(act, split)
    if not records:
        return {"act": act["id"], "status": "no_sections_found"}

    out_path = CANONICAL_DIR / f"{act['id']}.jsonl"
    write_jsonl(records, out_path)
    sample_path = REPORTS_DIR / f"corpus_spotcheck_{act['id']}.jsonl"
    sample_n = write_spot_check(records, sample_path)

    return {
        "act": act["id"],
        "status": "ok",
        "sections": len(records),
        "spot_check_file": str(sample_path.relative_to(ROOT)),
        "spot_check_n": sample_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act", help="process a single act id from configs/statute_sources.yaml")
    parser.add_argument("--skip-download", action="store_true", help="reuse PDFs already in data/raw/")
    args = parser.parse_args()

    acts = load_sources(args.act)
    session = requests.Session()
    results = [process_act(act, session, skip_download=args.skip_download) for act in acts]

    print("\n=== Statute DB build report ===")
    for r in results:
        extra = f" — {r['sections']} sections" if r["status"] == "ok" else ""
        print(f"  [{r['status']}] {r['act']}{extra}")

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n{len(ok)}/{len(results)} acts extracted.")
    if ok:
        print(
            "\nGATE (docs/ROADMAP.md): synthetic generation may not start until "
            f"manual spot-check comes back >={MIN_CLEAN_FRACTION:.0%} clean. Review "
            "the sampled rows in reports/corpus_spotcheck_<act>.jsonl, fill in "
            "`_review_clean` (true/false) for each, then confirm the gate "
            "before running scripts/04_generate_examples.py."
        )

    if len(ok) != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
