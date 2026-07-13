"""Step 3 — Build the raw legal corpus: the statute DB (single source of truth).

This script is a THIN ORCHESTRATOR over src/nyaya/corpus.py — the extraction,
splitting, validation and NCRB-mapping logic lives in the library (importable
and unit-tested in tests/test_corpus.py + tests/test_mappings.py); this file
only does I/O: download PDFs, loop over the config, write JSONL + reports, and
enforce the go/no-go gate.

For every enabled act in configs/acts.yaml:
    download the official India Code PDF into data/raw/acts/ (atomic, cached) ->
    extract_pdf_text (footnote/superscript-aware) -> slice_act_body (ToC and
    schedules trimmed) -> split_sections (chapter-attributed) ->
    validate_sections (count vs expected, numbering monotonicity/gaps, empty
    bodies) -> write data/canonical/<act_id>.jsonl (StatuteSection rows).

Then, unless --act is given, parse the NCRB Sankalan corresponding-section
tables (config `mappings`) into data/canonical/law_mappings.jsonl.

Also writes:
    reports/corpus_extraction_report.json   per-act quality metrics + mapping counts
    reports/corpus_spotcheck_sample.json    ~5% random sections per act, for the
                                            manual spot-check the roadmap requires
                                            before synthetic generation

Go/no-go gate (docs/ROADMAP.md): exits non-zero if any enabled act extracts
< 98% clean. The splitter is a best-effort heuristic (bare acts are not
perfectly uniform), which is why the 5% manual spot-check exists — a human
still eyeballs the sampled rows before step 4.

KNOWN ISSUE: indiacode.nic.in returns HTTP 403 to plain (non-browser)
requests. download_pdf() sends browser-like headers + the act's handle page as
Referer to work around this; if it still 403s, the site may require a session
cookie obtained by first visiting the handle page in a real browser.

Usage:
    python scripts/03_build_corpus.py                  # all enabled acts + mappings
    python scripts/03_build_corpus.py --act bns_2023    # one act only (no mappings)
    python scripts/03_build_corpus.py --skip-download   # reuse PDFs already in data/raw/
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.corpus import (  # noqa: E402
    extract_pdf_text,
    parse_ncrb_mapping,
    slice_act_body,
    split_articles,
    split_sections,
    validate_sections,
)

CONFIG = ROOT / "configs" / "acts.yaml"
RAW_DIR = ROOT / "data" / "raw" / "acts"
OUT_DIR = ROOT / "data" / "canonical"
REPORTS = ROOT / "reports"

CLEAN_GATE = 0.98
SPOTCHECK_FRACTION = 0.05
DOWNLOAD_RETRIES = 3


def _rel(path: Path) -> Path | str:
    """Repo-relative path for tidy logging; falls back to the full path when
    the output lives outside the repo (e.g. redirected in a test)."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}


def download_pdf(
    url: str,
    dest: Path,
    session,
    referer: str | None = None,
    skip_download: bool = False,
) -> bool:
    """Download `url` to `dest` with browser-like headers + retries.

    Returns True if the file is present afterwards (already-cached or freshly
    downloaded), False if the download failed. indiacode.nic.in 403s requests
    without a real User-Agent and Referer, so both are sent; NCRB mapping PDFs
    need neither, hence `referer` is optional.
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] {dest.name}")
        return True
    if skip_download:
        print(f"  [missing] {dest.name} — --skip-download set, nothing to reuse")
        return False

    headers = dict(BROWSER_HEADERS)
    if referer:
        headers["Referer"] = referer
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            resp = session.get(url, headers=headers, timeout=180)
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                print(f"  [warn] {url} did not return a PDF (got {resp.content[:8]!r})")
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            print(f"  [download] {dest.name}")
            return True
        except requests.RequestException as e:
            print(f"  [retry {attempt + 1}/{DOWNLOAD_RETRIES}] {url}: {e}")
            if attempt < DOWNLOAD_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    print(f"  [FAILED] could not download {url}")
    return False


def build_act(act: dict, session, skip_download: bool = False) -> tuple[list[dict], dict]:
    """Download + extract one act into StatuteSection-shaped rows + a report.

    Raises FileNotFoundError if the PDF could not be obtained — the caller
    records the act as failed.
    """
    pdf = RAW_DIR / f"{act['act_id']}.pdf"
    if not download_pdf(
        act["url"], pdf, session, referer=act.get("handle_url"), skip_download=skip_download
    ):
        raise FileNotFoundError(f"no PDF for {act['act_id']}")

    text = extract_pdf_text(pdf)
    splitter = split_articles if act.get("splitter") == "articles" else split_sections
    sections = splitter(slice_act_body(text))
    report = validate_sections(sections, expected_count=act.get("expected_sections"))

    rows = [
        {
            "act_id": act["act_id"],
            "act_name": act["act_name"],
            "section": s["section"],
            "title": s["title"],
            "text": s["text"],
            "chapter": s["chapter"],
            "subsection": None,
            "effective_date": act.get("effective_date"),
            "replaces": None,
            "punishment_summary": None,
            "tags": [],
            "source_url": act["url"],
        }
        for s in sections
    ]
    return rows, report


def build_mappings(mapping_configs: list[dict], session, skip_download: bool = False) -> dict:
    """NCRB corresponding-section tables -> data/canonical/law_mappings.jsonl."""
    rows, counts = [], {}
    for m in mapping_configs:
        pdf = RAW_DIR / "mappings" / f"{m['mapping_id']}.pdf"
        if not download_pdf(m["url"], pdf, session, skip_download=skip_download):
            print(f"  [skip] {m['mapping_id']}: PDF unavailable")
            continue
        pairs = sorted(set(parse_ncrb_mapping(pdf)))
        counts[m["mapping_id"]] = len(pairs)
        print(f"  [{m['mapping_id']}] {m['old_act']}->{m['new_act']}: {len(pairs)} pairs")
        rows.extend(
            {
                "old_act": m["old_act"],
                "old_section": old,
                "new_act": m["new_act"],
                "new_section": new,
                "note": None,
            }
            for new, old in pairs
        )
    if rows:
        out = OUT_DIR / "law_mappings.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  -> {_rel(out)} ({len(rows)} rows)")
    return counts


def spot_check_sample(rows: list[dict], fraction: float = SPOTCHECK_FRACTION) -> list[dict]:
    """Seeded ~`fraction` sample of {section, title, text} for manual review."""
    if not rows:
        return []
    k = max(3, round(len(rows) * fraction))
    sample = random.Random(42).sample(rows, min(k, len(rows)))
    return [{"section": r["section"], "title": r["title"], "text": r["text"]} for r in sample]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act", help="build a single act_id from configs/acts.yaml")
    parser.add_argument("--skip-download", action="store_true", help="reuse PDFs already in data/raw/")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if args.act:
        acts = [a for a in config["acts"] if a["act_id"] == args.act]
        if not acts:
            sys.exit(f"unknown act_id {args.act!r}")
    else:
        acts = [a for a in config["acts"] if a.get("enabled")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    full_report, spotcheck, failed = {}, {}, []
    for act in acts:
        print(f"[{act['act_id']}] {act['act_name']}")
        try:
            rows, report = build_act(act, session, skip_download=args.skip_download)
        except FileNotFoundError as e:
            print(f"  [FAILED] {e}")
            failed.append(act["act_id"])
            continue
        out = OUT_DIR / f"{act['act_id']}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        clean = report["clean_fraction"]
        print(
            f"  sections={report['extracted']} expected={report['expected']} "
            f"clean={clean:.2%} -> {_rel(out)}"
        )
        full_report[act["act_id"]] = report
        spotcheck[act["act_id"]] = spot_check_sample(rows)
        if clean < CLEAN_GATE:
            failed.append(act["act_id"])

    mapping_counts = {}
    if not args.act and config.get("mappings"):
        print("[mappings] NCRB corresponding-section tables")
        mapping_counts = build_mappings(config["mappings"], session, skip_download=args.skip_download)

    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "gate": CLEAN_GATE}
    (REPORTS / "corpus_extraction_report.json").write_text(
        json.dumps({"meta": meta, "acts": full_report, "mappings": mapping_counts}, indent=2),
        encoding="utf-8",
    )
    (REPORTS / "corpus_spotcheck_sample.json").write_text(
        json.dumps(spotcheck, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[out] {REPORTS / 'corpus_extraction_report.json'}")
    print(f"[out] {REPORTS / 'corpus_spotcheck_sample.json'} (manual 5% spot-check)")

    if failed:
        sys.exit(f"GATE FAILED (<{CLEAN_GATE:.0%} clean or download failed): {', '.join(failed)}")
    if full_report:
        print(f"[gate] all acts >= {CLEAN_GATE:.0%} clean")


if __name__ == "__main__":
    main()
