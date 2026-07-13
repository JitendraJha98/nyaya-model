"""Step 3 — Build the raw legal corpus: the statute DB (single source of truth).

For every enabled act in configs/acts.yaml:
    download the official India Code PDF into data/raw/ (atomic, cached) ->
    extract text (footnote/superscript-aware, src/nyaya/corpus.py) ->
    slice act body (ToC/schedules trimmed) -> split into sections ->
    validate (count vs expected, numbering monotonicity/gaps, empty bodies) ->
    write data/canonical/<act_id>.jsonl (StatuteSection rows).

Also writes:
    reports/corpus_extraction_report.json   per-act quality metrics
    reports/corpus_spotcheck_sample.json    ~5% random sections per act, for
                                            the manual spot-check the roadmap
                                            requires before synthetic generation

Go/no-go gate: exits non-zero if any enabled act extracts < 98% clean.

Usage:
    python scripts/03_build_corpus.py            # all enabled acts
    python scripts/03_build_corpus.py --act bns_2023
"""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.corpus import (
    extract_pdf_text,
    parse_ncrb_mapping,
    slice_act_body,
    split_sections,
    validate_sections,
)

CONFIG = ROOT / "configs" / "acts.yaml"
RAW_DIR = ROOT / "data" / "raw" / "acts"
OUT_DIR = ROOT / "data" / "canonical"
REPORTS = ROOT / "reports"

CLEAN_GATE = 0.98
SPOTCHECK_FRACTION = 0.05


def download_pdf(url: str, target: Path) -> Path:
    """Cached, atomic download (same discipline as scripts/00)."""
    if target.exists() and target.stat().st_size > 0:
        print(f"  [cached] {target.name}")
        return target
    print(f"  [download] {url}")
    tmp = target.with_suffix(".tmp")
    response = requests.get(
        url, timeout=180, headers={"User-Agent": "Mozilla/5.0 (research; nyaya-model)"}
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "pdf" not in content_type:
        raise ValueError(f"expected a PDF, got {content_type!r} from {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(response.content)
    tmp.rename(target)
    return target


def build_act(act: dict) -> tuple[list[dict], dict]:
    pdf = download_pdf(act["url"], RAW_DIR / f"{act['act_id']}.pdf")
    text = extract_pdf_text(pdf)
    sections = split_sections(slice_act_body(text))
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


def build_mappings(mapping_configs: list[dict]) -> dict:
    """NCRB corresponding-section tables -> data/canonical/law_mappings.jsonl."""
    rows, counts = [], {}
    for m in mapping_configs:
        pdf = download_pdf(m["url"], RAW_DIR / "mappings" / f"{m['mapping_id']}.pdf")
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
    out = OUT_DIR / "law_mappings.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  -> {out.relative_to(ROOT)} ({len(rows)} rows)")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act", help="build a single act_id from the config")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    acts = [a for a in config["acts"] if a.get("enabled")]
    if args.act:
        acts = [a for a in config["acts"] if a["act_id"] == args.act]
        if not acts:
            sys.exit(f"unknown act_id {args.act!r}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    full_report, spotcheck, failed = {}, {}, []
    for act in acts:
        print(f"[{act['act_id']}] {act['act_name']}")
        rows, report = build_act(act)
        out = OUT_DIR / f"{act['act_id']}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"  sections={report['extracted']} expected={report['expected']} "
            f"clean={report['clean_fraction']:.2%} -> {out.relative_to(ROOT)}"
        )
        full_report[act["act_id"]] = report
        if report["clean_fraction"] < CLEAN_GATE:
            failed.append(act["act_id"])

        sampler = random.Random(42)
        k = max(3, round(len(rows) * SPOTCHECK_FRACTION))
        spotcheck[act["act_id"]] = [
            {"section": r["section"], "title": r["title"], "text": r["text"]}
            for r in sampler.sample(rows, min(k, len(rows)))
        ]

    mapping_counts = {}
    if not args.act and config.get("mappings"):
        print("[mappings] NCRB corresponding-section tables")
        mapping_counts = build_mappings(config["mappings"])

    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "gate": CLEAN_GATE}
    (REPORTS / "corpus_extraction_report.json").write_text(
        json.dumps(
            {"meta": meta, "acts": full_report, "mappings": mapping_counts}, indent=2
        ),
        encoding="utf-8",
    )
    (REPORTS / "corpus_spotcheck_sample.json").write_text(
        json.dumps(spotcheck, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[out] {REPORTS / 'corpus_extraction_report.json'}")
    print(f"[out] {REPORTS / 'corpus_spotcheck_sample.json'} (manual 5% spot-check)")

    if failed:
        sys.exit(f"GATE FAILED (<{CLEAN_GATE:.0%} clean): {', '.join(failed)}")
    print(f"[gate] all acts >= {CLEAN_GATE:.0%} clean")


if __name__ == "__main__":
    main()
