"""Step 42 — Build a statute-DB file for any central act from the India Code API.

Replaces PDF splitting for new acts: India Code (DSpace 9, 2026) stores every
section of every central act as a record with number, title and text
(src/nyaya/indiacode.py). Reads the `api_acts` list in configs/acts.yaml, or a
single act given on the command line, and writes data/canonical/<act_id>.jsonl in
the StatuteSection schema plus a validation entry in
reports/corpus_extraction_report.json (same >=98%-clean gate as scripts/03).

After adding an act: register its aliases in src/nyaya/validators.py
(ACT_ALIASES, _ACT_ID_FAMILY) and src/nyaya/retrieval.py (_FAMILY_TO_ACT_ID),
run scripts/38 (punishment summaries), scripts/35 (coverage probe), the tests,
and re-upload the dataset.

Usage:
    python scripts/42_fetch_act_sections.py                           # every api_acts entry
    python scripts/42_fetch_act_sections.py --act-id tpa_1882         # one entry
    python scripts/42_fetch_act_sections.py --title "The Transfer of Property Act, 1882" --act-id tpa_1882 --act-name "Transfer of Property Act, 1882"
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya import indiacode  # noqa: E402
from nyaya.corpus import validate_sections  # noqa: E402

CANON = ROOT / "data" / "canonical"
REPORT = ROOT / "reports" / "corpus_extraction_report.json"
ACTS_YAML = ROOT / "configs" / "acts.yaml"


def build(title: str, act_id: str, act_name: str, expected: int | None = None) -> dict:
    act = indiacode.find_central_act(title)
    if act is None:
        raise SystemExit(f"[api] no central act titled {title!r} on India Code")
    sections = list(indiacode.iter_sections(act["act_id_prefix"]))
    effective = indiacode.to_iso_date(act.get("enforcement_date")) or indiacode.to_iso_date(act.get("enact_date"))
    rows = indiacode.section_rows(act_id, act_name, sections, effective, act["url"])
    # Validate numbering over ALL sections the act has, repealed ones included:
    # a repealed section is a legitimate hole in the live text, not an
    # extraction gap. Only live sections are written to the DB.
    placeholder = "[Repealed or omitted by amendment; number retained for continuity of numbering.]"
    numbering = [{"section": s["section"], "text": s["text"] or placeholder} for s in sorted(sections, key=lambda x: x["order"])]
    report = validate_sections(numbering, expected)
    report.update({"source": "indiacode-api", "handle": act["handle"], "indiacode_act_id": act["act_id"],
                   "live_sections": len(rows),
                   "repealed_or_empty_dropped": len(sections) - len(rows),
                   "repealed_sections": sorted((s["section"] for s in sections if s["repealed"] or not s["text"]),
                                               key=lambda x: int(x) if x.isdigit() else 10**6),
                   "pdfs": [f["name"] for f in act["bitstreams"] if f["name"].lower().endswith(".pdf")]})
    out = CANON / f"{act_id}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"[api] {act_id}: {len(rows)} sections written ({report['repealed_or_empty_dropped']} repealed/empty dropped), "
          f"clean_fraction={report['clean_fraction']}, gaps={report['numbering_gaps'][:8]} -> {out.name}")
    existing = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {"meta": {}, "acts": {}}
    existing.setdefault("acts", {})[act_id] = report
    REPORT.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--act-id", help="our act_id (e.g. tpa_1882); with --title/--act-name defines a one-off act")
    p.add_argument("--title", help="exact India Code title, e.g. 'The Transfer of Property Act, 1882'")
    p.add_argument("--act-name", help="act_name for the rows, e.g. 'Transfer of Property Act, 1882'")
    p.add_argument("--expected", type=int, help="expected section count for the clean gate")
    args = p.parse_args()

    if args.title:
        if not (args.act_id and args.act_name):
            sys.exit("--title needs --act-id and --act-name")
        build(args.title, args.act_id, args.act_name, args.expected)
        return
    entries = (yaml.safe_load(ACTS_YAML.read_text(encoding="utf-8")) or {}).get("api_acts", [])
    if args.act_id:
        entries = [e for e in entries if e["act_id"] == args.act_id]
        if not entries:
            sys.exit(f"[api] {args.act_id} not in configs/acts.yaml api_acts")
    if not entries:
        sys.exit("[api] nothing to do: add entries under api_acts in configs/acts.yaml or pass --title")
    for e in entries:
        if not e.get("enabled", True):
            continue
        build(e["indiacode_title"], e["act_id"], e["act_name"], e.get("expected_sections"))


if __name__ == "__main__":
    main()
