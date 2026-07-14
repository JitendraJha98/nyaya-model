"""Step 5 — Validate every generated example (one of the most important steps).

Thin orchestrator over src/nyaya/validators.py::validate_example — per
example: schema -> source_sections -> citation verification (deterministic,
against the full statute DB incl. old-law mapping whitelist) -> language/script
consistency -> answer length (80-600 words; safety answers exempt from the
minimum) -> eval-leakage against the FROZEN Nyaya-Eval-v0.

Near-duplicate REMOVAL is the next step, scripts/06_deduplicate.py.

A 15-20% rejection rate is healthy — it means the pipeline is filtering
garbage. Rejections are kept (with reasons) for error analysis.

Input:  data/generated/<version>_raw.jsonl
Output: data/validated/<version>_validated.jsonl
        data/validated/<version>_rejected.jsonl (with rejection reasons)
        reports/validation_report.json
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.dataset import load_jsonl
from nyaya.evaluation import load_eval_records
from nyaya.validators import load_statute_db, validate_example

OUT_DIR = ROOT / "data" / "validated"
REPORTS = ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="nyaya_instruct_v1")
    args = parser.parse_args()

    raw_file = ROOT / "data" / "generated" / f"{args.version}_raw.jsonl"
    records = load_jsonl(raw_file)
    statute_db = load_statute_db(include_old_law=True)
    eval_records = load_eval_records()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    accepted_file = OUT_DIR / f"{args.version}_validated.jsonl"
    rejected_file = OUT_DIR / f"{args.version}_rejected.jsonl"

    accepted, reason_counts = 0, Counter()
    with accepted_file.open("w", encoding="utf-8") as ok_fh, \
         rejected_file.open("w", encoding="utf-8") as bad_fh:
        for record in records:
            ok, reasons = validate_example(record, statute_db, eval_records)
            if ok:
                record["metadata"]["verified"] = True
                ok_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted += 1
            else:
                reason_counts.update(r.split(":", 1)[0] for r in reasons)
                bad_fh.write(json.dumps(
                    {"record": record, "reasons": reasons}, ensure_ascii=False) + "\n")

    report = {
        "version": args.version,
        "generated": len(records),
        "accepted": accepted,
        "rejected": len(records) - accepted,
        "rejection_rate": round(1 - accepted / max(1, len(records)), 4),
        "rejections_by_check": dict(reason_counts),
        "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["rejection_rate"] > 0.30:
        sys.exit("GATE: rejection rate above 30% — inspect the teacher/prompts "
                 "before scaling (docs/ROADMAP.md go/no-go).")


if __name__ == "__main__":
    main()
