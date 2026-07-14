"""Step 6 — Near-duplicate removal across the validated dataset.

Repetitive data is the classic cause of "loss looks great, answers are bad".
Uses validators.is_near_duplicate (token-Jaccard prefilter + SequenceMatcher,
threshold 0.92) over user questions; first occurrence wins. O(n^2) with the
prefilter is fine at <=25K examples; swap in MinHash if scale demands.

Input:  data/validated/<version>_validated.jsonl
Output: data/validated/<version>_deduped.jsonl + counts appended to
        reports/validation_report.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.dataset import load_jsonl
from nyaya.validators import is_near_duplicate

REPORTS = ROOT / "reports"


def question_of(record: dict) -> str:
    return next(m["content"] for m in record["messages"] if m["role"] == "user")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="nyaya_instruct_v1")
    args = parser.parse_args()

    in_file = ROOT / "data" / "validated" / f"{args.version}_validated.jsonl"
    out_file = ROOT / "data" / "validated" / f"{args.version}_deduped.jsonl"
    records = load_jsonl(in_file)

    kept: list[dict] = []
    kept_questions: list[str] = []
    dropped = 0
    for record in records:
        q = question_of(record)
        if any(is_near_duplicate(q, prev) for prev in kept_questions):
            dropped += 1
            continue
        kept.append(record)
        kept_questions.append(q)

    with out_file.open("w", encoding="utf-8") as fh:
        for record in kept:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_file = REPORTS / "validation_report.json"
    report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else {}
    report["dedup"] = {
        "input": len(records), "kept": len(kept), "dropped_near_duplicates": dropped,
        "deduped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[dedup] {len(records)} -> {len(kept)} (dropped {dropped}) "
          f"-> {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
