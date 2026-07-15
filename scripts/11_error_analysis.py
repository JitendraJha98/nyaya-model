"""Step 11 — Error analysis. This decides what goes into Nyaya-Instruct-v2.

Thin orchestrator over src/nyaya/evaluation.py::bucket_failures: loads the
frozen-eval predictions of a checkpoint (default: the best v1 checkpoint),
buckets every strict failure by task_type / legal_domain / language /
difficulty, classifies failure MODES (stale-law cited, wrong-or-incomplete
citation, no citation, over-abstention), and lists the most-missed facts.

The loop: DATA -> TRAIN -> EVALUATE -> FAILURE ANALYSIS -> BETTER DATA ->
TRAIN AGAIN. Not: more epochs.

Output: reports/error_analysis.json

Usage:
    python scripts/11_error_analysis.py
    python scripts/11_error_analysis.py --predictions path/to/predictions.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.dataset import load_jsonl
from nyaya.evaluation import bucket_failures

DEFAULT_PREDICTIONS = ROOT / "outputs" / "legal-3b-v1" / "eval" / "checkpoint-50" / "predictions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--label", default="legal-3b-v1/checkpoint-50")
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions)
    report = bucket_failures(predictions)
    report_doc = {
        "checkpoint": args.label,
        "predictions_file": args.predictions,
        "analysed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **report,
    }
    out = ROOT / "reports" / "error_analysis.json"
    out.write_text(json.dumps(report_doc, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in report_doc.items()
                      if k not in ("top_missing_facts",)}, indent=2)[:1600])
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
