"""Step 36 — Fact recall split by whether the gold statute reached the context.

The project's headline claim -- "the model is not the bottleneck: accuracy is
~63% when the right section is in context and ~17% when it is not" -- was
computed ad hoc and lived only in prose. This recomputes it from the committed
Eval-v1 predictions so it is traceable, and does so for every run at once.

For each scored record whose required facts name a section (resolved with the
same citation parser the retriever uses), the record lands in one bucket:
  gold_in_context  every gold section was in the retrieved list
  gold_missing     at least one gold section was not retrieved
  no_gold          the record's facts are phrase-only (no section to look for)

Writes reports/eval_v1_retrieval_outcome.json.

Usage:
    python scripts/36_retrieval_outcome.py                      # every run under outputs/eval-v1
    python scripts/36_retrieval_outcome.py --label base --label nyaya-3b-v3
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.retrieval import load_statute_index  # noqa: E402
from nyaya.scoring import score_record  # noqa: E402

RUNS = ROOT / "outputs" / "eval-v1"
REPORT = ROOT / "reports" / "eval_v1_retrieval_outcome.json"


def gold_keys(record: dict, index) -> set[str]:
    keys: set[str] = set()
    for fact in record.get("required_facts", []):
        keys.update(index.referenced_keys(fact, domain=record.get("legal_domain")))
    return keys


def outcome_for_run(label: str, index) -> dict:
    path = RUNS / label / "predictions.jsonl"
    preds = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    buckets: dict[str, list[float]] = {"gold_in_context": [], "gold_missing": [], "no_gold": []}
    for p in preds:
        record = p["record"]
        if record.get("task_type") == "safety_abstention":
            continue
        gold = gold_keys(record, index)
        recall = score_record(record, p["response"])["fact_recall"]
        retrieved = set(p.get("retrieved", []))
        if not gold:
            buckets["no_gold"].append(recall)
        elif gold <= retrieved:
            buckets["gold_in_context"].append(recall)
        else:
            buckets["gold_missing"].append(recall)
    return {name: {"n": len(v), "fact_recall": round(sum(v) / len(v), 4) if v else None}
            for name, v in buckets.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", action="append", help="run label under outputs/eval-v1 (repeatable)")
    args = p.parse_args()
    labels = args.label or sorted(d.name for d in RUNS.iterdir() if (d / "predictions.jsonl").exists())

    index = load_statute_index(ROOT / "data" / "canonical")
    report = {label: outcome_for_run(label, index) for label in labels}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'run':<16}{'gold in ctx':>16}{'gold missing':>16}{'no gold':>14}")
    for label, r in report.items():
        cells = [f"{r[b]['fact_recall']:.1%} (n={r[b]['n']})" if r[b]["n"] else "-"
                 for b in ("gold_in_context", "gold_missing", "no_gold")]
        print(f"{label:<16}{cells[0]:>16}{cells[1]:>16}{cells[2]:>14}")
    print(f"\n[outcome] wrote {REPORT}")


if __name__ == "__main__":
    main()
