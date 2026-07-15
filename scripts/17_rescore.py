"""Step 13 (v2) — Re-score saved frozen-eval predictions with the current scorer.

The scorer gained word<->digit numeral normalization and a lenient
token-level companion metric after the RAG runs. Answers are expensive
(GPU); judging is free — so every comparison table is rebuilt offline from
the saved predictions.jsonl files, keeping all runs on the same scorer.

Metrics per run:
  strict  — fact_present on every required fact, none forbidden (headline)
  lenient — fact_tokens_present instead (order/adjacency-free); section
            facts keep the citation-context rule

Output: reports/rag_eval_rescored.json

Usage:
    python scripts/17_rescore.py outputs/rag-v2/*/predictions.jsonl \
        outputs/legal-3b-v1/eval/checkpoint-50/predictions.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.dataset import load_jsonl
from nyaya.evaluation import fact_present, fact_tokens_present, load_eval_records


def rescore(predictions: list[dict], records_by_id: dict) -> dict:
    strict = lenient = 0
    scored = 0
    strict_by_retrieval = {}
    for p in predictions:
        rec = records_by_id[p["id"]]
        if rec["task_type"] == "safety_abstention":
            continue
        scored += 1
        response = p["response"]
        violated = any(fact_present(f, response) for f in rec["forbidden_facts"])
        s = (not violated) and all(fact_present(f, response)
                                   for f in rec["required_facts"])
        l = (not violated) and all(fact_tokens_present(f, response)
                                   for f in rec["required_facts"])
        strict += s
        lenient += l
        if "gold_in_context" in p:
            bucket = ("no_gold" if not p.get("gold_sections")
                      else "gold_in_context" if p["gold_in_context"]
                      else "gold_missing")
            cell = strict_by_retrieval.setdefault(bucket, [0, 0, 0])
            cell[0] += 1
            cell[1] += int(s)
            cell[2] += int(l)
    out = {
        "scored_total": scored,
        "strict_correct": strict,
        "strict_accuracy": round(strict / scored, 4),
        "lenient_correct": lenient,
        "lenient_accuracy": round(lenient / scored, 4),
    }
    if strict_by_retrieval:
        out["by_retrieval"] = {
            k: {"n": n, "strict": round(s / n, 4), "lenient": round(l / n, 4)}
            for k, (n, s, l) in sorted(strict_by_retrieval.items())
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_files", nargs="+")
    args = parser.parse_args()

    records_by_id = {r["id"]: r for r in load_eval_records()}
    report = {"rescored_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": {}}
    for path in args.prediction_files:
        label = Path(path).parent.name
        report["runs"][label] = rescore(load_jsonl(path), records_by_id)
        print(label, json.dumps(report["runs"][label], indent=2))

    out = ROOT / "reports" / "rag_eval_rescored.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
