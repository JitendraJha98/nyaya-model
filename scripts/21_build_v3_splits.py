"""Step 15 (v3) — Build Nyaya-Instruct-v3 splits from gated RAFT output.

Selection per task: the first gate-passing sample (sample_id order) — extra
samples exist only as DPO candidates (scripts/22). Split assignment is
inherited from v1's grouped split (metadata.split), preserving the
held-out-acts integrity. Answer-side eval-leakage is checked against the
frozen eval using the BARE question (metadata.rag.question) + new answer —
the RAG prompt itself would mask real overlap.

Reads  data/generated/nyaya_instruct_v3_raw.jsonl
Writes data/splits_rag_v3/{train,val,test}.jsonl + reports/v3_dataset_report.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.dataset import load_jsonl
from nyaya.evaluation import load_eval_records
from nyaya.validators import detect_eval_leakage

SPLITS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(ROOT / "data" / "generated" /
                                             "nyaya_instruct_v3_raw.jsonl"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "splits_rag_v3"))
    args = parser.parse_args()

    raw = load_jsonl(args.raw)
    eval_records = load_eval_records()

    by_task = defaultdict(list)
    for rec in raw:
        task_id = rec["metadata"]["run_id"].rsplit("_s", 1)[0]
        by_task[task_id].append(rec)

    picked, leaked = [], 0
    for task_id, recs in by_task.items():
        rec = min(recs, key=lambda r: r["metadata"]["sample_id"])
        probe = {"messages": [
            {"role": "user", "content": rec["metadata"]["rag"]["question"]},
            {"role": "assistant", "content": rec["messages"][-1]["content"]},
        ]}
        if detect_eval_leakage(probe, eval_records):
            leaked += 1
            continue
        picked.append(rec)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        rows = [r for r in picked if r["metadata"]["split"] == split]
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = len(rows)

    report = {
        "raw_records": len(raw), "tasks": len(by_task),
        "leakage_dropped": leaked, "splits": counts,
        "miss_examples": sum(1 for r in picked if r["metadata"]["rag"]["is_miss"]),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (ROOT / "reports" / "v3_dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
