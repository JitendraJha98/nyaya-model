"""Step 16 (v3) — DPO preference pairs from RAFT multi-sample generation.

chosen  = a gate-passing answer for a task
rejected = a gate-FAILING answer for the SAME task (cited outside context,
           invented citations on a miss, or skipped citing when gold was shown)

Pairs come exclusively from generated data — never from frozen-eval
predictions (that would train on the benchmark). TRAIN-split tasks only.

Reads  data/generated/nyaya_instruct_v3_raw.jsonl + _rejected.jsonl
Writes data/dpo/pairs.jsonl (prompt messages, chosen, rejected)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(ROOT / "data" / "generated" /
                                             "nyaya_instruct_v3_raw.jsonl"))
    parser.add_argument("--rejected", default=str(ROOT / "data" / "generated" /
                                                  "nyaya_instruct_v3_rejected.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "data" / "dpo" / "pairs.jsonl"))
    args = parser.parse_args()

    ok_by_task = defaultdict(list)
    for rec in load_jsonl(args.raw):
        if rec["metadata"]["split"] != "train":
            continue
        task_id = rec["metadata"]["run_id"].rsplit("_s", 1)[0]
        ok_by_task[task_id].append(rec)

    bad_by_task = defaultdict(list)
    for row in load_jsonl(args.rejected):
        if row.get("reason") == "trivial" or row.get("split") != "train":
            continue
        bad_by_task[row["task_id"]].append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs, reasons = 0, defaultdict(int)
    with out_path.open("w", encoding="utf-8") as fh:
        for task_id, bads in bad_by_task.items():
            oks = ok_by_task.get(task_id)
            if not oks:
                continue
            chosen_rec = min(oks, key=lambda r: r["metadata"]["sample_id"])
            for bad in bads:
                fh.write(json.dumps({
                    "prompt": chosen_rec["messages"][:-1],
                    "chosen": chosen_rec["messages"][-1]["content"],
                    "rejected": bad["answer"],
                    "reject_reason": bad["reason"],
                    "task_id": task_id,
                }, ensure_ascii=False) + "\n")
                pairs += 1
                reasons[bad["reason"]] += 1

    report = {"pairs": pairs, "by_reason": dict(reasons),
              "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (ROOT / "reports" / "dpo_pairs_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
