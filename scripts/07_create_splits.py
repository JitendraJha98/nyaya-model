"""Step 7 — Create train/val/test splits. NEVER randomly split rows.

Thin wrapper over src/nyaya/dataset.py::grouped_split: all examples sharing a
metadata.source_sections key land in exactly one split, and two ENTIRE ACTS
(POSH 2013 + MV Act 1988, per the README) are held out of train/val to
measure generalization to unseen statutes.

Nyaya-Eval-v0 (data/eval/) is separate and completely untouched.

Input:  data/validated/<version>_deduped.jsonl
Output: data/splits/{train,val,test}.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.dataset import grouped_split, load_jsonl

OUT_DIR = ROOT / "data" / "splits"
HOLDOUT_ACTS = ["posh_2013", "mv_act_1988"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="nyaya_instruct_v1")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(ROOT / "data" / "validated" / f"{args.version}_deduped.jsonl")
    splits = grouped_split(
        records,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        holdout_acts=HOLDOUT_ACTS,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        langs = Counter(r["metadata"]["language"] for r in rows)
        print(f"[{name}] {len(rows)} examples {dict(langs)} -> {path.relative_to(ROOT)}")
    heldout = sum(
        1 for r in splits["test"] if r["metadata"].get("source_act") in HOLDOUT_ACTS
    )
    print(f"[holdout] {heldout} test examples from held-out acts {HOLDOUT_ACTS}")


if __name__ == "__main__":
    main()
