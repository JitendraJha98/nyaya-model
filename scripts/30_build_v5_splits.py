"""Step 30 — assemble the v5 training splits (Milestone 2).

Combines the two rule-based generators into train/val:

    scripts/29  grounded citation selection + verbatim copying (RAG-formatted)
    scripts/19  extraction QA + old->new law mapping

Split by SOURCE SECTION, never by row. Two questions generated from BNS 92
share almost all of their text, so a row-wise split would put near-duplicates
on both sides and report a validation loss that measures memorisation. This is
a standing project rule, and it is enforced here with an assertion rather than
a convention.

No eval leakage is possible by construction — both generators exclude
eval-reachable sections — but it is re-checked here, because "by construction"
is how the earlier measurement bugs all described themselves.

Usage:
    python scripts/30_build_v5_splits.py
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SOURCES = [
    ROOT / "data" / "generated" / "grounding_v1.jsonl",
    ROOT / "data" / "generated" / "extraction_qa_v1.jsonl",
]
OUT_DIR = ROOT / "data" / "splits_v5"
REPORT = ROOT / "reports" / "v5_dataset_report.json"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[v5] {path} missing — run scripts/29 and scripts/19 first")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _section_key(record: dict) -> tuple:
    return tuple(sorted(record["metadata"].get("source_sections") or ["_none"]))


def split(records: list[dict], val_fraction: float, seed: int):
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(_section_key(r), []).append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    cut = int(len(keys) * (1 - val_fraction))
    train = [r for k in keys[:cut] for r in groups[k]]
    val = [r for k in keys[cut:] for r in groups[k]]
    return train, val


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    records = [r for path in SOURCES for r in _load(path)]
    train, val = split(records, args.val_fraction, args.seed)

    # Enforced, not assumed: no source section may appear on both sides.
    train_secs = {s for r in train for s in (r["metadata"].get("source_sections") or [])}
    val_secs = {s for r in val for s in (r["metadata"].get("source_sections") or [])}
    straddling = train_secs & val_secs
    if straddling:
        sys.exit(f"[v5] LEAK: {len(straddling)} sections in both splits, "
                 f"e.g. {sorted(straddling)[:3]}")

    # Re-check eval exclusion rather than trusting the generators.
    import importlib.util
    from nyaya.evaluation import load_eval_records
    from nyaya.retrieval import load_statute_index
    spec = importlib.util.spec_from_file_location(
        "gen19", ROOT / "scripts" / "19_generate_extraction_data.py")
    gen19 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen19)
    excluded = gen19.eval_excluded_keys(
        load_statute_index(str(ROOT / "data" / "canonical")), load_eval_records())
    leaked = (train_secs | val_secs) & excluded
    if leaked:
        sys.exit(f"[v5] EVAL LEAK: {len(leaked)} eval-reachable sections, "
                 f"e.g. {sorted(leaked)[:3]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "train": len(train),
        "val": len(val),
        "sections_train": len(train_secs),
        "sections_val": len(val_secs),
        "sections_straddling": 0,
        "eval_sections_excluded": len(excluded),
        "eval_leak": 0,
        "by_task_type": dict(Counter(r["metadata"]["task_type"] for r in records)),
        "by_subtype": dict(Counter(
            r["metadata"].get("subtype", "-") for r in records)),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[v5] wrote {OUT_DIR}/train.jsonl and val.jsonl")


if __name__ == "__main__":
    main()
