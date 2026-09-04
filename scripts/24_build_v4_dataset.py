"""Step 19 (v4) — Assemble Nyaya-Instruct-v4 splits.

Two ingredients, one output (data/splits_rag_v4):

1. RAFT records from data/generated/nyaya_instruct_v4_raw.jsonl — teacher
   answers under the MERGED retriever's RAG prompts (guidance appendix,
   dense fusion). Selection identical to scripts/21: first gate-passing
   sample per task, bare-question leakage check, split inherited from v1's
   grouped split.

2. Rule-based extraction-QA records (scripts/19 output) — validated,
   deduped, grouped-split with the same held-out acts, then wrapped in the
   inference RAG prompt (gold-injected context) like every other example.

Usage:
    python scripts/24_build_v4_dataset.py [--dense]
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.dataset import grouped_split, load_jsonl
from nyaya.evaluation import load_eval_records
from nyaya.retrieval import build_rag_training_record, load_statute_index
from nyaya.validators import (
    detect_eval_leakage,
    is_near_duplicate,
    load_statute_db,
    validate_example,
)

HOLDOUT_ACTS = ["posh_2013", "mv_act_1988"]  # same as scripts/07
SPLITS = ("train", "val", "test")


def select_raft(raw_path: Path, eval_records) -> dict[str, list[dict]]:
    """First gate-passing sample per task; bare-question leakage check."""
    by_task = defaultdict(list)
    for rec in load_jsonl(raw_path):
        task_id = rec["metadata"]["run_id"].rsplit("_s", 1)[0]
        by_task[task_id].append(rec)
    out = {s: [] for s in SPLITS}
    leaked = 0
    for task_id, recs in by_task.items():
        rec = min(recs, key=lambda r: r["metadata"]["sample_id"])
        probe = {"messages": [
            {"role": "user", "content": rec["metadata"]["rag"]["question"]},
            {"role": "assistant", "content": rec["messages"][-1]["content"]},
        ]}
        if detect_eval_leakage(probe, eval_records):
            leaked += 1
            continue
        out[rec["metadata"]["split"]].append(rec)
    return out, leaked


def prepare_extraction(path: Path, index, eval_records, statute_db,
                       k: int) -> tuple[dict[str, list[dict]], dict]:
    records = load_jsonl(path)
    kept, rejected = [], 0
    seen_answers: list[str] = []
    seen_fact_keys: set[tuple] = set()
    for rec in records:
        ok, _reasons = validate_example(rec, statute_db, eval_records)
        if not ok:
            rejected += 1
            continue
        # Template records are similar in FORM by design while carrying
        # distinct facts — text-similarity dedup is the wrong gate for them.
        # scripts/19 guarantees one record per (section, template); dedup on
        # that key instead.
        if rec["metadata"].get("generator", "").startswith("rule_"):
            key = (rec["metadata"]["task_type"],
                   tuple(rec["metadata"]["source_sections"]),
                   rec["messages"][1]["content"][:60])
            if key in seen_fact_keys:
                rejected += 1
                continue
            seen_fact_keys.add(key)
        else:
            answer = rec["messages"][-1]["content"]
            if any(is_near_duplicate(answer, prev) for prev in seen_answers[-200:]):
                rejected += 1
                continue
            seen_answers.append(answer)
        kept.append(rec)
    splits = grouped_split(kept, val_fraction=0.05, test_fraction=0.1,
                           seed=42, holdout_acts=HOLDOUT_ACTS)
    out = {s: [] for s in SPLITS}
    for split in SPLITS:
        for rec in splits.get(split, []):
            out[split].append(build_rag_training_record(rec, index, k=k))
    stats = {"in": len(records), "rejected": rejected,
             "kept": len(kept), "per_split": {s: len(out[s]) for s in SPLITS}}
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raft-raw", default=str(ROOT / "data" / "generated" /
                                                  "nyaya_instruct_v4_raw.jsonl"))
    parser.add_argument("--extraction", default=str(ROOT / "data" / "generated" /
                                                    "extraction_qa_v1.jsonl"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "splits_rag_v4"))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args()

    eval_records = load_eval_records()
    statute_db = load_statute_db(include_old_law=True)
    index = load_statute_index(ROOT / "data" / "canonical")
    if args.dense:
        from nyaya.dense import (
            DEFAULT_ATTACH_MODEL,
            attach_dense_index,
            doc_vector_cache,
        )
        attach_dense_index(index, cache_path=doc_vector_cache(
            DEFAULT_ATTACH_MODEL, ROOT / "data" / "generated"))

    raft, leaked = select_raft(Path(args.raft_raw), eval_records)
    extraction, ex_stats = prepare_extraction(
        Path(args.extraction), index, eval_records, statute_db, args.k)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    rng = random.Random(42)
    for split in SPLITS:
        rows = raft[split] + extraction[split]
        rng.shuffle(rows)
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = {"total": len(rows), "raft": len(raft[split]),
                         "extraction": len(extraction[split])}

    report = {
        "raft_leakage_dropped": leaked,
        "extraction": ex_stats,
        "splits": counts,
        "miss_examples": sum(1 for s in SPLITS for r in raft[s]
                             if r["metadata"]["rag"]["is_miss"]),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (ROOT / "reports" / "v4_dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
