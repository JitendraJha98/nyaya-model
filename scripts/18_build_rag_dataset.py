"""Step 14 (v2) — Nyaya-Instruct-v2: RAG-formatted training splits.

Rewraps every Nyaya-Instruct-v1 example in the inference-time RAG prompt
(build_rag_training_record): retrieved top-k context with the example's gold
source_sections force-injected at a per-record shuffled position, question
unchanged, answer unchanged. No teacher calls — contexts are reconstructed
from data/canonical via the stored source_sections.

Sanity gate: the share of examples whose answer cites a section absent from
its context is reported per split — those teach context-defying citations
and abort the build above --max-uncovered.

Reads  data/splits/{train,val,test}.jsonl
Writes data/splits_rag/{train,val,test}.jsonl + reports/rag_dataset_report.json

Usage:
    python scripts/18_build_rag_dataset.py
    python scripts/18_build_rag_dataset.py --k 8
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
from nyaya.retrieval import build_rag_training_record, load_statute_index

SPLITS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--in-dir", default=str(ROOT / "data" / "splits"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "splits_rag"))
    parser.add_argument("--max-uncovered", type=float, default=0.02,
                        help="abort if more than this share of non-abstention "
                             "answers cite sections missing from their context")
    args = parser.parse_args()

    index = load_statute_index(ROOT / "data" / "canonical")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"k": args.k, "splits": {}}
    for split in SPLITS:
        records = load_jsonl(Path(args.in_dir) / f"{split}.jsonl")
        rows_out, uncovered, prompt_words = [], 0, []
        for rec in records:
            out = build_rag_training_record(rec, index, k=args.k)
            context_keys = set(out["metadata"]["rag"]["context_keys"])
            gold = {f"{s.split(':')[0].lower()}:{s.split(':')[1].upper()}"
                    for s in rec["metadata"].get("source_sections", [])}
            if gold and not gold <= context_keys:
                uncovered += 1
                continue  # answer cites what the context lacks — drop, don't teach it
            user = next(m["content"] for m in out["messages"] if m["role"] == "user")
            prompt_words.append(len(user.split()))
            rows_out.append(out)

        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for r in rows_out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        prompt_words.sort()
        stats = {
            "in": len(records),
            "out": len(rows_out),
            "dropped_uncovered": uncovered,
            "prompt_words_p50": prompt_words[len(prompt_words) // 2],
            "prompt_words_p95": prompt_words[int(len(prompt_words) * 0.95)],
            "prompt_words_max": prompt_words[-1],
        }
        report["splits"][split] = stats
        print(f"[{split}] {stats}")
        if uncovered / max(1, len(records)) > args.max_uncovered:
            sys.exit(f"ABORT: {uncovered}/{len(records)} uncovered in {split} "
                     f"(> {args.max_uncovered:.0%}) — gold sections missing from "
                     f"the statute DB?")

    report["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report_path = ROOT / "reports" / "rag_dataset_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[out] {out_dir}  report: {report_path}")


if __name__ == "__main__":
    main()
