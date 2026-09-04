"""Step 41 — Build (question, gold section, hard negatives) pairs for retriever training.

The published training set NyayaLabs98/nyaya-train-v3 is 6,429 citizen questions
each paired with the statute sections that ground its answer. That is a free
contrastive training set for a bi-encoder and a small cross-encoder -- the
learned replacement for the hand-written synonym table, whose gains did not
generalise (docs/RESULTS.md §3.5).

For every train-split record that is not a retrieval-miss demonstration:
  query          the citizen question (metadata.rag.question)
  positive_keys  metadata.source_sections that exist in the statute DB
  negative_keys  the top-N BM25 statute sections that are NOT gold (hard negatives)

Questions that near-duplicate an Eval-v1 public question are dropped so the
retriever cannot memorise the benchmark. Writes data/generated/retriever_pairs.jsonl
(gitignored) and reports/retriever_pairs_report.json.

Usage:
    python scripts/41_build_retriever_pairs.py                 # downloads the dataset from the Hub
    python scripts/41_build_retriever_pairs.py --local data/splits_rag_v3/train.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.retrieval import load_statute_index, normalize_gold_keys  # noqa: E402
from nyaya.validators import is_near_duplicate  # noqa: E402

DATASET = "NyayaLabs98/nyaya-train-v3"
OUT = ROOT / "data" / "generated" / "retriever_pairs.jsonl"
REPORT = ROOT / "reports" / "retriever_pairs_report.json"
EVAL = ROOT / "data" / "eval" / "nyaya_eval_v1_public.jsonl"


def question_of(record: dict) -> str:
    rag = (record.get("metadata") or {}).get("rag") or {}
    if rag.get("question"):
        return rag["question"]
    user = next(m["content"] for m in record["messages"] if m["role"] == "user")
    return user.rsplit("Question:", 1)[-1].strip() if "Question:" in user else user.strip()


def build_pairs(records: list[dict], index, eval_questions: list[str], negatives: int = 20) -> tuple[list[dict], dict]:
    pairs, stats = [], {"records": len(records), "miss_skipped": 0, "no_gold_in_db": 0, "eval_near_duplicate": 0, "pairs": 0}
    for r in records:
        meta = r.get("metadata") or {}
        if (meta.get("rag") or {}).get("is_miss"):
            stats["miss_skipped"] += 1
            continue
        gold = [k for k in normalize_gold_keys(meta.get("source_sections", [])) if k in index.by_key]
        if not gold:
            stats["no_gold_in_db"] += 1
            continue
        q = question_of(r)
        if any(is_near_duplicate(q, e, 0.85) for e in eval_questions):
            stats["eval_near_duplicate"] += 1
            continue
        gold_set = set(gold)
        negs = []
        for _s, i in index._bm25(q):
            row = index.rows[i]
            if row["act_id"] == "procedures_kb":
                continue
            key = f"{row['act_id']}:{row['section'].upper()}"
            if key in gold_set:
                continue
            negs.append(key)
            if len(negs) >= negatives:
                break
        pairs.append({"id": r["id"], "query": q, "language": meta.get("language"),
                      "positive_keys": gold, "negative_keys": negs})
    stats["pairs"] = len(pairs)
    return pairs, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local", help="train.jsonl to use instead of downloading the Hub dataset")
    p.add_argument("--negatives", type=int, default=20)
    args = p.parse_args()

    if args.local:
        train_file = Path(args.local)
    else:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=["*.jsonl", "*.json", "*/*.jsonl"]))
        candidates = sorted(root.rglob("train*.jsonl")) or sorted(root.rglob("*.jsonl"))
        if not candidates:
            sys.exit(f"[pairs] no jsonl in {root}: {sorted(str(x.relative_to(root)) for x in root.rglob('*'))[:10]}")
        train_file = candidates[0]
    print(f"[pairs] training records from {train_file}")
    records = [json.loads(l) for l in train_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_questions = [json.loads(l)["question"] for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]

    index = load_statute_index(ROOT / "data" / "canonical")
    pairs, stats = build_pairs(records, index, eval_questions, args.negatives)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pairs), encoding="utf-8")
    stats["source"] = str(train_file.name)
    stats["negatives_per_query"] = args.negatives
    REPORT.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"[pairs] wrote {OUT}")


if __name__ == "__main__":
    main()
