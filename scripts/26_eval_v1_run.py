"""Step 26 — Run a model against Nyaya-Eval-v1 and score it with the v1 metric.

This is the run that finally answers "did the training help?". None of the
published numbers could: Eval-v0's metric scored its own gold answers at
~10.7%, so base / v3 / v4 were pinned within a 2-answer spread by the ruler,
not by the models.

WHY THIS SCRIPT SAVES RAW PREDICTIONS
-------------------------------------
The v1..v4 runs kept only aggregate JSON. `outputs/` is empty, so when the
scorer was found to be broken there was nothing to re-score — every earlier
result had to be thrown away and the models re-run on a GPU we no longer have.
That must not happen twice. Every run here writes predictions.jsonl with the
raw model text, and `--rescore` re-grades a saved run on CPU with no model
loaded at all. Metrics are cheap and revisable; generations are expensive and
gone forever if you do not keep them.

Usage:
    # base model, dense RAG
    python scripts/26_eval_v1_run.py --adapter none --dense --label base

    # the published merged release straight from the Hub
    python scripts/26_eval_v1_run.py --model NyayaLabs98/nyaya-3b-v3 \
        --adapter none --dense --label v3

    # re-grade a finished run on CPU — no GPU, no model download
    python scripts/26_eval_v1_run.py --rescore outputs/eval-v1/base/predictions.jsonl
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.scoring import aggregate, score_record  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
EVAL_FILES = {
    "public": ROOT / "data" / "eval" / "nyaya_eval_v1_public.jsonl",
    "private": ROOT / "data" / "eval" / "nyaya_eval_v1_private.jsonl",
    "all": ROOT / "data" / "eval" / "nyaya_eval_v1.jsonl",
}


def _results_path(out_dir: Path) -> Path:
    """Metrics live next to the predictions' --out-dir, not always in the repo."""
    return out_dir / "reports" / "eval_v1_results.json"


def _rag_helpers():
    """Reuse script 16's model/RAG plumbing rather than forking it."""
    spec = importlib.util.spec_from_file_location(
        "rag_eval_16", ROOT / "scripts" / "16_rag_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_with_oom_retry(generate, questions: list[str]) -> list[str]:
    """Generate, halving the batch on CUDA OOM instead of losing the run.

    Retrieval-augmented prompts vary hugely in length -- k=8 statute passages
    can be a few hundred tokens or a few thousand -- so a batch size that is
    fine for 200 questions can OOM on the 201st. On a 14.5 GiB T4 that killed
    a 40-minute run outright. Back off to smaller batches instead, and if even
    a single prompt will not fit, record it as an empty answer (scored as a
    miss, which is honest) rather than aborting everything.
    """
    import torch

    try:
        return generate(questions)
    except torch.cuda.OutOfMemoryError:  # torch.OutOfMemoryError alias only exists from 2.5
        torch.cuda.empty_cache()
        if len(questions) == 1:
            print(f"\n[eval-v1] ! OOM on a single prompt ({len(questions[0])} chars); "
                  f"recording an empty answer", flush=True)
            return [""]
        mid = len(questions) // 2
        print(f"\n[eval-v1] ! OOM at batch {len(questions)} -> splitting", flush=True)
        return (generate_with_oom_retry(generate, questions[:mid])
                + generate_with_oom_retry(generate, questions[mid:]))


def load_records(split: str, limit: int | None = None) -> list[dict]:
    path = EVAL_FILES[split]
    if not path.exists():
        sys.exit(f"[eval-v1] {path} not found — run scripts/25_build_eval_v1.py first")
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Records with no gradeable fact carry no signal; they are kept in the file
    # so the rewrite work stays visible, but they are not scored.
    records = [r for r in records if not r.get("needs_curation")]
    return records[:limit] if limit else records


def score_predictions(predictions: list[dict]) -> dict:
    scored = [score_record(p["record"], p["response"]) for p in predictions]
    metrics = aggregate(scored)
    metrics["mean_latency_s"] = round(
        sum(p.get("latency_s", 0) for p in predictions) / max(1, len(predictions)), 3)
    return metrics


def _print_metrics(label: str, metrics: dict) -> None:
    print(f"\n[eval-v1] === {label} ===")
    print(f"  fact_recall        : {metrics['fact_recall']:.1%}   <- headline")
    print(f"  citation_accuracy  : {metrics['citation_accuracy']:.1%}"
          f"  (n={metrics['citation_rows']})")
    print(f"  substance_recall   : {metrics['substance_recall']:.1%}")
    print(f"  all_facts (v0-like): {metrics['all_facts_accuracy']:.1%}")
    print(f"  forbidden violations: {metrics['forbidden_violation_rate']:.1%}")
    print(f"  scored             : {metrics['scored_total']}")


def _save(out_dir: Path, label: str, predictions: list[dict], metrics: dict, meta: dict) -> None:
    run_dir = out_dir / "outputs" / "eval-v1" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for p in predictions:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    results_path = _results_path(out_dir)
    results = {}
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except ValueError:
            results = {}
    results[label] = {"meta": meta, "metrics": metrics}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[eval-v1] predictions -> {run_dir / 'predictions.jsonl'}")
    print(f"[eval-v1] metrics     -> {results_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rescore", help="Score a saved predictions.jsonl on CPU; no model loaded")
    p.add_argument("--model", default=MODEL_ID, help="Base model id, merged dir, or Hub repo")
    p.add_argument("--adapter", default="none", help='LoRA adapter dir, or "none"')
    p.add_argument("--split", choices=sorted(EVAL_FILES), default="all")
    p.add_argument("--dense", action="store_true", help="hybrid BM25 + e5 retrieval")
    p.add_argument("--rewrite", action="store_true",
                   help="rewrite Hindi/Hinglish questions into statutory English before retrieval (nyaya.rewrite)")
    p.add_argument("--no-rag", action="store_true")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--label")
    p.add_argument("--out-dir", default=str(ROOT))
    args = p.parse_args()

    # --- CPU path: re-grade a saved run -----------------------------------
    if args.rescore:
        path = Path(args.rescore)
        predictions = [json.loads(l) for l in
                       path.read_text(encoding="utf-8").splitlines() if l.strip()]
        label = args.label or path.parent.name
        metrics = score_predictions(predictions)
        _print_metrics(f"{label} (rescored)", metrics)
        _save(Path(args.out_dir), label, predictions, metrics,
              {"rescored_from": str(path), "scorer": "nyaya.scoring (Eval-v1)"})
        return

    # --- GPU path: generate then grade -------------------------------------
    helpers = _rag_helpers()
    records = load_records(args.split, args.limit)
    adapter = None if args.adapter.lower() == "none" else args.adapter
    label = args.label or (
        "base" if args.model == MODEL_ID and not adapter
        else Path(args.model).name + (f"+{Path(adapter).name}" if adapter else ""))

    index = None
    if not args.no_rag:
        index = helpers.load_statute_index(str(ROOT / "data" / "canonical"))
        if args.dense:
            from nyaya.dense import attach_dense_index
            attach_dense_index(
                index, cache_path=ROOT / "data" / "generated" / "e5_doc_vectors.npy")

    print(f"[eval-v1] {label}: {len(records)} questions, model={args.model}, "
          f"adapter={adapter or 'none'}, k={'off' if args.no_rag else args.k}")

    tokenizer, model = helpers.load_model(adapter, model_id=args.model)
    retrieval_log: dict = {}
    rewrite_log: dict = {}
    generate = helpers.build_rag_generate_fn(
        tokenizer, model, index, args.k, retrieval_log,
        max_new_tokens=args.max_new_tokens,
        rewrite=args.rewrite, rewrite_log=rewrite_log)

    predictions = []
    t0 = time.time()
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        t1 = time.perf_counter()
        responses = generate_with_oom_retry(generate, [r["question"] for r in batch])
        latency = (time.perf_counter() - t1) / len(batch)
        for record, response in zip(batch, responses):
            predictions.append({
                "id": record["id"],
                "question": record["question"],
                "response": response,
                "retrieved": retrieval_log.get(record["question"], []),
                "rewritten_query": rewrite_log.get(record["question"]),
                "latency_s": round(latency, 3),
                # The record travels with the prediction so --rescore is
                # self-contained and never has to re-join against the eval file.
                "record": record,
            })
        done = min(start + args.batch_size, len(records))
        print(f"\r[eval-v1]   {done}/{len(records)}", end="", flush=True)
        # \r-overwritten progress never reaches line-based logs (Kaggle shows
        # nothing between [load] and the final metrics — an hour of silence
        # that reads as a hang). Emit a real line with an ETA periodically.
        if done % 48 == 0 or done == len(records):
            elapsed = time.time() - t0
            eta = elapsed / done * (len(records) - done)
            print(f"\n[eval-v1] progress {done}/{len(records)} "
                  f"elapsed {elapsed:.0f}s eta ~{eta:.0f}s", flush=True)

    elapsed = round(time.time() - t0, 1)
    metrics = score_predictions(predictions)
    _print_metrics(label, metrics)
    print(f"  wall clock         : {elapsed}s")
    _save(Path(args.out_dir), label, predictions, metrics, {
        "model": args.model,
        "adapter": adapter or "none",
        "split": args.split,
        "rag": not args.no_rag,
        "dense": args.dense,
        "rewrite": args.rewrite,
        "k": args.k,
        "max_new_tokens": args.max_new_tokens,
        "questions": len(records),
        "eval_seconds": elapsed,
        "scorer": "nyaya.scoring (Eval-v1)",
    })


if __name__ == "__main__":
    main()
