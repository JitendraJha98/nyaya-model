"""Step 13 (v2) — RAG eval: retrieval + v1 adapter on the frozen eval set.

The v2 architecture under test: the statute retriever surfaces top-k verbatim
sections, the v1 LoRA adapter (trained citing behaviour) answers from them.
Scored with the exact frozen-eval harness used for the 0.81% baseline and the
v1 checkpoint evals, so the three numbers are directly comparable.

Each prediction is annotated with what was retrieved and whether the gold
sections were in context, so accuracy can be split into retrieval-limited
vs model-limited — that split decides whether v2.1 is a better retriever
or a better prompt/model.

Outputs:
    reports/rag_eval.json
    outputs/rag-v2/<label>/predictions.jsonl

Usage:
    python scripts/16_rag_eval.py --adapter outputs/legal-3b-v1/checkpoint-50
    python scripts/16_rag_eval.py --adapter none            # RAG + base model ablation
    python scripts/16_rag_eval.py --no-rag                  # adapter only (sanity anchor)
"""

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Devanagari on cp1252 Windows consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.evaluation import load_eval_records, run_eval
from nyaya.prompts import NYAYA_SYSTEM_PROMPT
from nyaya.retrieval import build_rag_prompt, load_statute_index

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def load_model(adapter_dir: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model


def build_rag_generate_fn(tokenizer, model, index, k: int,
                          retrieval_log: dict, max_new_tokens: int = 384):
    """generate_fn for run_eval: retrieve per question, answer from context."""
    import torch

    def generate(questions):
        prompts = []
        for q in questions:
            hits = index.retrieve(q, k=k) if index else []
            retrieval_log[q] = [f"{h['act_id']}:{h['section'].upper()}" for h in hits]
            prompts.append(build_rag_prompt(q, hits) if index else q)
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": NYAYA_SYSTEM_PROMPT},
                 {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="outputs/legal-3b-v1/checkpoint-50",
                        help='LoRA adapter dir, or "none" for the base model')
    parser.add_argument("--no-rag", action="store_true",
                        help="skip retrieval (plain adapter, sanity anchor)")
    parser.add_argument("--dense", action="store_true",
                        help="hybrid retrieval: BM25 + multilingual-e5 (RRF)")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--canonical-dir", default=str(ROOT / "data" / "canonical"))
    parser.add_argument("--out-dir", default=str(ROOT))
    parser.add_argument("--label")
    args = parser.parse_args()

    adapter = None if args.adapter.lower() == "none" else args.adapter
    label = args.label or "_".join(filter(None, [
        "rag" if not args.no_rag else "norag",
        "dense" if args.dense else None,
        f"k{args.k}" if not args.no_rag else None,
        f"{Path(adapter).parent.name}-{Path(adapter).name}" if adapter else "base"]))

    index = None if args.no_rag else load_statute_index(args.canonical_dir)
    if index is not None and args.dense:
        from nyaya.dense import attach_dense_index
        attach_dense_index(index,
                           cache_path=ROOT / "data" / "generated" / "e5_doc_vectors.npy")
    records = load_eval_records()
    if args.limit:
        records = records[: args.limit]

    gold_by_question = {}
    if index:
        for rec in records:
            gold = set()
            for fact in rec.get("required_facts", []):
                gold.update(index.referenced_keys(fact))
            gold_by_question[rec["question"]] = gold

    print(f"[rag-eval] {label}: {len(records)} frozen questions, "
          f"adapter={adapter or 'base'}, k={'off' if args.no_rag else args.k}")
    tokenizer, model = load_model(adapter)
    retrieval_log: dict = {}
    generate = build_rag_generate_fn(tokenizer, model, index, args.k, retrieval_log)
    t0 = time.time()
    predictions, metrics = run_eval(generate, records, batch_size=args.batch_size)

    hit_split = {"gold_in_context": [0, 0], "gold_missing": [0, 0], "no_gold": [0, 0]}
    for p in predictions:
        retrieved = retrieval_log.get(p["question"], [])
        gold = gold_by_question.get(p["question"], set())
        p["retrieved"] = retrieved
        p["gold_sections"] = sorted(gold)
        p["gold_in_context"] = bool(gold) and gold <= set(retrieved)
        if p["is_safety_row"]:
            continue
        bucket = ("no_gold" if not gold
                  else "gold_in_context" if p["gold_in_context"] else "gold_missing")
        hit_split[bucket][0] += 1
        hit_split[bucket][1] += int(p["auto_strict_correct"])

    out = Path(args.out_dir) / "outputs" / "rag-v2" / label
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for p in predictions:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    report = {
        "label": label,
        "model": MODEL_ID,
        "adapter": adapter or "none",
        "rag": not args.no_rag,
        "k": None if args.no_rag else args.k,
        "frozen_eval_questions": len(records),
        "strict_accuracy": metrics["auto_strict_accuracy"],
        "strict_correct": metrics["auto_strict_correct"],
        "scored_total": metrics["scored_total"],
        "forbidden_fact_violations": metrics["forbidden_fact_violations"],
        "abstention_rate": metrics["abstention_rate"],
        "safety_rows": metrics["safety_rows"],
        "by_language": metrics["by_language"],
        "accuracy_by_retrieval": {
            name: {"n": n, "strict_correct": c,
                   "accuracy": round(c / n, 4) if n else None}
            for name, (n, c) in hit_split.items()
        },
        "eval_seconds": round(time.time() - t0),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path = Path(args.out_dir) / "reports" / "rag_eval.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if "runs" not in existing:
            existing = {"runs": {}}
    existing.setdefault("runs", {})[label] = report
    report_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "by_language"},
                     ensure_ascii=False, indent=2))
    print(f"[out] {report_path}")


if __name__ == "__main__":
    main()
