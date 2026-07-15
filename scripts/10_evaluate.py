"""Step 10 — Evaluate checkpoints on Nyaya-Eval-v0 (+ dataset behavioural eval).

Benchmarks EVERY saved checkpoint of a training run against the frozen
500-question eval set with the exact harness used for the baseline — the
best model may be checkpoint-150, not the final one. The winner (by strict
accuracy, language-match tiebreak) additionally gets the dataset behavioural
eval on the test split for direct comparison with reports/dataset_baseline.json.

Do NOT rely on train/val loss — low loss measures fit to a possibly-
hallucinated distribution, not legal correctness.

Outputs:
    reports/checkpoint_evals.json
    outputs/<run>/eval/<checkpoint>/predictions.jsonl (per checkpoint)

Usage:
    python scripts/10_evaluate.py --run-dir outputs/legal-3b-v1
    python scripts/10_evaluate.py --run-dir outputs/legal-3b-v1 --limit 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.dataset import load_jsonl
from nyaya.evaluation import load_eval_records, run_dataset_eval, run_eval
from nyaya.prompts import NYAYA_SYSTEM_PROMPT
from nyaya.validators import load_statute_db

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def load_adapter_model(adapter_dir: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model


def build_generate_fn(tokenizer, model, max_new_tokens: int = 384):
    import torch

    def generate(questions):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": NYAYA_SYSTEM_PROMPT},
                 {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True)
            for q in questions
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)

    return generate


def checkpoint_dirs(run_dir: Path) -> list[Path]:
    numbered = sorted(run_dir.glob("checkpoint-*"),
                      key=lambda p: int(p.name.split("-")[1]))
    final = run_dir / "final"
    return numbered + ([final] if final.exists() else [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/legal-3b-v1")
    parser.add_argument("--limit", type=int, help="eval only the first N frozen questions")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-file", default="data/splits/test.jsonl")
    parser.add_argument("--out-dir", default=str(ROOT))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoints = checkpoint_dirs(run_dir)
    if not checkpoints:
        sys.exit(f"no checkpoints under {run_dir}")
    eval_records = load_eval_records()
    if args.limit:
        eval_records = eval_records[: args.limit]
    statute_db = load_statute_db(include_old_law=True)
    print(f"[eval] {len(checkpoints)} checkpoints x {len(eval_records)} frozen questions")

    results = {}
    for ckpt in checkpoints:
        t0 = time.time()
        tokenizer, model = load_adapter_model(str(ckpt))
        generate = build_generate_fn(tokenizer, model)
        predictions, metrics = run_eval(generate, eval_records,
                                        batch_size=args.batch_size)
        out = Path(args.out_dir) / run_dir / "eval" / ckpt.name
        out.mkdir(parents=True, exist_ok=True)
        with (out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
            for p in predictions:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        results[ckpt.name] = {
            "strict_accuracy": metrics["auto_strict_accuracy"],
            "strict_correct": metrics["auto_strict_correct"],
            "scored_total": metrics["scored_total"],
            "forbidden_fact_violations": metrics["forbidden_fact_violations"],
            "abstention_rate": metrics["abstention_rate"],
            "safety_abstained": metrics["safety_rows"]["abstained"],
            "by_language": {k: f"{v['auto_strict_correct']}/{v['total']}"
                            for k, v in metrics["by_language"].items()},
            "eval_seconds": round(time.time() - t0),
        }
        print(f"  [{ckpt.name}] strict={metrics['auto_strict_accuracy']:.2%} "
              f"violations={metrics['forbidden_fact_violations']} "
              f"({results[ckpt.name]['eval_seconds']}s)")
        del model, tokenizer
        import torch
        torch.cuda.empty_cache()

    best_name = max(results, key=lambda k: (results[k]["strict_accuracy"],
                                            -results[k]["forbidden_fact_violations"]))
    print(f"[best] {best_name}")

    # dataset behavioural eval on the winner (comparable to dataset_baseline.json)
    test_records = load_jsonl(args.test_file)
    tokenizer, model = load_adapter_model(str(run_dir / best_name))
    generate = build_generate_fn(tokenizer, model)
    _, ds_metrics = run_dataset_eval(generate, test_records, statute_db,
                                     batch_size=args.batch_size)
    print(f"[best/dataset] cite={ds_metrics['answers_with_citations']:.0%} "
          f"pass={ds_metrics['citation_pass_rate']:.0%} "
          f"lang={ds_metrics['language_match_rate']:.0%} "
          f"ref_sim={ds_metrics['mean_reference_similarity']:.3f}")

    report_path = Path(args.out_dir) / "reports" / "checkpoint_evals.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "run_dir": str(run_dir), "model": MODEL_ID,
        "frozen_eval_questions": len(eval_records),
        "checkpoints": results, "best": best_name,
        "best_dataset_eval": ds_metrics,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {report_path}")


if __name__ == "__main__":
    main()
