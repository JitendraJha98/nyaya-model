"""Step 2 — Benchmark the untouched base model on Nyaya-Eval-v0.

Runs Qwen2.5-3B-Instruct (with NYAYA_SYSTEM_PROMPT, mirroring deploy) against
the eval set and records per question: response, extracted citations,
required/forbidden-fact coverage, abstention flag and latency.

This is the "Qwen2.5-3B-Instruct Baseline" — without it we cannot prove
training improved anything. Uses the frozen eval set if present, else the
draft (clearly stamped in the report; re-run after the freeze).

Outputs:
    outputs/baseline/predictions.jsonl
    reports/baseline.json

Usage:
    python scripts/02_run_baseline.py                 # full run
    python scripts/02_run_baseline.py --limit 25      # smoke run
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.evaluation import DRAFT_EVAL, FROZEN_EVAL, load_eval_records, run_eval, write_outputs
from nyaya.prompts import NYAYA_SYSTEM_PROMPT

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def build_generate_fn(max_new_tokens: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    def generate(questions: list[str]) -> list[str]:
        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": NYAYA_SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in questions
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", default=None, help="override eval jsonl path")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--out-dir", default=str(ROOT), help="base dir for outputs/ and reports/")
    args = parser.parse_args()

    eval_path = Path(args.eval_file) if args.eval_file else (
        FROZEN_EVAL if FROZEN_EVAL.exists() else DRAFT_EVAL
    )
    records = load_eval_records(eval_path)
    if args.limit:
        records = records[: args.limit]

    is_draft = "draft" in eval_path.name
    if is_draft:
        print(f"[WARNING] running against DRAFT eval set {eval_path.name} — "
              "re-run after the set is frozen.")
    print(f"[baseline] {MODEL_ID} on {len(records)} questions "
          f"(batch={args.batch_size}, max_new_tokens={args.max_new_tokens})")

    generate = build_generate_fn(args.max_new_tokens)
    predictions, metrics = run_eval(generate, records, batch_size=args.batch_size)

    out_base = Path(args.out_dir)
    write_outputs(
        predictions,
        metrics,
        out_base / "outputs" / "baseline" / "predictions.jsonl",
        out_base / "reports" / "baseline.json",
        meta={
            "model": MODEL_ID,
            "eval_file": eval_path.name,
            "eval_is_draft": is_draft,
            "questions": len(records),
            "system_prompt": NYAYA_SYSTEM_PROMPT,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy",
        },
    )
    print(f"[done] strict accuracy {metrics['auto_strict_accuracy']:.1%} "
          f"({metrics['auto_strict_correct']}/{metrics['scored_total']} scored); "
          f"abstention rate {metrics['abstention_rate']:.1%}; "
          f"mean latency {metrics['mean_latency_s']}s")
    print(f"[out] {out_base / 'outputs' / 'baseline' / 'predictions.jsonl'}")
    print(f"[out] {out_base / 'reports' / 'baseline.json'}")


if __name__ == "__main__":
    main()
