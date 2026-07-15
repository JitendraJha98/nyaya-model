"""Baseline the UNTRAINED base model on the generated dataset's test split.

Two complementary "before" measurements for fine-tuning comparisons:

1. Eval loss / perplexity on the assistant turns of data/splits/test.jsonl —
   the same quantity the trainer reports, teacher-forced.
2. Behavioural scores on generated answers to the test questions:
   citation-gate pass rate (statute-DB verified), old-law citation rate,
   language match (Hindi answered in Hindi?), similarity to the reference
   answers — bucketed by language and task type, incl. the held-out acts.

The frozen-eval baseline (scripts/02) stays the primary metric; this is the
in-distribution counterpart on the training-data mix.

Outputs:
    outputs/dataset_baseline/predictions.jsonl
    reports/dataset_baseline.json

Usage:
    python scripts/14_dataset_baseline.py [--limit 50] [--skip-loss]
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.dataset import load_jsonl
from nyaya.evaluation import run_dataset_eval
from nyaya.prompts import NYAYA_SYSTEM_PROMPT
from nyaya.validators import load_statute_db

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return tokenizer, model


def build_generate_fn(tokenizer, model, max_new_tokens: int):
    import torch

    def generate(questions):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": NYAYA_SYSTEM_PROMPT},
                 {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
            )
            for q in questions
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)

    return generate


def eval_loss(tokenizer, model, records, batch_size=4):
    """Mean NLL per assistant token over the test split (teacher-forced)."""
    import torch

    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        input_ids_list, labels_list = [], []
        for r in batch:
            prompt = tokenizer.apply_chat_template(
                r["messages"][:-1], tokenize=False, add_generation_prompt=True)
            answer = r["messages"][-1]["content"] + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + answer_ids)[-2048:]
            labels = ([-100] * len(prompt_ids) + answer_ids)[-2048:]
            input_ids_list.append(ids)
            labels_list.append(labels)
        width = max(len(x) for x in input_ids_list)
        pad = tokenizer.pad_token_id
        input_ids = torch.tensor(
            [[pad] * (width - len(x)) + x for x in input_ids_list], device=model.device)
        labels = torch.tensor(
            [[-100] * (width - len(x)) + x for x in labels_list], device=model.device)
        attention = (input_ids != pad).long()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), ignore_index=-100, reduction="sum")
        total_nll += loss.item()
        total_tokens += int((shift_labels != -100).sum())
    import math
    mean_nll = total_nll / max(1, total_tokens)
    return {"eval_loss": round(mean_nll, 4),
            "perplexity": round(math.exp(mean_nll), 2),
            "assistant_tokens": total_tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", default=str(ROOT / "data" / "splits" / "test.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--skip-loss", action="store_true")
    parser.add_argument("--out-dir", default=str(ROOT))
    args = parser.parse_args()

    records = load_jsonl(args.test_file)
    if args.limit:
        records = records[: args.limit]
    statute_db = load_statute_db(include_old_law=True)
    print(f"[dataset-baseline] {MODEL_ID} on {len(records)} test examples")

    tokenizer, model = load_model()

    loss_metrics = {}
    if not args.skip_loss:
        t0 = time.time()
        loss_metrics = eval_loss(tokenizer, model, records)
        print(f"[loss] {loss_metrics} ({time.time() - t0:.0f}s)")

    generate = build_generate_fn(tokenizer, model, args.max_new_tokens)
    predictions, metrics = run_dataset_eval(
        generate, records, statute_db, batch_size=args.batch_size)
    metrics.update(loss_metrics)

    out_base = Path(args.out_dir)
    pred_path = out_base / "outputs" / "dataset_baseline" / "predictions.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as fh:
        for p in predictions:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    report_path = out_base / "reports" / "dataset_baseline.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "meta": {"model": MODEL_ID, "test_file": Path(args.test_file).name,
                 "examples": len(records), "system_prompt": NYAYA_SYSTEM_PROMPT,
                 "decoding": "greedy", "max_new_tokens": args.max_new_tokens},
        "metrics": metrics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] citation_pass={metrics['citation_pass_rate']:.1%} "
          f"old_law={metrics['old_law_citation_rate']:.1%} "
          f"lang_match={metrics['language_match_rate']:.1%} "
          f"ref_sim={metrics['mean_reference_similarity']:.3f}")
    print(f"[out] {pred_path}")
    print(f"[out] {report_path}")


if __name__ == "__main__":
    main()
