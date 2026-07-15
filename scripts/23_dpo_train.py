"""Step 18 (v3) — DPO on citation-correctness preferences, on top of v3 SFT.

Policy: base model with the v3 SFT adapter MERGED, plus a fresh LoRA that DPO
trains (TRL uses the adapter-disabled model as the implicit reference — no
second model in memory). Pairs: gate-passing vs gate-failing teacher answers
for the same RAG prompt (scripts/22) — the preference IS the citation gate.

bf16 LoRA, no quantization (project decision 2026-07-13).

Usage:
    python scripts/23_dpo_train.py --sft-adapter outputs/legal-3b-v3/checkpoint-200
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def _filter_to_signature(cls, kwargs: dict) -> dict:
    accepted = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=str(ROOT / "data" / "dpo" / "pairs.jsonl"))
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--output-dir", default="outputs/legal-3b-v3-dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    data = Dataset.from_list([{
        "prompt": tokenizer.apply_chat_template(
            r["prompt"], tokenize=False, add_generation_prompt=True),
        "chosen": r["chosen"],
        "rejected": r["rejected"],
    } for r in rows])
    print(f"[dpo] {len(data)} preference pairs from {args.pairs}")

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, args.sft_adapter)
    model = model.merge_and_unload()  # v3 SFT becomes the backbone + reference

    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    dpo_kwargs = dict(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=6144,
        max_prompt_length=5600,
        report_to=[],
        run_name="legal-3b-v3-dpo",
    )
    config = DPOConfig(**_filter_to_signature(DPOConfig, dpo_kwargs))

    trainer_kwargs = dict(model=model, args=config, train_dataset=data,
                          peft_config=peft_config, processing_class=tokenizer,
                          tokenizer=tokenizer)
    trainer = DPOTrainer(**_filter_to_signature(DPOTrainer, trainer_kwargs))
    result = trainer.train()
    trainer.save_model(f"{args.output_dir}/final")

    (ROOT / "reports" / "dpo_train_report.json").write_text(json.dumps({
        "pairs": len(data), "sft_adapter": args.sft_adapter,
        "beta": args.beta, "metrics": result.metrics}, indent=2), encoding="utf-8")
    print(f"[done] DPO metrics: {result.metrics}")


if __name__ == "__main__":
    main()
