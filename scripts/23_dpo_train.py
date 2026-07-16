"""Step 18 (v3) — DPO on citation-correctness preferences, on top of v3 SFT.

Self-contained loop over nyaya/dpo.py (no TRL — the image's transformers-5/
torch-2.4 combination fits no TRL release). Policy: base with the v3 SFT
adapter MERGED plus a fresh LoRA that DPO trains; the reference log-probs
come from the same model with the adapter disabled — one model in memory.

Pairs: gate-passing vs gate-failing teacher answers for the same RAG prompt
(scripts/22) — the preference IS the deterministic citation gate.

bf16 LoRA, no quantization (project decision 2026-07-13).

Usage:
    python scripts/23_dpo_train.py --sft-adapter outputs/legal-3b-v3/checkpoint-300
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_LEN = 6144


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=str(ROOT / "data" / "dpo" / "pairs.jsonl"))
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--output-dir", default="outputs/legal-3b-v3-dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nyaya.dpo import dpo_loss, sequence_logprob

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    print(f"[dpo] {len(pairs)} preference pairs from {args.pairs}")

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, args.sft_adapter)
    model = model.merge_and_unload()  # v3 SFT becomes backbone + reference
    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    def encode(prompt_msgs, completion):
        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        c_ids = tokenizer(completion + tokenizer.eos_token,
                          add_special_tokens=False).input_ids
        ids = (p_ids + c_ids)[:MAX_LEN]
        return torch.tensor([ids]), min(len(p_ids), MAX_LEN - 1)

    def logprob(ids, prompt_len, use_adapter):
        ids = ids.to(model.device)
        if use_adapter:
            out = model(input_ids=ids).logits
            return sequence_logprob(out, ids, prompt_len)
        with torch.no_grad(), model.disable_adapter():
            out = model(input_ids=ids).logits
            return sequence_logprob(out, ids, prompt_len)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    rng = random.Random(7)
    step, history = 0, []
    started = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        for i, idx in enumerate(order):
            pair = pairs[idx]
            enc_c, pl_c = encode(pair["prompt"], pair["chosen"])
            enc_r, pl_r = encode(pair["prompt"], pair["rejected"])
            ref_c = logprob(enc_c, pl_c, use_adapter=False)
            ref_r = logprob(enc_r, pl_r, use_adapter=False)
            pol_c = logprob(enc_c, pl_c, use_adapter=True)
            pol_r = logprob(enc_r, pl_r, use_adapter=True)
            loss, margin = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=args.beta)
            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                history.append({"step": step, "epoch": epoch,
                                "loss": round(loss.item(), 4),
                                "margin": round(margin.item(), 4)})
                if step % 5 == 0:
                    print(f"  step {step} epoch {epoch} loss {loss.item():.4f} "
                          f"margin {margin.item():.4f}", flush=True)

    # The DPO LoRA sits on top of MERGED v3 — loading it onto the plain base
    # would be wrong. Save the fully-merged model so eval loads it directly
    # (scripts/16 --model <dir> --adapter none).
    out_dir = Path(args.output_dir) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (ROOT / "reports" / "dpo_train_report.json").write_text(json.dumps({
        "pairs": len(pairs), "sft_adapter": args.sft_adapter, "beta": args.beta,
        "epochs": args.epochs, "optimizer_steps": step,
        "final_loss": history[-1]["loss"] if history else None,
        "final_margin": history[-1]["margin"] if history else None,
        "history": history, "seconds": round(time.time() - started),
    }, indent=2), encoding="utf-8")
    print(f"[done] DPO {step} steps in {round(time.time()-started)}s; "
          f"adapter -> {out_dir}")


if __name__ == "__main__":
    main()
