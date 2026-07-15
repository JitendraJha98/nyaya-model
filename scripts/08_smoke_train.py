"""Step 8 — 1K-example smoke training run (configs/smoke.yaml).

The goal is NOT model quality. Verify: dataset loads, chat template correct,
tokenization correct, loss decreases, no OOM, checkpoint saves, adapter reloads,
inference works, evaluation pipeline works.

After training, manually inspect: ask 50 questions, compare base Qwen vs smoke
model. Look for: legal terminology improved? Hinglish improved? repetitive?
answers too long? general reasoning degraded? citation hallucinations increased?
"consult a lawyer" for everything? memorized training examples?

Only after this passes do we train v1. Method: bf16 LoRA (no quantization).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.trainer import load_config, train

SMOKE_PROMPTS = [
    "What is the punishment for cheating under current Indian law?",
    "Police FIR nahi likh rahi, kya karu?",
]


def verify_adapter_reload(config: dict) -> None:
    """The smoke run's point: prove the checkpoint reloads and generates."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    final_dir = str(Path(config["output_dir"]) / "final")
    print(f"[verify] reloading adapter from {final_dir}")
    tokenizer = AutoTokenizer.from_pretrained(final_dir)
    base = AutoModelForCausalLM.from_pretrained(
        config["model_id"], dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, final_dir)
    model.eval()
    for prompt in SMOKE_PROMPTS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        answer = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
        print(f"\n[verify] Q: {prompt}\n[verify] A: {answer[:400]}")


if __name__ == "__main__":
    config_path = ROOT / "configs" / "smoke.yaml"
    metrics = train(config_path)
    print(f"[done] smoke training metrics: {metrics}")
    verify_adapter_reload(load_config(config_path))
    print("[done] adapter reload + inference verified")
