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

from nyaya.trainer import train

if __name__ == "__main__":
    metrics = train(ROOT / "configs" / "smoke.yaml")
    print(f"[done] smoke training metrics: {metrics}")
