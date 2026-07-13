"""Step 9 — Train Nyaya-3B-v1 (configs/train_v1.yaml).

Start with 8K–15K validated examples, not 25K automatically.
LoRA (bf16, no quantization), seq 4096, r=32/alpha=64, lr 1e-4, 1 epoch,
effective batch 32–64, gradient checkpointing, Flash Attention 2 if supported.

Benchmark every meaningful checkpoint — do not automatically use the final one.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.trainer import train

if __name__ == "__main__":
    metrics = train(ROOT / "configs" / "train_v1.yaml")
    print(f"[done] v1 training metrics: {metrics}")
