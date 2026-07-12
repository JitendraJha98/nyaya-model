"""Step 9 — Train Nyaya-3B-v1 (configs/train_v1.yaml).

Start with 8K–15K validated examples, not 25K automatically.
QLoRA, seq 4096, r=32/alpha=64, lr 1e-4, 1 epoch, effective batch 32–64,
BF16, gradient checkpointing, Flash Attention 2 if supported.

Benchmark every meaningful checkpoint — do not automatically use the final one.

TODO: thin wrapper over src/nyaya/trainer.py with configs/train_v1.yaml.
"""

raise NotImplementedError("Implement via src/nyaya/trainer.py after the smoke run passes.")
