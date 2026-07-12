"""Step 8 — 1K-example smoke training run (configs/smoke.yaml).

The goal is NOT model quality. Verify: dataset loads, chat template correct,
tokenization correct, loss decreases, no OOM, checkpoint saves, adapter reloads,
inference works, evaluation pipeline works.

After training, manually inspect: ask 50 questions, compare base Qwen vs smoke
model. Look for: legal terminology improved? Hinglish improved? repetitive?
answers too long? general reasoning degraded? citation hallucinations increased?
"consult a lawyer" for everything? memorized training examples?

Only after this passes do we train v1.

TODO: thin wrapper over src/nyaya/trainer.py with configs/smoke.yaml.
"""

raise NotImplementedError("Implement via src/nyaya/trainer.py once splits exist.")
