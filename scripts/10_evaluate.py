"""Step 10 — Evaluate checkpoints on Nyaya-Eval-v0 (+ external benchmarks).

Layers (docs/ROADMAP.md, "Evaluation plan" — build the harness BEFORE training run 1):
  1. Citation Accuracy Score (primary): % answers with all citations correct,
     % with any hallucinated citation — verified against the statute DB.
  2. Held-out-acts generalization — the honest overfitting detector.
  3. AIBE bar-exam MCQs — single headline number.
  4. IL-TUR + BhashaBench-Legal — standardized benchmarks, before/after deltas.
  5. Blind human eval (v1 ship gate): 100 real questions, 2 raters;
     ship criteria: win/tie >=60% on Hinglish citizen queries, zero dangerous answers.

Plus a 50-question regression set every checkpoint must pass (unit tests).

Do NOT rely on train/val loss — low loss measures fit to a possibly-hallucinated
distribution, not legal correctness.

Output: reports/ per-checkpoint metrics.

TODO: implement via src/nyaya/evaluation.py.
"""

raise NotImplementedError("Implement via src/nyaya/evaluation.py — harness needed before training run 1.")
