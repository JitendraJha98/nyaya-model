"""Step 2 — Benchmark the untouched base model on Nyaya-Eval-v0.

Runs Qwen2.5-3B-Instruct against all 500 eval questions and records, per question:
question, model response, expected answer, extracted legal citations,
correct/incorrect, language quality, hallucination, abstention behavior, latency.

Outputs:
    outputs/baseline/predictions.jsonl
    reports/baseline.json

This is the "Qwen2.5-3B-Instruct Baseline". Without it we cannot prove that
training improved anything. Requires data/eval/nyaya_eval_v0.jsonl to exist first.

TODO: implement using src/nyaya/evaluation.py once Nyaya-Eval-v0 is curated.
"""

raise NotImplementedError("Build Nyaya-Eval-v0 (data/eval/nyaya_eval_v0.jsonl) first — see docs/ROADMAP.md, Step 4.")
