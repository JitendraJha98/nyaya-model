"""Evaluation harness — build this BEFORE training run 1.

Layers:
  1. Citation Accuracy Score (primary metric) — via validators.extract_citations
     + statute-DB verification.
  2. Held-out-acts generalization.
  3. AIBE bar-exam MCQs.
  4. IL-TUR + BhashaBench-Legal deltas.
  5. Blind human eval (ship gate) + 50-question regression set.

Per-question record: question, response, expected answer, extracted citations,
correct/incorrect, language quality, hallucination, abstention, latency.

TODO: implement run_eval(model, eval_file) -> predictions.jsonl + metrics dict.
"""
