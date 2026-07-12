"""Step 4 — Generate training examples (Nyaya-Instruct-v1) — SYNTHETIC DATA, LATER.

Grounded generation only: every example is generated FROM verbatim statute text
passed into the prompt (src/nyaya/prompts.py::GROUNDED_QA_PROMPT). Never let the
teacher model cite from memory — that bakes hallucinations into training data.

Build in stages: 1,000 -> smoke train -> 5,000 -> 10,000 -> 25,000+.
First 10K composition target: see README.md "Synthetic data".

Every record uses the universal training schema (src/nyaya/schemas.py::TrainingRecord):
{"id", "messages": [...], "metadata": {language, legal_domain, task_type,
 source_act, source_sections, generator, verified, dataset_version}}

Outputs -> data/generated/nyaya_instruct_v1_raw.jsonl

TODO: implement after statute DB + procedure KB exist and Nyaya-Eval-v0 is frozen.
"""

raise NotImplementedError("Synthetic generation comes after the corpus (step 3) and eval set exist.")
