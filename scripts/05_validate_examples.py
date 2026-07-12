"""Step 5 — Validate every generated example (one of the most important steps).

Pipeline (docs/ROADMAP.md, Step 8):
    JSON schema validation -> required fields -> source verification ->
    citation verification (regex-extract sections, resolve against statute DB,
    DROP the whole example on any failure) -> language detection ->
    answer length check (80–600 words) -> near-duplicate detection ->
    train/eval leakage detection -> quality scoring -> ACCEPT / REJECT

Prints a rejection report (generated / schema failures / citation failures /
duplicates / low quality / leakage / accepted). A 15–20% rejection rate is
healthy — it means the pipeline is filtering garbage.

Input:  data/generated/*.jsonl
Output: data/validated/*.jsonl + reports/validation_report.json

TODO: implement using src/nyaya/validators.py.
"""

raise NotImplementedError("Implement with src/nyaya/validators.py once generation (step 4) produces data.")
