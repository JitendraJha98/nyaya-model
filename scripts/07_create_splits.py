"""Step 7 — Create train/val/test splits. NEVER randomly split rows.

Split by SOURCE SECTION / source document / legal procedure: if 50 examples were
generated from BNS Section 318, all 50 belong to exactly one split.

Targets: 90% train / 5% val / 5% internal test.
Additionally hold out 2 ENTIRE ACTS (e.g., POSH + MV Act) from training to
measure generalization to unseen statutes.

Nyaya-Eval-v0 (data/eval/) is separate and completely untouched.

Input:  data/validated/*_deduped.jsonl
Output: data/splits/{train,val,test}.jsonl

TODO: thin wrapper over src/nyaya/dataset.py::grouped_split (keyed on
metadata.source_sections).
"""

raise NotImplementedError("Implement grouped-by-source splitting after steps 5–6.")
