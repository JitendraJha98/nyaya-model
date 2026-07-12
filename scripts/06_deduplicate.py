"""Step 6 — Near-duplicate removal across the validated dataset.

MinHash or embedding cosine similarity > 0.92 = duplicate.
Repetitive data is the classic cause of "loss looks great, answers are bad".

Input:  data/validated/*.jsonl
Output: data/validated/*_deduped.jsonl + counts in reports/validation_report.json

TODO: implement (datasketch MinHash or sentence-transformers embeddings).
"""

raise NotImplementedError("Implement after step 5 validation exists.")
