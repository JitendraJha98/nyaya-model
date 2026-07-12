"""Step 11 — Error analysis. This decides what goes into Nyaya-Instruct-v2.

Bucket every failure from evaluation (metadata makes this possible — never throw
metadata away): by task_type, legal_domain, language, difficulty, act/section.

Example outcome: 30% procedural, 25% complicated BNSS, 20% Hinglish,
15% hallucinated sections, 10% other -> v2 adds more procedural questions,
BNSS questions, hard Hinglish examples, citation-abstention examples.

The loop: DATA -> TRAIN -> EVALUATE -> FAILURE ANALYSIS -> BETTER DATA -> TRAIN AGAIN.
Not: more epochs.

Output: reports/error_analysis.json

TODO: implement once step 10 produces predictions.
"""

raise NotImplementedError("Implement after evaluation (step 10) produces failure data.")
