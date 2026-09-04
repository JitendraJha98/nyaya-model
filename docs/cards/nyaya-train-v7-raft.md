---
license: apache-2.0
language:
- en
- hi
tags:
- legal
- india
- indian-law
- bns
- bnss
- bsa
- instruction-tuning
- rag
- synthetic
- distillation
pretty_name: Nyaya-Train-v7-RAFT — teacher answers under the serving prompt (not used for training)
size_categories:
- 1K<n<10K
task_categories:
- question-answering
- text-generation
---

# Nyaya-Train-v7-RAFT

1,456 chat-format records (1,293 train / 163 val): a citizen question, the statute
sections the Nyaya retriever surfaced for it (BM25 + `nyaya-embed-v1`, k=8), and an
answer written by `Qwen/Qwen2.5-14B-Instruct-AWQ` (Apache-2.0) served with vLLM on
two Kaggle T4s. Every citation in every kept answer resolves against the statute DB
**and** appears in that record's own context; 315 answers that failed this gate are
in `nyaya_instruct_v7_rejected.jsonl` (148 cited on a deliberate retrieval miss, 146
cited outside the context, 21 cited nothing). 0 records overlap the frozen evaluation
sets (`reports/v7_raft_dataset_report.json` in the repository).

## Why it exists, and why it was not trained on

This is task C4 of the project's release plan: distil a stronger teacher into the
small reader. The teacher was first scored on the project's own benchmark under the
exact serving prompt (`scripts/26_eval_v1_run.py --endpoint`): **45.0% fact recall**
against 39.7% for the 3B reader on the same retriever (paired 95% CI [+2.3, +8.4]).
It passed that gate, and the data was generated. The same day, the reader shootout
found that `Qwen/Qwen3-4B-Instruct-2507` reaches **50.6%** with no training at all.
A teacher that scores below the student has nothing to teach it, so no v7 model was
trained. The records are published because they are clean, citation-verified,
Apache-2.0-licensed training data for anyone fine-tuning a *smaller* reader than the
teacher, and because the outcome is part of the project's evidence.

## Composition

| | count |
|---|---:|
| english / hinglish / hindi | 1,056 / 213 / 187 |
| grounded_qa / procedural / hinglish_qa / hindi_qa / safety_abstention / terminology | 533 / 250 / 213 / 187 / 149 / 96 |
| deliberate retrieval-miss demonstrations | 6 |
| mean / median / p90 answer length (words) | 135 / 131 / 187 |

Questions come from [`NyayaLabs98/nyaya-train-v3`](https://huggingface.co/datasets/NyayaLabs98/nyaya-train-v3)
(`metadata.rag.question`), shuffled with seed 7; the first 1,600 train and 200 val
tasks were sent to the teacher (27 timed out at 300 s and are absent).

## Schema

Same as Nyaya-Train-v3: `messages` (system / user with the RAG prompt / assistant),
`metadata` with `language`, `task_type`, `source_act`, `source_sections`,
`rag.context_keys`, `rag.is_miss`, `rag.question`, `generator`, `split`.

## Licence

Apache-2.0 for the generated text (teacher: Qwen2.5-14B-Instruct, Apache-2.0).
Statutory passages are Government of India material, public domain under Section
52(1)(q) of the Copyright Act, 1957. **Not legal advice.** Built with
[github.com/JitendraJha98/nyaya-model](https://github.com/JitendraJha98/nyaya-model).
