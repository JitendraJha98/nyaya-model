---
license: apache-2.0
base_model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
base_model_relation: finetune
language:
- en
- hi
library_name: sentence-transformers
pipeline_tag: text-ranking
tags:
- legal
- india
- indian-law
- bns
- bnss
- bsa
- reranker
- cross-encoder
- retrieval
- hinglish
datasets:
- NyayaLabs98/nyaya-statute-db
model-index:
- name: nyaya-reranker-mini-v1
  results:
  - task:
      type: text-ranking
      name: Statute-section reranking (BM25 top-20)
    dataset:
      type: NyayaLabs98/nyaya-eval-v0
      name: Nyaya-Eval-v1 (graded successor of nyaya-eval-v0, in the repository), never-audited slice (n=118)
    metrics:
    - type: recall_at_1
      value: 51.7
      name: full-hit recall@1
    - type: recall_at_8
      value: 82.2
      name: full-hit recall@8
---

# Nyaya-Reranker-Mini-v1 — a 118M cross-encoder for Indian statute sections

`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0) fine-tuned to score a citizen's
question against a candidate **section of current Indian law**. It reorders the BM25
top-20 from [`NyayaLabs98/nyaya-statute-db`](https://huggingface.co/datasets/NyayaLabs98/nyaya-statute-db)
in the [Nyaya](https://github.com/JitendraJha98/nyaya-model) retriever. One fifth the size
of the `BAAI/bge-reranker-v2-m3` the project also supports, and weaker than it.

> **⚖️ Not legal advice.** This model ranks statute sections. What to do about them is a
> question for an advocate enrolled under the Advocates Act, 1961.

## Numbers

Full-hit recall (every gold section of a question inside the top-k) after reranking the
BM25 top-20, on the Eval-v1 questions never used to tune retrieval (n=118),
`scripts/15_retrieval_recall.py --rerank`:

| Ranker over BM25 top-20 | @1 | @3 | @5 | @8 | Size | CPU latency, depth 20 |
|---|---:|---:|---:|---:|---:|---:|
| none (BM25 order) | 45.8% | 61.0% | 74.6% | 81.4% | – | – |
| **nyaya-reranker-mini-v1** | **51.7%** | **70.3%** | **76.3%** | **82.2%** | 118M | 3.2 s (Kaggle CPU) |
| bge-reranker-v2-m3 | 58.5% | 69.5% | 74.6% | 83.9% | 568M | slower |

Report: `reports/retrieval_recall_rerank_mini.json`. The 3.2 s CPU latency is why the
browser demo still runs BM25 only.

## Training

- **Examples:** 21,668 (question, section text, label) triples from 4,412 training
  questions: every gold section as a positive and four of the twenty BM25 hard negatives
  as negatives (`scripts/41_build_retriever_pairs.py`; all Eval-v1 questions excluded).
- **Recipe:** `CrossEncoder.fit`, binary relevance, batch 32, 1 epoch, 200 warm-up
  steps, mixed precision, one Kaggle T4 (272 s). Final training loss 0.30.
- **Passage text:** `nyaya.rerank.passage_text` (act name, section title, first 1,600
  characters of the section).

## Use

```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("NyayaLabs98/nyaya-reranker-mini-v1", max_length=512)
scores = model.predict([("police FIR nahi likh rahi, kya karu?", "Bharatiya Nagarik Suraksha Sanhita, 2023 — Section 173 ...")])
```

In the repository: `python scripts/15_retrieval_recall.py --rerank NyayaLabs98/nyaya-reranker-mini-v1 --rerank-depth 20`.

## Limits

- Bounded by its candidates: nothing outside the BM25 top-20 can be recovered.
- Below `bge-reranker-v2-m3` at k=1 by 6.8 points on n=118; use bge when you can afford it.
- Trained on 27 acts plus the Constitution only.
