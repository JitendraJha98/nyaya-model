---
license: mit
base_model: intfloat/multilingual-e5-base
base_model_relation: finetune
language:
- en
- hi
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- legal
- india
- indian-law
- bns
- bnss
- bsa
- retrieval
- sentence-transformers
- feature-extraction
- hinglish
datasets:
- NyayaLabs98/nyaya-statute-db
model-index:
- name: nyaya-embed-v1
  results:
  - task:
      type: retrieval
      name: Statute-section retrieval (hybrid BM25 + dense, RRF)
    dataset:
      type: NyayaLabs98/nyaya-eval-v1
      name: Nyaya-Eval-v1, never-audited slice (n=118)
    metrics:
    - type: recall_at_1
      value: 49.2
      name: full-hit recall@1
    - type: recall_at_8
      value: 88.1
      name: full-hit recall@8
---

# Nyaya-Embed-v1 — a bi-encoder for Indian statute retrieval

`intfloat/multilingual-e5-base` (278M parameters, MIT) fine-tuned so that a citizen's
question in English, Hindi or Hinglish lands on the right **section of current Indian
law**. It is the dense half of the hybrid retriever in the
[Nyaya](https://github.com/JitendraJha98/nyaya-model) legal-guidance system; the other
half is BM25 over [`NyayaLabs98/nyaya-statute-db`](https://huggingface.co/datasets/NyayaLabs98/nyaya-statute-db).

> **⚖️ Not legal advice.** This model finds statute sections. What to do about them is
> a question for an advocate enrolled under the Advocates Act, 1961.

## What changed against zero-shot e5-base

Full-hit recall (every gold section of a question inside the top-k), hybrid BM25 + dense
with reciprocal-rank fusion, measured by `scripts/15_retrieval_recall.py` on the Eval-v1
questions that were **never used to tune retrieval** (n=118):

| Retriever | @1 | @3 | @5 | @8 |
|---|---:|---:|---:|---:|
| BM25 alone (synonyms, citation resolution) | 45.8% | 61.0% | 74.6% | 81.4% |
| BM25 + zero-shot e5-base, RRF | worse than BM25 alone at every k (report `retrieval_recall_dense.json`) | | | |
| **BM25 + nyaya-embed-v1, RRF** | **49.2%** | **73.7%** | **78.0%** | **88.1%** |
| BM25 + bge-reranker-v2-m3 (568M cross-encoder, depth 20) | 58.5% | 69.5% | 74.6% | 83.9% |

**Effect on the reader** (`reports/eval_v1_comparison_base-768-embed-v1.json`): same base
Qwen2.5-3B-Instruct, same 413 Eval-v1 questions, 768 tokens, k=8, only the dense model
swapped — fact recall **35.8% → 39.7%**, paired 95% CI **[+0.9, +7.0]** points, better on 64
questions, worse on 45. The first end-to-end gain in the project with an interval clear of
zero. This model is the default dense stage of the retriever from 2026-09-04.

Report: `reports/retrieval_recall_dense_embed_v1.json` in the repository. Per language
(all 150 gold-bearing questions, @8): English 84.2% (n=139), Hindi 80.0% (n=5),
Hinglish 50.0% (n=6). The Hindi and Hinglish counts are too small to rank anything.

## Training

- **Pairs:** 4,412 (question, gold section) pairs built by `scripts/41_build_retriever_pairs.py`
  from the project's training questions (Nyaya-Train-v3 metadata), each with 20 BM25 hard
  negatives. Every Eval-v1 question was excluded before pairing. 300 pairs were held out.
- **Loss:** MultipleNegativesRankingLoss (in-batch negatives), batch 32, 1 epoch,
  learning rate 2e-5, 5% warm-up, fp16, on one Kaggle T4 (132 s).
- **Text format:** `query: <question>` and `passage: <act name> — <title>. <text>`, the e5
  convention. Keep the prefixes at inference.

## Use

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("NyayaLabs98/nyaya-embed-v1")
q = model.encode(["query: police FIR nahi likh rahi, kya karu?"], normalize_embeddings=True)
d = model.encode(["passage: Bharatiya Nagarik Suraksha Sanhita, 2023 — Information in cognizable cases. ..."],
                 normalize_embeddings=True)
print(q @ d.T)
```

In the repository: `python scripts/26_eval_v1_run.py --dense --dense-model NyayaLabs98/nyaya-embed-v1 ...`
or `attach_dense_index(index, model_name="NyayaLabs98/nyaya-embed-v1")`.

## Limits

- Trained on questions about 27 acts plus the Constitution; sections outside the statute
  DB are not represented.
- 4,412 pairs is small. The gain over BM25 is real at k=3 and k=8 on the never-audited
  slice; at k=1 it is within noise of BM25 (n=118).
- Fine-tuned from the `intfloat/multilingual-e5-base` checkpoint (MIT). Released under MIT.
