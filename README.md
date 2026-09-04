# Nyaya — an open Indian legal guidance system

Ask a legal question in **English, Hindi or Hinglish** and get a plain-language
answer **cited to a section of current Indian law** — BNS / BNSS / BSA as in
force after 1 July 2024, with IPC↔BNS bridging.

> **⚖️ Not legal advice.** Nyaya provides legal *information*. The practice of
> law in India is reserved to advocates enrolled under the Advocates Act, 1961.
> Consult a licensed advocate for anything consequential. Free legal aid is
> available through NALSA / DLSA.

---

## It is a system, not a model

That distinction is the point, and it was learned the hard way — see
[Results](#results-what-actually-worked).

| Part | What it does | State |
|---|---|---|
| **Statute DB** | 13 acts + the Constitution (2,528 sections), 1,257 official IPC↔BNS / CrPC↔BNSS / IEA↔BSA mappings, 70 guidance notes | ✅ |
| **Retriever** | Exact-citation lookup, BM25, optional dense fusion | ✅ |
| **Reranker** | Cross-encoder picks which retrieved sections actually answer the question | ✅ **+12.7 pts @k=1** |
| **Model** | Reads the retrieved sections, writes a cited answer | Qwen2.5-3B-Instruct |
| **Scorer** | Grades citations strictly, substance with partial credit | ✅ |

Anyone can download a 3B model for free. What is hard to obtain is **current
Indian law as clean, section-level data, and a retriever that finds the right
section.** That is what this repository is.

---

## Results: what actually worked

Measured on Nyaya-Eval-v1 (413 gradeable questions, 409 scored; 4 safety rows are
graded separately), paired, 10k-round bootstrap.

### Retrieval — the win

Cross-encoder reranking, scored **only on the 118 records never used to tune
anything** (see `scripts/28_validate_generalization.py`):

| | k=1 | k=3 | k=8 |
|---|---|---|---|
| BM25 | 45.8% | 61.0% | 81.4% |
| **+ reranker** | **58.5%** | **69.5%** | **83.9%** |

The right statute now reaches the **top result** far more often. That matters
because the model scores **63.2%** when the gold section is in context and
**17.1%** when it is not.

### Fine-tuning — five attempts, none beat the base model

| | fact recall | vs base |
|---|---|---|
| **base Qwen2.5-3B + RAG** | **34.3%** | — |
| v3 (RAFT) | 32.9% | tied (CI spans 0) |
| v5 (grounded citation data) | 24.0% | **worse**, CI [−13.5, −7.2] |
| v6 (v5 + answer-style fix) | 23.4% | **worse**, CI [−14.0, −7.8] |

The cause is answer length, and it is monotone: base 173 words → v5 90 → v6 57.
Shorter answers carry fewer of the facts being scored. Rule-based synthetic
targets have a fixed shape, and the model learns the shape rather than the task.

**Conclusion: base + retrieval is the product.** Beating base would need a
strong teacher model, not another template.

### Four measurement bugs found in our own evaluation

Each of these made a number look like progress while measuring something else:

1. **Eval-v0 scored its own gold answers at 10.7%** — no model could exceed
   that, so base/v3/v4 were pinned within a 2-answer spread by the ruler.
2. **Subsection citations were unmatchable.** `\b` after `)` never matches, so
   every `Section 103(2)` fact scored 0 regardless of the answer.
3. **Correct old→new bridging was penalised.** "Section 173 BNSS, which
   replaced Section 154 CrPC" was scored as citing stale law.
4. **Retrieval phrase coverage reported 5.1%, actually 64%** — the metric
   demanded verbatim text where statutes say "may extend to three years" and
   the eval says "up to 3 years".

A vocabulary fix that looked like +16 pts turned out to be **+0.9** on records
it had not been tuned against. `scripts/28` now prints an explicit
`DID NOT GENERALISE` verdict so that cannot recur silently.

---

## Quickstart

```bash
git clone https://github.com/JitendraJha98/nyaya-model.git
cd nyaya-model
pip install -r requirements.txt

python scripts/03_build_corpus.py          # build the statute DB
python scripts/15_retrieval_recall.py --k 1 3 5 8       # measure retrieval
python scripts/15_retrieval_recall.py --rerank --rerank-depth 20   # with reranking (GPU)
```

Reranking is GPU-practical only: the multilingual cross-encoder runs at
~1.6 s/pair on CPU, i.e. ~80 s per query at depth 50.

---

## Repository map

```
src/nyaya/
  retrieval.py    statute index: exact citation lookup, BM25, RRF fusion
  rerank.py       cross-encoder second stage (nyaya.rerank)
  dense.py        optional dense embedding stage
  scoring.py      Eval-v1 scorer: strict citations, partial-credit substance
  evaluation.py   Eval-v0 harness (retained for continuity)
  trainer.py      LoRA SFT; precision follows the hardware
scripts/
  15  retrieval recall, with --rerank and a hard --max-minutes budget
  25  build Eval-v1 from the frozen v0 set
  26  run a model on Eval-v1; always saves raw predictions
  27  paired bootstrap comparison between two runs
  28  does a retrieval change generalise, or did it fit the eval?
  29  RAG-grounded training data generator
docs/
  RELEASE_PLAN.md   what ships, and what must never be claimed
```

Every eval run writes `predictions.jsonl`. Scoring can then be revised on CPU
forever — v1–v4 kept only aggregates and their results died with the GPU environment
that produced them.

---

## Licensing — read before using the weights

- **Code**: Apache-2.0.
- **Weights**: any released Nyaya adapter is a derivative of
  `Qwen/Qwen2.5-3B-Instruct`, which is **`qwen-research` — non-commercial**,
  *not* Apache-2.0. The 3B is one of the two Qwen2.5 sizes with a restricted
  licence.
- **Statutory text**: Government of India material, public domain under
  Section 52(1)(q) of the Copyright Act, 1957.
- Some aggregated research datasets referenced here are CC-BY-NC. See `NOTICE`.

## Claims this project does not make

One external benchmark has been run: BhashaBench-Legal (1,500-question sample,
exact MCQ scoring) — base 47.8%, v3 45.2%, tied (`reports/bhashabench_scores.json`).
No human evaluation has been passed. Nyaya is **not** claimed to be the best
Indian legal model, and no fine-tune here has beaten its own base model.
See `docs/RELEASE_PLAN.md`.
