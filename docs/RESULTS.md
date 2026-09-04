# Nyaya — everything that was measured

Every number here is traceable to a committed file in `reports/` or
`outputs/eval-v1/`. Where a number was later found to be wrong, both the wrong
number and the correction are kept — the corrections are the most useful part
of this document.

---

## 1. Does fine-tuning beat the base model?

**No.** Asked four ways, answered the same way each time.

### Our own benchmark (Nyaya-Eval-v1, 409 questions, paired bootstrap)

| model | fact recall | vs base | verdict |
|---|---|---|---|
| `Qwen2.5-3B-Instruct` + RAG | **34.3%** | — | baseline |
| v3 (RAFT) | 32.9% | −1.3, CI [−4.4, +1.8] | **tied** |
| v3, 768-token rerun, corrected tokenizer (vs `base-768` 35.8%) | 33.8% | −2.0, CI [−5.2, +1.2] | **tied**; citations worse |
| v5 (grounded citation data) | 24.0% | −10.3, CI [−13.5, −7.2] | **worse** |
| v6 (v5 + answer-style fix) | 23.4% | −10.8, CI [−14.0, −7.8] | **worse** |

### Base-model shootout, session 1 (2026-09-04)

Same retriever (zero-shot e5-base), same 413 questions, 768 new tokens, batch 2
on one Kaggle T4; paired 10,000-round bootstrap against `base-768`
(`reports/eval_v1_comparison_qwen3-4b.json`, `reports/eval_v1_comparison_nyaya-3b-v3-768.json`):

| reader | fact recall | citation | substance | Δ fact recall vs base-768 | better / worse / tied |
|---|---|---|---|---|---|
| `Qwen2.5-3B-Instruct` (`base-768`) | 35.8% | 55.6% | 31.6% | — | — |
| `nyaya-3b-v3` (768 tokens) | 33.8% | 48.6% | 30.6% | −2.0, CI [−5.2, +1.2] **tied** | 56 / 70 / 283 |
| **`Qwen3-4B-Instruct-2507`** | **50.6%** | **72.2%** | **46.8%** | **+14.8, CI [+11.4, +18.3]** | **119 / 22 / 268** |

Three things this settles:

- **v3 is tied with base a second time**, now with the tokenizer and config
  files corrected (the Hub copy was written by transformers 5.12; see §3) and
  under exactly the same retriever. Citation recall is *worse* (−6.9,
  CI [−13.9, −0.4]). The 32.9% figure was not a loading artefact.
- **Reader scale moves the score more than anything else tried.** One
  generation of base model and one billion parameters are worth +14.8 points,
  against +3.9 for the fine-tuned embedder and nothing for five fine-tunes.
- **Length is not the whole story.** Qwen3-4B writes 306 words per answer
  against 185 for the 3B base, and the substance scorer rewards coverage; but
  citation accuracy — strict, length-blind — rises by the same 16.7 points, and
  the paired per-question count is 119 better to 22 worse. Cost: 31.5 s per
  question on a T4 against 12.3 s, and 1 extra GB of weights.

`Qwen3-4B-Instruct-2507` is Apache-2.0, unlike the 3B base (qwen-research). It
is the default reader from this commit. The combined configuration (Qwen3-4B
reader, `nyaya-embed-v1` retriever) is being measured against both parents; its
paired interval will become the system's headline number.

### An external benchmark (BhashaBench-Legal, 1,500 MCQs, exact scoring)

| model | overall | English | Hindi |
|---|---|---|---|
| base | **47.8%** | 54.9% | 40.7% |
| v3 | 45.2% | 51.6% | 38.8% |

delta −2.6%, CI [−6.2, +1.0] → **tied**. Random guessing is 25%, so both
models genuinely know Indian law; they just know it equally well.

### Why v5 and v6 regressed

Answer length, and it is monotone:

| | mean answer | fact recall |
|---|---|---|
| base | 173 words | 34.3% |
| v5 | 90 words | 24.0% |
| v6 | 57 words | 23.4% |

Mean answer length is over all 413 predictions; fact recall is over the 409
scored rows, exactly as in the table above.

Shorter answers carry fewer of the facts being scored. Training on templated
27-word targets taught 57-word answers. Fixing v5's verbatim-recitation problem
made answers shorter still, which is why the fix made things worse rather than
better: recitation was a symptom, brevity was the cause.

**Conclusion.** Rule-based synthetic targets have a fixed shape and the model
learns the shape rather than the task. Beating base needs targets with base's
coverage *and* better accuracy, i.e. a strong teacher model — not another
template. Fine-tuning at 3B with rule-based data is a closed path.

---

## 2. What actually improved the system: retrieval

Accuracy is dominated by whether the right statute reaches the context:

| retrieval outcome (base, Eval-v1) | n | fact recall |
|---|---|---|
| every gold statute retrieved | 94 | **63.2%** |
| at least one gold statute missed | 51 | **20.3%** |
| no section in the gold facts | 264 | 26.7% |

A 43-point gap. The model was never the bottleneck. Recomputed from the
committed predictions by `scripts/36_retrieval_outcome.py`
(`reports/eval_v1_retrieval_outcome.json`); an earlier ad-hoc figure of 17.1%
over 43 records could not be reproduced and is withdrawn.

**Retriever-version caveat.** The base and v3 predictions were generated on
2026-08-06, before the second vocabulary pass (commit 19189df); v5 and v6 on
2026-08-07, after it. Under the later retriever 109 records had every gold
section in context, against 94 for base/v3, so v5 and v6 saw *better*
retrieval and still scored worse. The paired CIs stand, but "same retriever"
is only exactly true for base vs v3. The 768-token reruns planned for the
base-model shootout put every run under one retriever.

### Fine-tuned bi-encoder (2026-09-04)

`intfloat/multilingual-e5-base` fine-tuned with in-batch negatives on 4,412
(question, gold section) pairs from the project's own training questions, every
Eval-v1 question excluded (`scripts/41_build_retriever_pairs.py`,
`scripts/kaggle_train_retriever.ipynb`, 132 s on one T4). Published as
`NyayaLabs98/nyaya-embed-v1`. Full-hit recall on the 118 never-tuned records,
hybrid BM25 + dense with reciprocal-rank fusion
(`reports/retrieval_recall_dense_embed_v1.json`):

| | k=1 | k=3 | k=5 | k=8 |
|---|---|---|---|---|
| BM25 | 45.8% | 61.0% | 74.6% | 81.4% |
| BM25 + zero-shot e5-base | below BM25 at every k (`retrieval_recall_dense.json`) | | | |
| **BM25 + nyaya-embed-v1** | **49.2%** | **73.7%** | **78.0%** | **88.1%** |
| BM25 + bge-reranker-v2-m3 | 58.5% | 69.5% | 74.6% | 83.9% |
| BM25 + nyaya-reranker-mini-v1 (118M) | 51.7% | 70.3% | 76.3% | 82.2% |

**Effect on the reader.** Same base Qwen2.5-3B, same 413 questions, 768 new
tokens, k=8; only the dense model changed
(`reports/eval_v1_comparison_base-768-embed-v1.json`, 10,000-round paired bootstrap):

| dense stage | fact recall | citation | substance | paired Δ fact recall |
|---|---|---|---|---|
| zero-shot e5-base (`base-768`) | 35.8% | 55.6% | 31.6% | — |
| **nyaya-embed-v1** | **39.7%** | 61.1% | 34.6% | **+3.9, 95% CI [+0.9, +7.0]** |

Better on 64 questions, worse on 45, tied on 300. Citation and substance recall
move the same way but their intervals still touch zero. This is the first
end-to-end improvement in the project whose confidence interval excludes zero,
and it comes from retrieval. `nyaya-embed-v1` is the default dense model from
this date; every earlier run used zero-shot e5-base (their `dense_model` field
is absent). The 384- vs 768-token comparison of the base model itself is a tie
(+1.5, CI [−0.2, +3.3], `reports/eval_v1_comparison_base-768.json`), so the
token budget is not what moved.

The mini reranker costs 3.2 s per query on CPU at depth 20 (Kaggle CPU), so it
stays out of the browser demo; the embedder needs a GPU or a precomputed vector
file for the 5,063 rows.

### Cross-encoder reranking

Scored **only on the 118 records never used to tune anything**:

| | k=1 | k=3 | k=8 |
|---|---|---|---|
| BM25 | 45.8% | 61.0% | 81.4% |
| + reranker | **58.5%** | **69.5%** | **83.9%** |

**+12.7 points at k=1.** The right statute now arrives at the top far more
often, which matters more than recall@8: at k=8, seven of eight passages are
noise the model must sift.

Cost, measured: ~1.6 s/pair on CPU (≈80 s/query at depth 50), so this stage is
GPU-only in practice.

---

## 3. Measurement bugs found in our own evaluation

Each made a number look like progress while measuring something else. This is
the section to read first if you are continuing the project.

**1. Eval-v0 scored its own gold answers at 10.7%.**
The strict metric required every required fact, and ~85% of facts were
free-text phrases matched as substrings. *"imprisonment for life or death"*
failed *"death or imprisonment for life"*. No model could exceed ~10.7%, which
is why base, v3 and v4 sat within a 2-answer spread — the ruler, not the
models. Eval-v1 scores gold at **100%** by construction.

**2. Subsection citations were unmatchable.**
`\b` after `)` never matches, so every `Section 103(2)` fact scored 0 no matter
what the model said. Fixing it lifted citation accuracy on gold answers
82.9% → 96.6%.

**3. Correct old→new bridging was penalised as stale law.**
*"Section 173 BNSS, which replaced Section 154 CrPC"* was scored as citing
repealed law, because the forbidden-fact check matched the section number and
ignored the framing. Every correct IPC→BNS answer was being zeroed.

**4. Retrieval phrase coverage reported 5.1%; it was 64%.**
Same verbatim-matching defect, 12× understated. It had the project believing
retrieval surfaced almost nothing for 70% of questions.

**5. A retrieval "improvement" that did not generalise.**
Hand-written vocabulary synonyms appeared to add **+16 points**. Split by
whether the record's failure had been inspected while writing them:

| group | n | before | after |
|---|---|---|---|
| audited | 32 | 0.0% | 71.9% |
| **never audited** | **118** | **80.5%** | **81.4%** |

The gain was almost entirely memorisation of inspected failures.
`scripts/28_validate_generalization.py` now prints an explicit
`DID NOT GENERALISE` verdict, and `scripts/15` reports the never-audited slice
on every run so the honest number cannot be lost.

---

## 4. Portability defects (80 GB GPUs → free Kaggle T4)

The pipeline had only ever run on 80 GB datacentre GPUs. Every one of these was invisible
there and fatal elsewhere — they are what "self-hostable" actually costs.

| defect | symptom |
|---|---|
| hardcoded `bf16` | Turing has no bf16 cores; `is_bf16_supported()` returns True *with emulation*. Silently ~5× slower — cost 4.5h before it was spotted. |
| `device_map="auto"` | accelerate wraps `forward` in a `functools.partial`; TRL's chunked-CE patch needs `__func__`. Dies before step 1. |
| `max_seq_length: 6144` | longest real example is 3,850 tokens; the rest was activation memory spent on padding |
| dual-GPU default | Kaggle's T4×2 → DataParallel puts inputs on `cuda:1`, model on `cuda:0` |
| aggregate-only eval | v1–v4 saved no predictions, so when the scorer was found broken there was nothing to re-score. Every result died with the environment that produced them. |

`scripts/26` now always writes `predictions.jsonl`, and `--rescore` re-grades a
saved run on CPU with no model loaded.

---

## 5. Known weaknesses

- **Hindi is 14 points behind English** (40.7% vs 54.9% on BhashaBench) on the
  model side. On the retrieval side, 19 of the 53 Devanagari questions in the
  269 real citizen questions retrieved **zero** statute sections, because the
  index is English statute text. **Query rewriting fixes most of that:** having
  the reader model rewrite a Hindi/Hinglish question into one line of statutory
  English before retrieval (`nyaya.rewrite`, `--rewrite`) cuts zero-hit
  Devanagari questions from 19 to 3 of 53 and Hinglish from 1 to 0 of 214, at
  5.2 s per rewrite on a CPU with the Q4 GGUF (`reports/rewrite_measurement.json`,
  `scripts/33_measure_rewrite.py`, rewriter = nyaya-3b-v3 Q4_K_M). Whether the
  sections it now finds are the *right* ones is not yet measured: the 9
  gold-bearing Hindi/Hinglish Eval-v1 questions were already full hits before
  and after. That needs the Hindi holdout.
- **Hindi statute text is not obtainable from India Code.** The official Hindi
  PDFs are image scans: the site's own text extraction of the Hindi BNS PDF is
  64 KB with 0% Devanagari characters (checked 2026-09-04 against the new
  indiacode.gov.in API). Indexing Hindi statute text is closed as a path;
  Hindi retrieval relies on query rewriting.
- **Retrieval still misses ~19%** of gold sections at k=8.
- **Coverage is 27 acts plus the Constitution** (fourteen added from the India Code API in
  Sept 2026; 3% of real citizen questions still fall in domains with no act). Absence is flagged by
  the coverage gate but a retriever still returns the nearest
  thing it has regardless.
- **No case law**, statutory text only.
- **No human evaluation** has been run. The project's own ship gate is unmet.
- **`nyaya-eval-v0` was published**, so it is contaminated as a held-out
  benchmark from that point on.

---

## 6. What is defensible to claim

✅ An open Indian legal guidance system: BNS/BNSS/BSA-native (post-July-2024),
English/Hindi/Hinglish, citation-verified, self-hostable, free.
✅ 47.8% on BhashaBench-Legal against a 25% chance floor.
✅ Cross-encoder reranking improves retrieval by +12.7 points at k=1, validated
on held-out records.
✅ Swapping the reader to `Qwen3-4B-Instruct-2507` (Apache-2.0) lifts fact recall
from 35.8% to 50.6% under the same retriever (paired 95% CI [+11.4, +18.3]).
✅ A bi-encoder fine-tuned on the project's own pairs lifts the base reader's
fact recall from 35.8% to 39.7% (paired 95% CI [+0.9, +7.0]) with no change
to the weights — the only end-to-end gain with an interval clear of zero.

❌ "Best" anything — one baseline and one external benchmark is not a ranking.
❌ That the fine-tuned weights beat the base model. They do not.
❌ Apache-2.0 on the weights — the base is `qwen-research`, non-commercial.
