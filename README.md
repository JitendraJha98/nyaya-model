# Nyaya — an open Indian legal guidance system

![tests](https://github.com/JitendraJha98/nyaya-model/actions/workflows/tests.yml/badge.svg)
![code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue)
![default stack: Apache-2.0 / MIT](https://img.shields.io/badge/default%20stack-Apache--2.0%20%2F%20MIT-blue)
![nyaya-3b-v3: qwen-research](https://img.shields.io/badge/nyaya--3b--v3-qwen--research%20(non--commercial)-orange)
![statutes: public domain](https://img.shields.io/badge/statutes-public%20domain-green)

Ask a legal question in **English, Hindi or Hinglish** and get a plain-language
answer **cited to a section of current Indian law** — the Bharatiya Nyaya
Sanhita, Bharatiya Nagarik Suraksha Sanhita and Bharatiya Sakshya Adhiniyam as
in force since 1 July 2024, with the official IPC↔BNS / CrPC↔BNSS bridging.

> **⚖️ Not legal advice.** Nyaya provides legal *information*. The practice of
> law in India is reserved to advocates enrolled under the Advocates Act, 1961.
> Consult a licensed advocate for anything consequential. Free legal aid is
> available through NALSA / DLSA.

> **Measured, 2026-09-04.** The default configuration — `Qwen/Qwen3-4B-Instruct-2507`
> reading sections found by BM25 + `nyaya-embed-v1` — scores **52.0% fact recall and
> 77.1% citation accuracy** on the project's 413-question benchmark, against 35.8% and
> 55.6% for the system as first published (paired 95% CI on fact recall
> **[+12.7, +19.9]** points, better on 135 questions, worse on 24). Every prediction
> behind every number is committed under `outputs/eval-v1/` and re-scorable on a CPU.

---

## Try it

**In the browser, nothing to install:**
[huggingface.co/spaces/NyayaLabs98/nyaya-demo](https://huggingface.co/spaces/NyayaLabs98/nyaya-demo)
— the retriever ported to JavaScript, running client-side over the statute
database, verified to return the same sections as the Python code on 389 real
questions (`scripts/39_build_static_demo.py --check`).

**From the command line:**

```bash
pip install "git+https://github.com/JitendraJha98/nyaya-model"
nyaya ask "police FIR nahi likh rahi, kya karu?"
```

That is the retriever on its own: standard-library Python, no GPU, the statute
database downloads once from the Hub (about 5 MB). It prints the sections of
current law the question resolves to — the same sections the reader model is
given.

The full demo, with the reader model writing the answer beside the retrieved
sections:

```bash
git clone https://github.com/JitendraJha98/nyaya-model.git && cd nyaya-model
pip install -e ".[demo]"
python app.py --no-model      # retrieval only, CPU, instant
pip install -e ".[train]"
python app.py                 # with the reader model (GPU recommended)
```

The demo runs `Qwen/Qwen3-4B-Instruct-2507` as the reader, not the project's
own fine-tune: `NyayaLabs98/nyaya-3b-v3` is statistically tied with the 3B base
it came from, and Qwen3-4B beats that base by 14.8 points under the same
retriever (see Results). Pass `--model Qwen/Qwen2.5-3B-Instruct` to the eval
scripts to reproduce the older runs.

---

## How it works

| Part | What it does | State |
|---|---|---|
| **Statute DB** | 27 acts + the Constitution (3,736 sections), 1,257 official IPC↔BNS / CrPC↔BNSS / IEA↔BSA mappings, 70 procedural guidance notes | ✅ [`NyayaLabs98/nyaya-statute-db`](https://huggingface.co/datasets/NyayaLabs98/nyaya-statute-db) |
| **Retriever** | Exact-citation lookup (any script, old or new law), BM25 with lay-to-statute vocabulary, dense fusion with a bi-encoder fine-tuned on the project's own question–section pairs | ✅ `src/nyaya/retrieval.py`, [`NyayaLabs98/nyaya-embed-v1`](https://huggingface.co/NyayaLabs98/nyaya-embed-v1) |
| **Reranker** | Cross-encoder picks which retrieved sections actually answer the question | ✅ +12.7 points at k=1 (`bge-reranker-v2-m3`), +5.9 with the 118M [`nyaya-reranker-mini-v1`](https://huggingface.co/NyayaLabs98/nyaya-reranker-mini-v1); both validated on never-audited records |
| **Reader** | Reads the retrieved sections, writes a cited answer | `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0; +14.8 points over the 3B base, swappable) |
| **Scorer** | Grades citations strictly, substance with partial credit; every prediction is kept so scoring can be redone on CPU | ✅ `src/nyaya/scoring.py` |

Anyone can download a 3B model for free. What is hard to obtain is **current
Indian law as clean, section-level data, and a retriever that finds the right
section.** That is what this repository is.

---

## Results

Every number below comes from a committed file in `reports/`; the file is named
next to each table.

### The system, end to end

Nyaya-Eval-v1, 413 questions (409 scored), 768 new tokens, k=8, one Kaggle T4. Each
row changes one thing from the row it is paired against; every interval is a
10,000-round paired bootstrap on the same questions (`reports/eval_v1_comparison_*.json`):

| configuration | fact recall | citation accuracy | paired against | Δ fact recall, 95% CI |
|---|---|---|---|---|
| Qwen2.5-3B + BM25 + zero-shot e5-base (`base-768`, as first published) | 35.8% | 55.6% | — | — |
| Qwen2.5-3B + BM25 + `nyaya-embed-v1` | 39.7% | 61.1% | `base-768` | **+3.9 [+0.9, +7.0]** |
| Qwen3-4B + BM25 + zero-shot e5-base | 50.6% | 72.2% | `base-768` | **+14.8 [+11.4, +18.3]** |
| **Qwen3-4B + BM25 + `nyaya-embed-v1`** (default) | **52.0%** | **77.1%** | `base-768` | **+16.2 [+12.7, +19.9]** |
| | | | Qwen3-4B + e5-base | +1.4 [−1.5, +4.4], tied |
| | | | Qwen2.5-14B-AWQ + embed-v1 (served) | **+7.0 [+3.9, +10.1]** |

Two honest readings. The embedder's gain is real for the 3B reader and does not
reach significance for the 4B one (+1.4, interval spans zero; citation accuracy
+4.9, also spanning zero): a stronger reader recovers more from imperfect
retrieval. And the 4B reader beats a served 14B model from the previous Qwen
generation under identical retrieval. Answers from the default configuration
average 305 words and take ~30 s per question on a T4.

### Retrieval is where the accuracy comes from

Fact recall of the base model on Nyaya-Eval-v1, split by whether the retriever
put the gold section in front of it (`reports/eval_v1_retrieval_outcome.json`):

| retrieval outcome | n | fact recall |
|---|---|---|
| every gold statute retrieved | 94 | **63.2%** |
| a gold statute missed | 51 | **20.3%** |

Cross-encoder reranking, scored **only on the 118 records never used to tune
anything** (`reports/retrieval_recall.json`, `reports/retrieval_recall_rerank.json`,
`scripts/28_validate_generalization.py`):

| gold section in top-k | k=1 | k=3 | k=8 |
|---|---|---|---|
| BM25 + exact lookup | 45.8% | 61.0% | 81.4% |
| **+ reranker** | **58.5%** | **69.5%** | **83.9%** |

A bi-encoder fine-tuned on 4,412 of the project's own question–section pairs
([`NyayaLabs98/nyaya-embed-v1`](https://huggingface.co/NyayaLabs98/nyaya-embed-v1),
`reports/retrieval_recall_dense_embed_v1.json`), fused with BM25, same 118 records:

| gold section in top-k | k=1 | k=3 | k=8 |
|---|---|---|---|
| BM25 + exact lookup | 45.8% | 61.0% | 81.4% |
| **BM25 + nyaya-embed-v1 (RRF)** | **49.2%** | **73.7%** | **88.1%** |

And it reaches the answers. Same base reader, same 413 questions, only the dense
model swapped (`reports/eval_v1_comparison_base-768-embed-v1.json`):

| reader = base Qwen2.5-3B, 768 tokens, k=8 | fact recall | paired difference |
|---|---|---|
| dense stage: zero-shot e5-base | 35.8% | — |
| **dense stage: nyaya-embed-v1** | **39.7%** | **+3.9 points, 95% CI [+0.9, +7.0]** |

This was the first change in the project to move the end-to-end score with a
confidence interval clear of zero. It came from retrieval, not from the weights,
and `nyaya-embed-v1` is now the default dense model. With the Qwen3-4B reader the
same swap adds +1.4 points (CI [−1.5, +4.4]): real for the weaker reader, not
proven for the stronger one.

### Which small reader? Qwen3-4B, by a wide margin

Same retriever (zero-shot e5-base, k=8), same 413 questions, 768 new tokens,
one Kaggle T4, paired against `base-768` (`reports/eval_v1_comparison_qwen3-4b.json`,
`reports/eval_v1_comparison_nyaya-3b-v3-768.json`):

| reader | licence | fact recall | citation accuracy | vs base-768 | words / answer | s / question (T4) |
|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` (`base-768`) | qwen-research | 35.8% | 55.6% | — | 185 | 12.3 |
| `NyayaLabs98/nyaya-3b-v3` | qwen-research | 33.8% | 48.6% | tied on facts, CI [−5.2, +1.2]; **worse** on citations, CI [−13.9, −0.4] | 169 | 10.9 |
| **`Qwen/Qwen3-4B-Instruct-2507`** | **Apache-2.0** | **50.6%** | **72.2%** | **+14.8, CI [+11.4, +18.3]** | 306 | 31.5 |
| `Llama-3.2-3B-Instruct` (Unsloth re-upload) | llama3.2 | 36.2% | 57.3% | tied, CI [−2.5, +3.4] | 149 | 11.7 |
| `gemma-3-4b-it` | gemma | — | — | not evaluable: 413 empty answers in fp16 on a T4 | — | — |

Qwen3-4B is better on 119 questions and worse on 22. Its answers are longer, which
the substance scorer partly rewards, but citation accuracy (strict, length-blind)
rises by the same margin. It is also Apache-2.0, which the 3B base is not. It is
the default reader from this commit; combined with `nyaya-embed-v1` it is the
configuration in the table above (52.0%). Session 2 settled the 3B class: Llama-3.2-3B
and Qwen2.5-3B are indistinguishable (better on 51, worse on 58). Gemma-3-4B could
not be scored: Gemma 3 overflows in float16 and the T4 has no bf16, so every answer
came back empty (`reports/eval_v1_gemma3_fp16_failure.json`); Phi-4-mini was dropped
to stay inside the weekly GPU quota. A served 14B teacher (`Qwen2.5-14B-Instruct-AWQ`) scored 45.0%
under the embed-v1 retriever, above the 3B reader there but below Qwen3-4B, so
distillation was dropped and its 1,456 verified answers were published instead
(`NyayaLabs98/nyaya-train-v7-raft`; `docs/RESULTS.md` §1).

### Fine-tuning: five attempts, none beat the base model

Nyaya-Eval-v1, 413 gradeable questions, 409 scored, paired 10,000-round
bootstrap (`reports/eval_v1_results.json`, `reports/eval_v1_comparison_*.json`):

| reader (all with retrieval, k=8) | fact recall | vs base |
|---|---|---|
| **base Qwen2.5-3B-Instruct** | **34.3%** | — |
| base, 768-token budget | 35.8% | tied, CI [−0.2, +3.3] |
| v3 (RAFT) — the published `nyaya-3b-v3` | 32.9% | tied, CI [−4.4, +1.8] |
| v3, 768-token rerun with corrected tokenizer files | 33.8% | tied vs base-768, CI [−5.2, +1.2]; citations worse |
| v5 (grounded citation data) | 24.0% | **worse**, CI [−13.5, −7.2] |
| v6 (v5 + answer-style fix) | 23.4% | **worse**, CI [−14.0, −7.8] |

The cause is answer length, and it is monotone: base 173 words → v5 90 → v6 57.
Shorter answers carry fewer of the facts being scored. Rule-based synthetic
targets have a fixed shape, and the model learns the shape rather than the task.
One caveat: v5 and v6 ran under a retriever with an extended vocabulary table,
so "same retriever" is exact only for base vs v3 (`docs/RESULTS.md` §2).

### External benchmark

BhashaBench-Legal, 3,000 questions drawn once with a fixed seed from the full
24,365 (2,084 English, 916 Hindi), both readers on the same questions, answer
chosen by comparing the four option letters' next-token logits so nothing is
generated or parsed; chance is 25% (`reports/bhashabench_paired3000_logit.json`,
per-question rows under `reports/bhashabench_rows/`):

| reader (no retrieval; pure MCQ knowledge) | overall | English | Hindi | paired vs base |
|---|---|---|---|---|
| base Qwen2.5-3B-Instruct | 49.6% | 54.9% | 37.4% | — |
| **Qwen3-4B-Instruct-2507** | **52.5%** | 56.7% | **43.1%** | **+3.0, 95% CI [+1.0, +5.0]** |

Better on 523 questions, worse on 434, tied on 2,043. The gain is concentrated
in Hindi (+5.7 points), the project's clearest measured weakness. A first pass
that parsed generated letters was discarded: Qwen3-4B left 522 answers without a
bare letter in 8 tokens, the base 2, so those accuracies were not comparable
(`reports/bhashabench_paired3000_generation.json`). The August sample (1,500
questions, generation-scored: base 47.8%, v3 45.2%, tied) is kept in
`reports/bhashabench_scores.json` for the record.

---

## What we learned

**Retrieval plus the right open reader is the product.** Five fine-tunes of a 3B
model on Indian law were tied with or worse than the base model once the
evaluation could tell the difference. A served 14B teacher passed its gate
against the 3B reader and still scored below Qwen3-4B, so distillation had no
headroom either. What moved the score was a retriever trained on our own
question–section pairs (+3.9) and a newer-generation reader (+14.8).

**Measure before you claim.** Four bugs in our own evaluation each made a
number look like progress: the first benchmark scored its own gold answers at
10.7%, subsection citations were unmatchable, correct old→new bridging was
penalised as stale law, and a retrieval metric reported 5.1% where the truth
was 64%. A vocabulary fix that looked like +16 points was +0.9 on records it had
not been tuned against. All of it, with the corrections, is in
[`docs/RESULTS.md`](docs/RESULTS.md).

---

## Coverage and limits

- **Acts indexed:** Bharatiya Nyaya Sanhita 2023, Bharatiya Nagarik Suraksha
  Sanhita 2023, Bharatiya Sakshya Adhiniyam 2023, the Constitution, Motor
  Vehicles Act 1988, Negotiable Instruments Act 1881, Information Technology
  Act 2000, Consumer Protection Act 2019, Code on Wages 2019, Special Marriage
  Act 1954, Hindu Marriage Act 1955, Protection of Women from Domestic Violence
  Act 2005, Right to Information Act 2005, POSH Act 2013, Transfer of Property
  Act 1882, Indian Contract Act 1872, Protection of Children from Sexual Offences
  Act 2012, Juvenile Justice (Care and Protection of Children) Act 2015, Hindu
  Succession Act 1956, Indian Succession Act 1925, Dowry Prohibition Act 1961,
  Maintenance and Welfare of Parents and Senior Citizens Act 2007, Guardians and
  Wards Act 1890, Hindu Minority and Guardianship Act 1956, Hindu Adoptions and
  Maintenance Act 1956, Limitation Act 1963, Legal Services Authorities Act 1987,
  Code on Social Security 2020 (the last fourteen pulled section-by-section from
  the India Code API, `scripts/42`).
- **Absence is flagged, not hidden.** Of 269 real citizen questions collected for
  this project, 66 (25%) originally fell in domains with no act in the database —
  rent, property, loans, children, parents — and the retriever returned eight
  confident sections anyway. After the fourteen acts added from the India Code
  API, 9 (3%) remain in domains with no act at all, and a coverage gate flags
  questions the database cannot answer (`reports/coverage_probe.json`,
  `scripts/35_coverage_probe.py`).
- **Hindi.** 19 of the 53 Devanagari questions in that set retrieved no statute
  at all, because the index is built over English statute text. Rewriting the
  question into statutory English with the reader model before retrieval
  (`--rewrite`) brings that down to 3 of 53 (`reports/rewrite_measurement.json`);
  whether the sections found are the right ones still needs a Hindi holdout.
  Eval-v1 has only 9 Hindi and 10 Hinglish questions, so the multilingual claim
  rests on BhashaBench, not on our own benchmark.
- **No case law**, statutory text only. **No human evaluation** has been passed.
- `nyaya-eval-v0` is public, so it is contaminated as a held-out benchmark;
  Eval-v1's private half derives from it.

---

## Reproduce

```bash
pip install -e ".[dev]"                                    # retriever, scorer, tests
python -m pytest -q                                        # ~380 tests, CPU, no downloads

# re-grade any committed run on CPU — no model needed
python scripts/26_eval_v1_run.py --rescore outputs/eval-v1/base/predictions.jsonl --label base
python scripts/27_compare_runs.py --a base --b nyaya-3b-v3 # paired bootstrap CI

# retrieval recall (never-audited slice is the number to quote)
python scripts/15_retrieval_recall.py --k 1 3 5 8 --skip-phrase-coverage
python scripts/35_coverage_probe.py                        # coverage of real citizen questions
```

GPU runs (`pip install -e ".[train]"` or `pip install -r requirements-train.txt`)
are documented for Kaggle T4 notebooks in `scripts/kaggle_*.ipynb`. Two things
to know: an Eval-v1 **model** run needs `python scripts/25_build_eval_v1.py`
first (it regenerates the gitignored full set locally — never publish the
private half), and the reranker costs ~1.6 s per pair on CPU, so `--rerank`
sweeps belong on a GPU. Rebuilding the statute DB (`scripts/03_build_corpus.py`)
is a maintainer action; the canonical JSONL is committed.

Pinned versions that passed the suite: `requirements.lock`.

---

## Repository map

```
src/nyaya/
  retrieval.py    statute index: exact citation lookup, BM25, RRF fusion, coverage
  rerank.py       cross-encoder second stage
  dense.py        dense stage (nyaya-embed-v1 by default), fused with BM25
  scoring.py      Eval-v1 scorer: strict citations, partial-credit substance
  evaluation.py   Eval-v0 harness (retained for continuity)
  trainer.py      LoRA SFT; precision follows the hardware
  cli.py          `nyaya ask`
scripts/
  03  build the statute DB from India Code PDFs
  15  retrieval recall, with --rerank and a hard --max-minutes budget
  25  build Eval-v1 from the frozen v0 set
  26  run a model on Eval-v1; always saves raw predictions; --rescore on CPU
  27  paired bootstrap comparison between two runs
  28  does a retrieval change generalise, or did it fit the eval?
  31  publish a release to the Hub (uploads the maintained card from docs/cards/)
  35  coverage probe over real citizen questions
  36  fact recall by retrieval outcome
docs/
  RESULTS.md        everything that was measured, including what was wrong
  RELEASE_PLAN.md   what ships, and what must never be claimed
  cards/            the Hub model and dataset cards, versioned here
```

---

## Licensing — read before using the weights

- **Code**: Apache-2.0.
- **Weights**: any released Nyaya adapter or merge is a derivative of
  `Qwen/Qwen2.5-3B-Instruct`, which is **`qwen-research` — non-commercial**,
  *not* Apache-2.0. The 3B is one of the two Qwen2.5 sizes with a restricted
  licence.
- **Default reader**: `Qwen/Qwen3-4B-Instruct-2507` is Apache-2.0, as are
  `NyayaLabs98/nyaya-reranker-mini-v1` (Apache-2.0) and `NyayaLabs98/nyaya-embed-v1`
  (MIT). A deployment that uses those and not the 3B fine-tune carries no
  non-commercial term from this project.
- **Statutory text**: Government of India material, public domain under
  Section 52(1)(q) of the Copyright Act, 1957.
- Some aggregated research datasets referenced here are CC-BY-NC. See `NOTICE`.

## Claims this project does not make

One external benchmark has been run: BhashaBench-Legal, where the default reader
scores 52.5% against 49.6% for the 3B base on 3,000 paired questions (chance 25%);
no comparison against any other legal model has been made. No human evaluation has
been passed. Nyaya is **not** claimed to be the best Indian legal model, no fine-tune
here has beaten its own base model, and nothing here is legal advice.
See `docs/RELEASE_PLAN.md`.

## Citation

See `CITATION.cff`. Contributions of new acts go through `configs/acts.yaml`
(the header comment describes the recipe and the ≥98%-clean gate).
