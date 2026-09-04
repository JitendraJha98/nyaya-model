# Nyaya — an open Indian legal guidance system

![tests](https://github.com/JitendraJha98/nyaya-model/actions/workflows/tests.yml/badge.svg)
![code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue)
![weights: qwen-research](https://img.shields.io/badge/weights-qwen--research%20(non--commercial)-orange)
![statutes: public domain](https://img.shields.io/badge/statutes-public%20domain-green)

Ask a legal question in **English, Hindi or Hinglish** and get a plain-language
answer **cited to a section of current Indian law** — the Bharatiya Nyaya
Sanhita, Bharatiya Nagarik Suraksha Sanhita and Bharatiya Sakshya Adhiniyam as
in force since 1 July 2024, with the official IPC↔BNS / CrPC↔BNSS bridging.

> **⚖️ Not legal advice.** Nyaya provides legal *information*. The practice of
> law in India is reserved to advocates enrolled under the Advocates Act, 1961.
> Consult a licensed advocate for anything consequential. Free legal aid is
> available through NALSA / DLSA.

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

The demo runs the **base** `Qwen/Qwen2.5-3B-Instruct` as the reader on
purpose: the project's own fine-tune, `NyayaLabs98/nyaya-3b-v3`, is
statistically tied with it (see Results), so the base model is the honest
default.

---

## How it works

| Part | What it does | State |
|---|---|---|
| **Statute DB** | 13 acts + the Constitution (2,528 sections), 1,257 official IPC↔BNS / CrPC↔BNSS / IEA↔BSA mappings, 70 procedural guidance notes | ✅ [`NyayaLabs98/nyaya-statute-db`](https://huggingface.co/datasets/NyayaLabs98/nyaya-statute-db) |
| **Retriever** | Exact-citation lookup (any script, old or new law), BM25 with lay-to-statute vocabulary, optional dense fusion | ✅ `src/nyaya/retrieval.py` |
| **Reranker** | Cross-encoder picks which retrieved sections actually answer the question | ✅ +12.7 points at k=1, validated on never-audited records |
| **Reader** | Reads the retrieved sections, writes a cited answer | `Qwen/Qwen2.5-3B-Instruct` (swappable) |
| **Scorer** | Grades citations strictly, substance with partial credit; every prediction is kept so scoring can be redone on CPU | ✅ `src/nyaya/scoring.py` |

Anyone can download a 3B model for free. What is hard to obtain is **current
Indian law as clean, section-level data, and a retriever that finds the right
section.** That is what this repository is.

---

## Results

Every number below comes from a committed file in `reports/`; the file is named
next to each table.

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

### Fine-tuning: five attempts, none beat the base model

Nyaya-Eval-v1, 413 gradeable questions, 409 scored, paired 10,000-round
bootstrap (`reports/eval_v1_results.json`, `reports/eval_v1_comparison_*.json`):

| reader (all with retrieval, k=8) | fact recall | vs base |
|---|---|---|
| **base Qwen2.5-3B-Instruct** | **34.3%** | — |
| v3 (RAFT) — the published `nyaya-3b-v3` | 32.9% | tied, CI [−4.4, +1.8] |
| v5 (grounded citation data) | 24.0% | **worse**, CI [−13.5, −7.2] |
| v6 (v5 + answer-style fix) | 23.4% | **worse**, CI [−14.0, −7.8] |

The cause is answer length, and it is monotone: base 173 words → v5 90 → v6 57.
Shorter answers carry fewer of the facts being scored. Rule-based synthetic
targets have a fixed shape, and the model learns the shape rather than the task.
One caveat: v5 and v6 ran under a retriever with an extended vocabulary table,
so "same retriever" is exact only for base vs v3 (`docs/RESULTS.md` §2).

### External benchmark

BhashaBench-Legal, 1,500-question sample (750 English, 750 Hindi), exact MCQ
scoring, chance is 25% (`reports/bhashabench_scores.json`):

| reader | overall | English | Hindi |
|---|---|---|---|
| base | **47.8%** | 54.9% | 40.7% |
| v3 | 45.2% | 51.6% | 38.8% |

Difference −2.6 points, CI [−6.2, +1.0]: tied. The 14-point Hindi gap is the
clearest measured weakness.

---

## What we learned

**Base + retrieval is the product.** Five fine-tunes of a 3B model on Indian
law were tied with or worse than the base model once the evaluation could tell
the difference. Beating base would need a strong teacher model, not another
template.

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
  Act 2005, Right to Information Act 2005, POSH Act 2013.
- **Absence is silent.** Of 269 real citizen questions collected for this
  project, 66 (25%) concern rent, property, loans, children or parents — domains
  with no act in the database — and the retriever still returns eight confident
  sections for them (`reports/coverage_probe.json`, `scripts/35_coverage_probe.py`).
  Adding those acts and a coverage gate is the current work.
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
python -m pytest -q                                        # ~360 tests, CPU, no downloads

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
  dense.py        optional dense embedding stage
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
- **Statutory text**: Government of India material, public domain under
  Section 52(1)(q) of the Copyright Act, 1957.
- Some aggregated research datasets referenced here are CC-BY-NC. See `NOTICE`.

## Claims this project does not make

One external benchmark has been run: BhashaBench-Legal (1,500-question sample,
exact MCQ scoring) — base 47.8%, v3 45.2%, tied (`reports/bhashabench_scores.json`).
No human evaluation has been passed. Nyaya is **not** claimed to be the best
Indian legal model, and no fine-tune here has beaten its own base model.
See `docs/RELEASE_PLAN.md`.

## Citation

See `CITATION.cff`. Contributions of new acts go through `configs/acts.yaml`
(the header comment describes the recipe and the ≥98%-clean gate).
