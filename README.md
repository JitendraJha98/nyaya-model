# Nyaya-3B — Indian Legal Guidance Model

**Goal:** Train the best open legal **guidance/information** model for Indian citizens.
Any Indian, in English / Hindi / Hinglish, can ask a legal question and get an accurate,
plain-language, citation-verified answer — with a clear signal of when they need a real lawyer.

> **Positioning:** This is a legal *guidance/information* model, **not a legal advisor**.
> The Advocates Act, 1961 reserves the practice of law for enrolled advocates.
> All prompts, model cards, and outputs use "guidance" language and carry a
> "not legal advice — consult a licensed advocate" disclaimer, plus pointers to
> NALSA/DLSA free legal aid.

## Base model (frozen for v1)

`Qwen/Qwen2.5-3B-Instruct` — starting smaller than the 7B in the original plan, on purpose:
faster iteration on data quality, cheaper experiments, and the pipeline scales up to 7B later
without changes. Do **not** change models during the first experiment cycle.

Training method for v1: **SFT with LoRA** (full-precision bf16 base — no quantization).

## What "best" means (the four axes)

1. **Current law** — BNS/BNSS/BSA-native (post-July-2024), with IPC↔BNS bridging. Most existing models/datasets are stale on this.
2. **Language** — Hinglish/Hindi/English code-switching handled natively.
3. **Truthfulness** — every cited section verified programmatically against the statute DB. Citation accuracy is the primary metric, not loss.
4. **Safety** — knows its limits, escalates to "consult an advocate" appropriately.

## Repository structure

```
├── README.md
├── requirements.txt
├── docs/
│   └── ROADMAP.md           # Technical roadmap: 12 steps, data plan, eval plan, gates
├── configs/
│   ├── hf_datasets.yaml     # HF dataset IDs to pull into data/hf/
│   ├── smoke.yaml           # 1K-example smoke training config
│   └── train_v1.yaml        # Nyaya-3B-v1 full training config
├── data/
│   ├── raw/                 # Scraped statutes (India Code), mapping tables, procedure KB
│   ├── hf/                  # Hugging Face datasets, one subfolder per dataset ID
│   ├── canonical/           # Cleaned single-source-of-truth JSONL (statute DB, mappings)
│   ├── generated/           # Synthetic training examples (added later)
│   ├── validated/           # Examples that passed the validation pipeline
│   ├── splits/              # train/val/internal-test (split by source section, never by row)
│   └── eval/                # Nyaya-Eval-v0 — 500 frozen eval questions. NEVER train on these.
├── scripts/                 # Numbered pipeline steps, run in order
│   ├── 00_download_hf_datasets.py
│   ├── 01_download_model.py
│   ├── 02_run_baseline.py
│   ├── 03_build_corpus.py
│   ├── 04_generate_examples.py
│   ├── 05_validate_examples.py
│   ├── 06_deduplicate.py
│   ├── 07_create_splits.py
│   ├── 08_smoke_train.py
│   ├── 09_train.py
│   ├── 10_evaluate.py
│   └── 11_error_analysis.py
├── src/nyaya/               # Shared library code
│   ├── schemas.py           # Canonical record schemas (statute, mapping, train, eval)
│   ├── prompts.py           # System prompts + grounded generation prompt skeletons
│   ├── validators.py        # Citation verification, dedup, leakage detection
│   ├── dataset.py           # Dataset loading/formatting for TRL
│   ├── trainer.py           # LoRA training wrapper (bf16, no quantization)
│   └── evaluation.py        # Eval harness (citation accuracy, benchmarks)
├── outputs/                 # Model outputs and checkpoints (gitignored)
│   ├── baseline/            # Base Qwen predictions on Nyaya-Eval-v0
│   ├── smoke/               # Smoke-run checkpoints
│   └── nyaya-3b-v1/         # v1 checkpoints
└── reports/                 # baseline.json, validation_report.json, per-checkpoint metrics, error_analysis.json
```

## The 12-step roadmap (do in exactly this order)

```
1. Freeze model (Qwen2.5-3B-Instruct)     7. Generate training examples
2. Set up repository          ✅ done      8. Validate + deduplicate dataset
3. Download base model                     9. Create splits (by source section)
4. Create Nyaya-Eval-v0 (500 questions)   10. Run 1K-example smoke training
5. Run base-model benchmark               11. Train Nyaya-3B-v1 (8K–15K examples)
6. Build raw legal corpus                 12. Evaluate + error analysis → v2 data
```

The core loop: **DATA → TRAIN → EVALUATE → FAILURE ANALYSIS → BETTER DATA → TRAIN AGAIN.**
Never "more epochs, more epochs, more epochs."

## Quickstart

```bash
git clone https://github.com/JitendraJha98/nyaya-model.git
cd nyaya-model

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# or, to use the src/nyaya package directly:  pip install -e .

# Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Pull the ready-made Indian legal datasets from Hugging Face.
# aalap + IL-TUR are gated: log in and accept their terms on huggingface.co first.
huggingface-cli login
python scripts/00_download_hf_datasets.py

# Download + smoke-test the base model (do not train until this works)
python scripts/01_download_model.py
```

## Data plan

### Core data assets to build (in order)

| Asset | What | Where |
|---|---|---|
| 1. Statute DB | ~3–4K sections from 13 priority acts (BNS, BNSS, BSA, Constitution, CPA 2019, RTI, DV Act, HMA/SMA, NI Act, MV Act, IT Act, POSH, wages), one JSONL row per section, from indiacode.nic.in | `data/canonical/` |
| 2. IPC↔BNS / CrPC↔BNSS mapping table | ~600 + ~500 rows from official MHA comparison tables — training data AND runtime lookup | `data/canonical/` |
| 3. Procedure KB | 60–80 hand-written "how do I…" docs (FIR, consumer complaint, RTI, cheque bounce, bail, divorce, challans, cybercrime, POSH…) verified against act text | `data/raw/` → `data/canonical/` |
| 4. Judgment corpus | Existing HF sets + ~200 landmark SC judgments; bulk scraping is a v2 feature | `data/hf/` |

### Hugging Face datasets (populated by `scripts/00_download_hf_datasets.py`)

Edit `configs/hf_datasets.yaml` to add/remove IDs. Referenced in the planning docs:

| Dataset | Use |
|---|---|
| `Exploration-Lab/IL-TUR` | 8-task Indian legal benchmark (evaluation) |
| `opennyaiorg/aalap_instruction_dataset` | ~22K instruction examples (filter citizen-relevant tasks; check per-slice license) |
| BhashaBench-Legal (bharatgenai) | 24,365 exam MCQs English+Hindi (evaluation) |
| NyayaAnumana / ILDC / MILDSum | Judgment prediction & summarization corpora |
| `viber1/indian-law-dataset` | Supplementary instruction data |

**Licensing note:** government primary law is public domain (Copyright Act §52(1)(q)), but
aggregated datasets carry mixed licenses (some CC-BY-NC). Check each before commercial use.
Never scrape Indian Kanoon HTML — use the paid API or government sources.

### Synthetic data (later — `data/generated/`)

Grounded generation only: every example generated **from verbatim statute text**, never from
model memory. The single most important rule: *if you let a teacher LLM invent section numbers,
you bake hallucinations permanently into your training data.* Every generated example passes a
deterministic citation-verification gate (regex-extract sections → resolve against statute DB →
drop the whole example on any failure). Expect and welcome a 15–20% rejection rate.

First 10K target composition: 3,000 grounded legal QA · 1,500 procedural guidance ·
1,000 old→new law mapping · 1,500 Hindi · 1,500 Hinglish · 500 safety/abstention ·
500 ambiguous/insufficient info · 500 legal terminology.
**8,000 excellent examples beat 25,000 mediocre ones.**

## Non-negotiable rules

1. **Eval set first.** Build Nyaya-Eval-v0 (500 curated questions) and baseline the untouched
   base model *before* creating any training data. Without a baseline we cannot prove training helped.
2. **The eval set stays frozen.** Never train on it; run leakage detection on every dataset version.
3. **Split by source section/document, never by row.** All Q&A generated from BNS §318 go in the same split. Hold out 2 entire acts to measure generalization.
4. **Every citation verified.** Deterministic post-check against the statute DB — the primary metric.
5. **Version everything.** Every dataset a version, every experiment a config, every checkpoint traceable.
6. **Benchmark every meaningful checkpoint** — the best model might be checkpoint-750, not the final one.

## Go/no-go gates (from the master plan)

- Statute extraction spot-check ≥98% clean before generating synthetic data
- Generation rejection rate <30% before scaling to 25K
- Citation accuracy ≥95% (with RAG, in later phases) before any public demo
- Human eval passes before any "best/better than" claim

## License & disclaimer

Code is licensed under **Apache-2.0** (see `LICENSE`). See `NOTICE` for third-party
component licensing — in particular, the base model, primary law, and aggregated
datasets carry their own terms (some dataset slices are CC-BY-NC / non-commercial),
which you must verify before commercial use.

**Nyaya provides legal information/guidance, not legal advice**, and is not a substitute
for a licensed advocate. The practice of law in India is reserved to advocates under the
Advocates Act, 1961. Any deployment must carry a prominent "not legal advice — consult a
licensed advocate" disclaimer and point users to free legal aid (NALSA/DLSA).
