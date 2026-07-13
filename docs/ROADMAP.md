# Nyaya — Technical Roadmap

Public, technical version of the project plan. It covers *what* we build and *in what order*.
See the top-level `README.md` for the goal, repo layout, and rules.

## Guiding principle

**DATA → TRAIN → EVALUATE → FAILURE ANALYSIS → BETTER DATA → TRAIN AGAIN.**
Never "more epochs, more epochs, more epochs." Citation accuracy against the statute DB is the
primary metric — not train/val loss.

## The 12 steps (do in exactly this order)

1. **Freeze the base model** — `Qwen/Qwen2.5-3B-Instruct`. Don't change models during an experiment cycle. Method for v1: SFT with LoRA (full-precision bf16 base — no quantization).
2. **Set up the repository** — versioned data, config per experiment, traceable checkpoints.
3. **Download + smoke-test the base model** — verify inference before training anything (`scripts/01_download_model.py`).
4. **Create Nyaya-Eval-v0** — 500 manually curated eval questions, frozen. Built *before* any training data. Category split in `data/eval/README.md`.
5. **Baseline the untouched model** — run base Qwen on all 500 questions; record per-question correctness, citations, hallucination, abstention, latency. Without a baseline we can't prove training helped.
6. **Build the raw legal corpus** — the four core data assets (below).
7. **Generate training examples** — grounded synthetic data, in stages: 1K → 5K → 10K → 25K+.
8. **Validate + deduplicate** — schema, citation verification, language, length, near-dup, eval-leakage, quality scoring. A 15–20% rejection rate is healthy.
9. **Split correctly** — by source section/document, never by row; hold out 2 entire acts for generalization.
10. **Smoke-train on 1K** — verify the pipeline end-to-end (loads, tokenizes, loss drops, no OOM, checkpoints, adapter reloads, eval works). Quality is *not* the goal here.
11. **Train Nyaya-3B-v1** — 8K–15K validated examples; benchmark every meaningful checkpoint (the best may not be the last).
12. **Error analysis → v2** — bucket failures by task/domain/language/section; the failures decide what data v2 adds.

## Core data assets (build in this order)

1. **Statute DB** — ~3–4K sections from 13 priority acts (Constitution, BNS, BNSS, BSA, Consumer Protection Act 2019, RTI 2005, DV Act 2005, HMA/SMA, NI Act, MV Act 1988, IT Act 2000, POSH 2013, wages), one JSONL row per section, from `indiacode.nic.in` / `legislative.gov.in`. Pipeline: PDF → PyMuPDF text → regex section splitter → 5% manual spot-check → JSONL. Prefer HTML sources for Hindi to avoid Devanagari mojibake. **Gate: ≥98% clean before any synthetic generation.**
2. **IPC↔BNS / CrPC↔BNSS mapping tables** — ~1,100 rows from official MHA comparison tables. Doubles as training data and a runtime lookup.
3. **Procedure knowledge base** — 60–80 hand-written "how do I…" docs (FIR, consumer complaint, RTI, cheque bounce, bail, divorce, challans, cybercrime, POSH…), each verified against act text.
4. **Landmark judgments** — ~200 key Supreme Court cases pulled from existing Hugging Face datasets (not scraping). Bulk case-law is a v2 feature.

## Synthetic training data (grounded generation)

The single most important rule: **generate every example from verbatim statute text**, never from
model memory. If a teacher model invents section numbers, you bake hallucinations permanently into
the training data. Every generated example passes a deterministic citation-verification gate:
regex-extract sections → resolve against the statute DB → drop the whole example on any failure.

First 10K target composition: 3,000 grounded QA · 1,500 procedural · 1,000 old→new law mapping ·
1,500 Hindi · 1,500 Hinglish · 500 safety/abstention · 500 ambiguous/insufficient · 500 terminology.
**8,000 excellent examples beat 25,000 mediocre ones.**

## Training configuration (v1)

LoRA on the full-precision bf16 base (no quantization) · seq 4096 · r=32 / alpha=64 / dropout 0.05 ·
target all attention + MLP linear modules (`q,k,v,o,gate,up,down`) · lr 1e-4 · cosine · warmup 0.03 ·
1 epoch · effective batch 32–64 · gradient checkpointing · fused AdamW · Flash Attention 2 if
supported. Full configs in `configs/smoke.yaml` and `configs/train_v1.yaml`.

## Evaluation plan (build the harness before training run 1)

1. **Citation Accuracy Score (primary)** — % answers with all citations correct, % with any hallucinated citation, verified against the statute DB.
2. **Held-out-acts generalization** — accuracy on the 2 acts excluded from training; the honest overfitting detector.
3. **AIBE bar-exam MCQs** — a single comparable headline number.
4. **IL-TUR + BhashaBench-Legal** — standardized benchmarks, before/after deltas.
5. **Blind human eval (ship gate)** — 100 real questions, 2 raters, scored on correctness / completeness / language / safety; ship criteria: win/tie ≥60% on Hinglish citizen queries, zero dangerous answers.

Plus a 50-question regression set every checkpoint must pass (the project's unit tests).

## Go/no-go gates

- Statute extraction spot-check ≥98% clean before generation
- Generation rejection rate <30% before scaling to 25K
- Citation accuracy ≥95% (with retrieval, later phases) before any public demo
- Human eval passes before any "best/better than" claim

## Positioning & safety

Nyaya is a legal **information/guidance** model, not a legal advisor — the practice of law in India
is reserved to advocates under the Advocates Act, 1961. All prompts and outputs carry a
"not legal advice — consult a licensed advocate" disclaimer and route users to free legal aid
(NALSA/DLSA, Legal Services Authorities Act, 1987). See `NOTICE` for data/model licensing.
