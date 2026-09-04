# Nyaya — handoff brief

Read this first in any new session. It is the compressed state of the project:
what is true, what was tried, what failed, and what not to repeat.

Full evidence: `docs/RESULTS.md`. Release rules: `docs/RELEASE_PLAN.md`.

---

## 1. What the project is

An **open Indian legal guidance system** — not a model. Ask a question in
English/Hindi/Hinglish, get a plain-language answer cited to a section of
current law (BNS/BNSS/BSA, post-1-July-2024, with IPC↔BNS bridging).

Four parts: **statute DB → retriever → reranker → model**. The model is
`Qwen/Qwen2.5-3B-Instruct`, unmodified. That is deliberate — see §3.

## 2. The single most important fact

**No fine-tune has ever beaten the base model.** Measured four independent ways:

| | fact recall | vs base |
|---|---|---|
| base Qwen2.5-3B + RAG | **34.3%** | — |
| v3 (RAFT) | 32.9% | tied, CI spans 0 |
| v5 (grounded citation data) | 24.0% | **worse**, CI [−13.5, −7.2] |
| v6 (v5 + answer-style fix) | 23.4% | **worse**, CI [−14.0, −7.8] |
| BhashaBench (external MCQ) | v3 45.2% vs base 47.8% | tied, CI [−6.2, +1.0] |

**Do not attempt another rule-based fine-tune.** v5 and v6 failed for the same
structural reason: templated targets have a fixed shape and the model learns
the shape, not the task. Answer length collapsed 173 → 90 → 57 words, and
shorter answers carry fewer of the facts being scored. Beating base needs a
strong teacher model (the self-hosted Gemma the project no longer has), not
another template.

## 3. What actually worked: retrieval

Accuracy is dominated by whether the right statute reaches the context:

- every gold statute retrieved → **63.2%** fact recall (n=94)
- a gold statute missed → **20.3%** (n=51)   (`reports/eval_v1_retrieval_outcome.json`)

**Cross-encoder reranking**, scored only on the 118 records never used for
tuning: k=1 45.8% → **58.5%**, k=3 61.0% → **69.5%**. That is the one
intervention in this project that generalised.

Cost: ~1.6 s/pair on CPU (≈80 s/query at depth 50) → GPU-only in practice.

## 4. Traps this project already fell into

Read before trusting any number.

1. **Eval-v0 scored its own gold answers at 10.7%.** Every result predating
   Eval-v1 is meaningless. Eval-v1 scores gold at 100% by construction.
2. **A retrieval "win" of +16 pts was +0.9 on records not used to build it.**
   Always run `scripts/28_validate_generalization.py`; `scripts/15` prints
   `full_hit_never_audited` on every run. **Quote that column, never the other.**
3. **Verbatim string matching lies.** Three separate bugs came from it —
   subsection citations unmatchable, correct IPC→BNS bridging scored as stale
   law, phrase coverage reported 5.1% when it was 64%.
4. **Timing the wrong span.** Twice a guard aborted a healthy run because the
   stopwatch included a model download. Time steady state, or parse the number
   the script itself reports.
5. **"Committed" was assumed, not checked.** `outputs/**` is gitignored;
   prediction files silently were not committed. `git ls-files` settles it.

## 5. Hardware reality (80 GB GPUs → Kaggle T4)

The original GPU environment is gone. Everything runs on Kaggle T4s now, and these were all
invisible on 80 GB GPUs:

- **bf16 is emulated on Turing.** `torch.cuda.is_bf16_supported()` returns True
  *with emulation* — ~5× slower, cost 4.5h before it was spotted. Gate on
  `get_device_capability()[0] >= 8`.
- **`device_map="auto"`** breaks TRL (accelerate wraps forward in a
  `functools.partial`). Pin to `{"": 0}`.
- **Kaggle "T4 x2"** → DataParallel splits tensors across devices. Set
  `CUDA_VISIBLE_DEVICES=0`.
- **Kaggle secrets are per-notebook**, and `machine_shape` must be
  `NvidiaTeslaT4` (not `gpuT4x2`, which silently downgrades to P100).
- **Training cost:** ~152 s/step at seq 4096 → 1,200 examples × 1 epoch ≈ 3.2h.

Every Kaggle notebook now has a timed smoke gate that projects the full run and
aborts if it exceeds budget. It has caught 6 bad runs at 2–4 minutes each.

## 6. Current state

**Published (HuggingFace, cards merged):**
- `NyayaLabs98/nyaya-statute-db` — 4,271 rows: 17 acts + Constitution, official mappings, guidance notes
- `NyayaLabs98/nyaya-3b-v3` — licence corrected to `qwen-research`
  (non-commercial, NOT Apache-2.0); card states it is tied with base

**In git:** all four prediction sets (re-scorable on CPU forever), all reports,
statute DB, retriever, reranker, scorer, `app.py` demo, 346 tests.

**Deliberately not published:** v5/v6 adapters — measurably worse than base.

## 7. What can and cannot be claimed

✅ "An open Indian legal guidance system: BNS/BNSS/BSA-native, EN/HI/Hinglish,
citation-verified, self-hostable, free. 47.8% on BhashaBench-Legal vs a 25%
chance floor."

❌ **"Best"** — at least five comparable open projects exist (LawGlance,
NYAYA.ai, nyaya-gpt, LexBharat, Legal Assist AI). Legal Assist AI is 8B and
publishes 60.08% on AIBE. And it is **unmeasurable head-to-head**: their
artifacts are RAG apps or papers, not downloadable weights.

❌ That the fine-tune beats base. It does not.
❌ Apache-2.0 on the weights. Base is `qwen-research`, non-commercial.

## 8. The best next project

**Hindi.** BhashaBench shows base at **40.7% Hindi vs 54.9% English** — a
14-point gap that no amount of retrieval work addresses. It is a real, measured
weakness, and unlike "beat base with templates" nobody has shown it is a dead
end.

Second candidate: improve first-stage retrieval recall. ~19% of gold sections
still miss at k=8, and reranking cannot recover what was never retrieved.

## 9. Housekeeping

- Rotate the HF write token and the Kaggle API key — both were pasted into a
  chat transcript.
- `jitendrajha98/bhashabench-legal-cache` (a private copy of gated data made to
  work around a missing secret) should be deleted once no longer needed.
- `NyayaLabs98/nyaya-eval-v0` was published, so it is contaminated as a
  held-out benchmark. Eval-v1's private split is derived from it and is
  therefore reconstructible — it is damage limitation, not a true holdout. A
  genuine holdout needs new, never-published questions.
