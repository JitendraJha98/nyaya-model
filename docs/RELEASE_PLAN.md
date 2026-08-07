# Nyaya — Release Plan

**Status:** draft, pending the v6 paired-CI result.
**Owner decision required before anything below goes public.**

---

## The positioning

> **Nyaya — the open Indian legal guidance system.**
> BNS/BNSS/BSA-native (post-July-2024). English / Hindi / Hinglish.
> Every answer cited to a section, verified against the statute text.

"System", not "model", and the distinction is the selling point. Anyone can
download Qwen2.5-3B for free; if our weights perform the same, we have given
them nothing. What nobody else has assembled openly is the **current-law
statute database and the retrieval stack that finds the right section**.

The four parts:

| Part | What it does | Status |
|---|---|---|
| Statute DB | 16 acts, IPC↔BNS mapping | ✅ built, publishable now |
| Retriever | BM25 + dense over the DB | ✅ built |
| **Reranker** | Picks the 3 that actually answer | ✅ **+12.7pts @k=1, generalises** |
| Model | Reads context, writes cited answer | ⏳ v6 pending |

---

## What ships, in order

### Tier 1 — ship regardless of v6 (this is the real contribution)

1. **`NyayaLabs98/nyaya-statute-db`** (dataset) — *already public, needs a card*
   16 acts as clean JSONL, one row per section, plus the official IPC↔BNS and
   CrPC↔BNSS mapping tables. Post-July-2024 law. This is the asset people will
   actually reuse, and the one thing here that is genuinely hard to reproduce.

2. **`JitendraJha98/nyaya-model`** (GitHub) — *already public*
   The retrieval stack: statute index, dense fusion, cross-encoder reranking,
   plus the pipeline that builds everything. Needs the README rewritten around
   the system rather than around "train a model".

### Tier 2 — only if v6's 95% CI excludes zero

3. **`NyayaLabs98/nyaya-3b-v6`** (model) — new repo, NOT an overwrite of v3
   Card must carry: the paired CI, the licence (`qwen-research`,
   non-commercial), what it is not, and the retrieval requirement.
   Mark `nyaya-3b-v3` superseded; do not delete it — its card documents a real
   negative result and deleting it would erase reproducible history.

### Tier 3 — later, on the owner's call

4. A demo Space. Highest-impact item for a general audience, and the fastest
   way to show the system works. Needs a GPU or a smaller reranker: the
   multilingual cross-encoder is ~1.6 s/pair on CPU, i.e. ~80 s per query.
5. Eval-v1 as a public benchmark. **Held back for now per owner instruction.**

---

## The v3 question

`NyayaLabs98/nyaya-3b-v3` is public and its card is honest: correct licence,
base-vs-v3 numbers side by side, "statistically tied with the base model",
"human-eval ship gate has NOT been passed".

Eval-v1 since confirmed the tie on a benchmark that can actually detect a
difference (32.9% vs base 34.3%, CI spans zero).

**Decision: leave it up, do not promote it.** It is an honest artifact. Never
swap different weights in under the same name — anyone who downloaded it would
silently get a different model, and its published numbers would stop
describing the file.

---

## Announcement drafts

### If v6 wins

> I built an open Indian legal AI system — and the interesting part is what
> didn't work.
>
> Nyaya answers everyday legal questions in English, Hindi and Hinglish, cites
> the exact section of current law (BNS/BNSS/BSA, post-July-2024), and runs
> free on your own machine.
>
> I fine-tuned four versions before measuring properly. Then I found my
> evaluation was scoring its own correct answers at 10.7% — it could not have
> detected improvement if there was any. Fixed it, and discovered every
> fine-tune had been statistically tied with the base model.
>
> The real bottleneck was retrieval, not the model. Adding a cross-encoder
> reranker moved the right statute into the top result 46% → 58% of the time —
> validated on questions I never tuned against.
>
> [v6 numbers + CI]
>
> Statute DB and code are open. Links below.

### If v6 ties

> I fine-tuned a 3B model on Indian law five times. None of them beat the base
> model. Here's what I learned by measuring properly.
>
> My benchmark was scoring its own gold answers at 10.7% — no model could have
> exceeded that. After fixing it: every fine-tune was tied with base, and one
> was 10 points WORSE because I had trained it to recite statute text instead
> of answering questions.
>
> The real bottleneck was never the model. When the right statute reaches the
> context, accuracy is 63%. When it doesn't, 17%. So I built a cross-encoder
> reranker: the right section now lands in the top result 46% → 58% of the
> time, validated on held-out questions.
>
> What I'm publishing: a BNS/BNSS/BSA-native statute database for post-July-2024
> Indian law, and the retrieval stack that searches it. Not a model — the
> honest result is that the base model was already good enough, and the work
> was in finding the right law.
>
> Measure before you claim.

Both drafts are true as written. The second is the stronger engineering post:
a negative result stated plainly reads as competence, and it cannot be
demolished by anyone who checks.

---

## Claims that must never be made

- ❌ "Best Indian legal model" — no external comparison has been run.
- ❌ "Better than [any model]" without a paired CI excluding zero.
- ❌ Apache-2.0 for the weights — base is `qwen-research`, non-commercial.
- ❌ Any accuracy figure from Eval-v0 — that metric scored gold answers at 10.7%.
- ❌ Retrieval numbers from the tuned-on set — quote the never-audited slice.

## Pre-publication checklist

- [ ] Licence stated as `qwen-research` non-commercial on every model card
- [ ] Every quoted number traceable to a committed report file
- [ ] Retrieval numbers taken from `full_hit_never_audited`
- [ ] "Not legal advice" + NALSA/DLSA pointer on every public surface
- [ ] Leaked credentials rotated (HF write token, Kaggle key)
