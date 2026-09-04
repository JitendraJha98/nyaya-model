# Nyaya Release Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the nyaya-model repository and the NyayaLabs98 Hugging Face surfaces from "honest but unpresentable" to a citable, reproducible, cluster-free open retrieval system for current Indian law, with every GPU step running on Kaggle.

**Architecture:** Four lanes run in parallel. Lane A (repository, CPU) scrubs every trace of the decommissioned cluster, fixes the defects found in the audit, and restructures the docs. Lane B (Hugging Face, needs a write token) corrects the three cards and the org page, publishes GGUF weights and a CPU Space. Lane C (Kaggle GPU, run by the owner) produces the base-model shootout, the full BhashaBench run, and a trained small reranker. Lane D (data, CPU plus a human reviewer) widens statute coverage, adds a coverage gate, and builds a true holdout from the 269 unpublished citizen questions.

**Tech Stack:** Python 3.10+, pytest, ruff, transformers ≥ 4.56, TRL ≥ 0.12, peft, sentence-transformers, huggingface_hub, Gradio, llama.cpp GGUF (via the gguf-my-repo Space), Kaggle T4 notebooks.

**Spec:** the audit at https://claude.ai/code/artifact/49bc6358-6233-41a4-ad30-28291999fa0c (Parts III, IV, V, VI, VIII), plus `docs/RESULTS.md` and `docs/RELEASE_PLAN.md` for the claims policy.

## Status — 2026-09-04 (end of day 1)

**Done:** A1–A13; B2–B5 (cards, org page, GGUF); B6 as a free *static* Space
(`space-static/`, JS port parity-checked on 389 questions) instead of Gradio, which
now requires a paid plan; D3 (`replaces` / `punishment_summary`); D2 tooling (180
holdout drafts selected, reviewer brief); C3 preparation (4,712 retriever pairs,
training notebook); C2 notebook rewritten to load the full gated set from the Hub;
C1 session 1 running on Kaggle (T4, requested via `machine_shape`).
Recomputed figures: retrieval outcome 63.2% / **20.3%** (17.1% withdrawn); v5/v6
ran under a later retriever than base/v3; coverage gate 10.0; rewriting cuts
Devanagari zero-hit questions 19 → 3 of 53.

**Later on day 1:** India Code moved to indiacode.gov.in (DSpace 9 REST API). New
client `src/nyaya/indiacode.py` + `scripts/42` build any central act section by
section from the API (verified identical to the committed BNS on all 358 sections).
D1 done for 14 acts: Transfer of Property 1882, Indian Contract 1872, POCSO 2012,
Juvenile Justice 2015, Hindu Succession 1956, Indian Succession 1925, Dowry
Prohibition 1961, Senior Citizens 2007, Guardians and Wards 1890, Hindu Minority and
Guardianship 1956, Hindu Adoptions and Maintenance 1956, Limitation 1963, Legal
Services Authorities 1987, Code on Social Security 2020 (3,736 sections, 27 acts +
Constitution); real questions with no act in the DB fell from 66 (25%) to 9 (3%).
Not on India Code: Gratuity, Maternity Benefit and EPF Acts (absorbed into the
Social Security Code) and any Model Tenancy Act. A14 closed as NO-GO: the Hindi PDFs
are image scans (0% Devanagari in India Code's own text extraction).

**Blocked:** GitHub push and About/topics (PAT lacks Contents/Workflows/Administration
write); history rewrite (needs explicit approval of the destructive command);
C1 session 2 and C2 (need the `HF_TOKEN` Kaggle secret and accepted gated-model
terms); C4 dropped (paid teacher API); D2 gold review (needs a human reviewer).
Push retried 2026-09-04 with the same PAT: still 403 (the branch adds
`.github/workflows/tests.yml`, so the token needs *Workflows* write as well as
*Contents*; *Administration* only for About/topics).

**Day 2 (2026-09-04, afternoon).** Shootout v6 finished `base-768` (35.8% fact recall,
tied with the 384-token run: +1.5 points, CI spans zero; predictions committed) and then
crashed loading `NyayaLabs98/nyaya-3b-v3`: the Hub files were written by transformers 5.12
(`extra_special_tokens` as a list breaks 4.5x; no `rope_theta`, so 4.x silently uses
10000 instead of 1000000). Fixed files are ready (Qwen2.5-3B-Instruct tokenizer, verified
identical ids; config with both rope keys) but the Hub commit needs the owner's approval;
the shootout (v7, v3 + qwen3-4b) evaluates a locally patched copy meanwhile. The GGUF is
unaffected (rope base 1e6 in its header).
C3 done: `nyaya-train-retriever` v2 trained both models on one T4 (272 s + 132 s).
Never-audited full-hit recall @1/@3/@5/@8 — BM25 45.8/61.0/74.6/81.4; **mini reranker
51.7/70.3/76.3/82.2** (118M, 3.2 s CPU at depth 20, so not in the browser demo); **BM25 +
embed-v1 RRF 49.2/73.7/78.0/88.1** (zero-shot e5-base was below BM25 at every k). Published
`NyayaLabs98/nyaya-embed-v1` (MIT) and `NyayaLabs98/nyaya-reranker-mini-v1` (Apache-2.0)
with cards. `scripts/26` gained `--dense-model` (per-model vector cache) and `--endpoint`
(any OpenAI-compatible server, same prompts, greedy); `scripts/20` reads `TEACHER_MODEL`.
Kernel `nyaya-retriever-effect` (base reader + embed-v1 retriever, paired vs base-768) is
running. **C4 reopened without a paid API:** `scripts/kaggle_teacher.ipynb` serves
`Qwen/Qwen2.5-14B-Instruct-AWQ` with vLLM on the T4s, scores it on Eval-v1 through
`--endpoint`, and generates RAFT v7 data only if the teacher beats base-768 by ≥5 points
with a CI excluding zero; running as kernel `nyaya-teacher` v1.

**Retriever effect (kernel `nyaya-retriever-effect`, 83 min):** same base reader, 768 tokens,
k=8, dense stage swapped from zero-shot e5-base to `nyaya-embed-v1`: fact recall **35.8% →
39.7%**, paired Δ +3.9 points, 95% CI **[+0.9, +7.0]**, better on 64 / worse on 45 / tied on
300 — the first end-to-end improvement in the project with an interval clear of zero.
`nyaya-embed-v1` is now `DEFAULT_ATTACH_MODEL`; `doc_vector_cache()` keeps each embedder's
vectors apart; README, RESULTS §2/§6, the v3 card, the embedder card and the org card
carry the numbers. Kaggle source snapshot re-versioned.

**Shootout session 1 done (kernel v7, 4.9 h):** `nyaya-3b-v3-768` (corrected tokenizer/config,
local copy) 33.8% — tied with base-768 on facts (CI [−5.2, +1.2]), worse on citations
(CI [−13.9, −0.4]); **`Qwen/Qwen3-4B-Instruct-2507` 50.6% fact recall, 72.2% citation, +14.8
points, CI [+11.4, +18.3], better on 119 / worse on 22**, Apache-2.0, 306 words and 31.5 s per
question vs 185 words / 12.3 s. Default reader switched to Qwen3-4B (`MODEL_ID`; the 3B base is
`LEGACY_MODEL_ID`); unlabelled eval runs are named after the model. README/RESULTS/cards updated
and pushed. Kernel `nyaya-qwen3-embed` (Qwen3-4B + embed-v1, paired vs both parents) is running;
its number becomes the system headline. Teacher kernel v3 (vLLM tp2 came up in 170 s in v2; v2
died when the query encoder found 30 MB free on GPU 0 — fixed by pre-encoding the statute
vectors and capping vLLM at 80%) is running; its gate baseline (base-768-embed-v1) must move to
`qwen3-4b-embed-v1` once that exists — a 14B teacher has to beat the 4B reader, not the 3B.

## Global Constraints

- **GPU work runs only on Kaggle.** Accelerator "GPU T4 x2", always `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` before importing torch, fp16 (never bf16) on T4, 12-hour session cap, roughly 30 GPU-hours per week (check the quota bar in the notebook sidebar). Secrets are per notebook: add `HF_TOKEN` to each notebook that needs it.
- **No trace of the decommissioned cluster remains.** After Task A1 this must print nothing: `grep -rIn -i -E "askdata-ng|azurecr|kubectl|A100|/pvc/|hf-model-cache|inference-server|nyaya-teacher" --exclude-dir=.git --exclude-dir=archive .` and `tests/test_no_infra_traces.py` enforces it forever.
- **Never publish `data/eval/nyaya_eval_v1_private.jsonl` or `data/eval/nyaya_eval_v1.jsonl`.** The eval upload path gets an explicit allow-list (Task A7).
- **Licence wording on every public surface:** code Apache-2.0; weights derived from `Qwen/Qwen2.5-3B-Instruct` are `qwen-research`, non-commercial; statutory text public domain under Section 52(1)(q) of the Copyright Act, 1957.
- **Every public number is traceable to a committed file in `reports/`.** If a number cannot be traced, it does not go on a card or in the README.
- **Coverage wording:** "13 acts plus the Constitution (14 act files, 2,528 sections), 1,257 official IPC↔BNS / CrPC↔BNSS / IEA↔BSA mappings, 70 procedural guidance notes." Never "16 acts".
- **Naming:** public artifacts are `nyaya-3b-vN` (Hub) and `Nyaya-3B-vN` (prose); training run names stay `legal-3b-vN` only inside `configs/` and `reports/` where they already exist. No new `legal-3b-*` names.
- **Commit after every task** with a conventional-commit message. Do not push and do not tag without the owner; the owner pushes.
- **Test command** (this Windows machine needs torch imported before pyarrow): `python -c "import torch, pytest, sys; sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider']))"`. On Kaggle or Linux, plain `python -m pytest -q` works. All 346 tests pass today; the count only goes up.

---

## Lanes and parallelism

```
Day 1   A1 scrub ──► A2 archive/rename ──► A3 docs numbers ──► A4 results record
        B1 owner: rotate tokens, delete gated mirror, accept gated terms, GitHub About   (30 min, unblocks B and C)
        C0 owner: create the Kaggle notebook shells, add HF_TOKEN secret

Day 2   A5 trainer ──► A6 deps ──► A7 publish script ──► A8 script fixes
        B2 model card ──► B3 org card ──► B4 dataset cards      (needs A3 wording, A7 script)
        B5 GGUF + Modelfile                                      (independent; needs only the weights)
        C1 shootout session 1: base@768, v3@768, Qwen3-4B      (needs A8; ~5 GPU-h)

Day 3-4 A9 CI/citation ──► A10 CLI + light install ──► A11 README
        B6 Space (retrieval-only, CPU)                          (needs A10)
        C1 shootout session 2: Gemma-3-4B, Llama-3.2-3B, Phi-4-mini (~5 GPU-h)
        D1 acts: first 4 acts through the pipeline

Day 5   A12 guidance floor + coverage gate ──► A13 query rewriting ──► A14 Hindi PDF spike
        C2 BhashaBench full, model 1 (~5.5 GPU-h)
        D2 holdout: select 180 questions, hand to reviewer
        D3 populate replaces / tags / punishment_summary

Week 2  C2 BhashaBench full, model 2 ── C3 retriever pairs + train mini reranker + bi-encoder (~3 GPU-h)
        D1 acts: remaining 10 acts ── D2 gold review in progress
        B6 Space gets the mini reranker

Week 3  C3 evaluate + publish reranker ── D2 holdout scored for base, best reader, (v7 if trained)
        C4 optional v7 distillation (needs teacher API key; ~4 GPU-h)

Week 4  Write-up: HF blog post, short posts, BhashaBench leaderboard submission, release tag.
```

Kaggle GPU budget (T4, fp16):

| Run | Est. GPU-hours | Notes |
|---|---:|---|
| C1 shootout, 6 runs at 768 new tokens, batch 2 | 9 | Two sessions; base and v3 reruns remove the 384-token truncation confound |
| C2 BhashaBench-Legal, all 24,365 questions, 2 models | 11 | ~5.5 h per model; one model per session |
| C3 reranker + bi-encoder training and recall sweeps | 3 | Small models; most time is the sweep |
| C4 v7 LoRA (1,200 examples, 1 epoch) + Eval-v1 run | 4.5 | Only if the teacher data exists |
| **Total** | **~28** | Fits inside one week's quota if spread over two weeks |

---

## Lane A: repository (CPU, can start immediately)

### Task A1: Remove every trace of the decommissioned cluster

**Files:**
- Delete: `k8s/` (all 14 files), `Dockerfile`, `.dockerignore`
- Modify: `configs/generation.yaml:4-5,16,18`, `configs/smoke.yaml:7`, `configs/train_v5.yaml:51`, `docs/HANDOFF.md:35,69-72`, `docs/RESULTS.md:126-128,137`, `README.md:126`, `scripts/04_generate_examples.py:9-11,102,132`, `scripts/16_rag_eval.py:45`, `scripts/20_generate_raft.py:19`, `scripts/24_publish_hf.py:1-30`, `src/nyaya/dense.py:12`, `src/nyaya/dpo.py:3`, `src/nyaya/generation.py:7`, `src/nyaya/trainer.py:4,172`
- Create: `tests/test_no_infra_traces.py`

**Interfaces:**
- Produces: a tree where the grep in Global Constraints prints nothing; the guard test.

- [ ] **Step 1: Write the failing guard test**

```python
# tests/test_no_infra_traces.py
"""v1-v4 were trained on a private cluster that no longer exists. Nothing in
the tree may reference it: the manifests were deleted in Sept 2026 and every
remaining mention of that environment was reworded to be hardware-generic."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(
    r"askdata-ng|azurecr\.io|kubectl|\bA100\b|/pvc/|hf-model-cache"
    r"|inference-server|nyaya-teacher",
    re.IGNORECASE,
)
PATTERNS = ("*.py", "*.yaml", "*.yml", "*.md", "*.txt", "*.toml", "*.ipynb", "*.json", "*.cfg")
SKIP_PREFIXES = ("docs/archive/", "outputs/", "data/", "reports/eval_v1_kaggle_run.log")


def _scan_files():
    for pattern in PATTERNS:
        for path in ROOT.rglob(pattern):
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(".git/") or rel.startswith(SKIP_PREFIXES):
                continue
            if path.name == Path(__file__).name:
                continue
            yield path, rel


def test_no_decommissioned_cluster_references():
    hits = []
    for path, rel in _scan_files():
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if FORBIDDEN.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()[:100]}")
    assert not hits, "references to the decommissioned cluster:\n" + "\n".join(hits)


def test_no_kubernetes_manifests_or_cluster_dockerfile():
    assert not (ROOT / "k8s").exists()
    assert not (ROOT / "Dockerfile").exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -c "import torch, pytest, sys; sys.exit(pytest.main(['-q', 'tests/test_no_infra_traces.py']))"`
Expected: FAIL listing roughly 130 hits across k8s/, configs/, scripts/, src/, docs/.

- [ ] **Step 3: Delete the infrastructure files**

```bash
git rm -r k8s Dockerfile .dockerignore
```

- [ ] **Step 4: Reword every remaining mention (exact replacements)**

| File:line | Replace | With |
|---|---|---|
| `configs/generation.yaml:4-5` | `# Teacher: any OpenAI-compatible endpoint. Default is the in-cluster vLLM` / `# serving Gemma 4 31B-IT (k8s/teacher-vllm.yaml; Apache-2.0, so training` | `# Teacher: any OpenAI-compatible chat-completions endpoint. v1-v4 used a` / `# self-hosted Gemma 4 31B-IT (Apache-2.0, so training` |
| `configs/generation.yaml:16` | `base_url: "http://localhost:8000/v1"   # kubectl port-forward svc/inference-server 8000:8000 -n askdata-ng` | `base_url: "http://localhost:8000/v1"   # override with TEACHER_BASE_URL (e.g. a hosted API)` |
| `configs/generation.yaml:18` | `api_key_env: NYAYA_TEACHER_API_KEY     # optional; plain vLLM needs none` | `api_key_env: NYAYA_TEACHER_API_KEY     # required for hosted APIs; local servers may need none` |
| `configs/smoke.yaml:7` | `# the A100-80GB pool fits a 3B bf16 base + LoRA comfortably).` | `# the original 80 GB training GPUs fit a 3B bf16 base + LoRA comfortably; Kaggle T4s need train_v5's settings).` |
| `configs/train_v5.yaml:51` | `# T4s have 16GB, not the A100 80GB this pipeline was built on. Batch 1 with` | `# T4s have 16GB, not the 80 GB GPUs this pipeline was built on. Batch 1 with` |
| `docs/HANDOFF.md:35` | `strong teacher model (the A100-served Gemma the project no longer has), not` | `strong teacher model (the self-hosted Gemma the project no longer has), not` |
| `docs/HANDOFF.md:69` | `## 5. Hardware reality (A100 cluster → free GPU)` | `## 5. Hardware reality (80 GB GPUs → Kaggle T4)` |
| `docs/HANDOFF.md:71-72` | `The cluster is gone. Everything runs on Kaggle T4s now, and these were all` / `invisible on A100s:` | `The original GPU environment is gone. Everything runs on Kaggle T4s now, and these were all` / `invisible on 80 GB GPUs:` |
| `docs/RESULTS.md:126` | `## 4. Portability defects (A100 cluster → free GPU)` | `## 4. Portability defects (80 GB GPUs → free Kaggle T4)` |
| `docs/RESULTS.md:128` | `The pipeline had only ever run on 80 GB A100s. Every one of these was invisible` | `The pipeline had only ever run on 80 GB datacentre GPUs. Every one of these was invisible` |
| `docs/RESULTS.md:137` | `Every result died with the cluster that produced them.` | `Every result died with the environment that produced them.` |
| `README.md:126` | `forever — v1–v4 kept only aggregates and their results died with the cluster` | `forever — v1–v4 kept only aggregates and their results died with the GPU environment` |
| `scripts/04_generate_examples.py:9-11` | the three lines starting `Teacher: any OpenAI-compatible endpoint (configs/generation.yaml). Default is` | `Teacher: any OpenAI-compatible chat-completions endpoint (configs/generation.yaml).` / `Set TEACHER_BASE_URL and NYAYA_TEACHER_API_KEY for a hosted API.` |
| `scripts/04_generate_examples.py:102` | `# In-cluster runs reach the teacher via Service DNS — no port-forward.` | `# TEACHER_BASE_URL overrides the config so the same script works against any host.` |
| `scripts/04_generate_examples.py:132` | `# Teacher calls run concurrently (vLLM batches server-side); parsing and` | `# Teacher calls run concurrently (servers batch concurrent requests); parsing and` |
| `scripts/16_rag_eval.py:45` | `Training ran on A100s (native bf16), but evals now run on whatever free GPU` | `Training ran on GPUs with native bf16, but evals now run on whatever free GPU` |
| `scripts/20_generate_raft.py:19` | `TEACHER_BASE_URL=http://inference-server:8000/v1 python scripts/20_generate_raft.py ...` | `TEACHER_BASE_URL=https://<openai-compatible-host>/v1 python scripts/20_generate_raft.py ...` |
| `scripts/24_publish_hf.py:3-7,19-28` | every `(from the cluster PVC)`, the two lines `Run this FROM THE CLUSTER ... back up.`, and `/pvc/outputs/legal-3b-v3-merged` | `(a merged model directory)`, delete the two lines, `outputs/nyaya-3b-v3-merged` |
| `src/nyaya/dense.py:12` | `cluster PVC). Default e5-base, the model behind the committed frozen-eval` | `a named .npy). Default e5-base, the model behind the committed frozen-eval` |
| `src/nyaya/dpo.py:3` | `The cluster image pins transformers 5.x with torch 2.4: every TRL release's` | `The original training image pinned transformers 5.x with torch 2.4: every TRL release's` |
| `src/nyaya/generation.py:7` | `wires it to an OpenAI-compatible endpoint (vLLM serving the teacher model).` | `wires it to any OpenAI-compatible chat-completions endpoint.` |
| `src/nyaya/trainer.py:4` | `NO quantization — project decision (2026-07-13): the GPU pool (A100 80GB)` | `NO quantization — project decision (2026-07-13): the original 80 GB GPUs` |
| `src/nyaya/trainer.py:172` | `# inherited from the 80 GB A100 setup where it was equally unnecessary.` | `# inherited from the original 80 GB multi-GPU setup where it was equally unnecessary.` |

- [ ] **Step 5: Run the guard test and the full suite**

Run: `python -c "import torch, pytest, sys; sys.exit(pytest.main(['-q']))"`
Expected: 348 passed (346 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: remove decommissioned-cluster manifests and reword every reference to it"
```

### Task A2: Archive planning transcripts, strip the personal path, renumber the publish script

**Files:**
- Move: `docs/superpowers/` → `docs/archive/2026-07-planning/`
- Create: `docs/archive/README.md`
- Modify: `reports/error_analysis.json:3`
- Rename: `scripts/24_publish_hf.py` → `scripts/31_publish_hf.py`
- Modify: `tests/test_publish_card.py:24`, `README.md` (any `24_publish` mention), `docs/RELEASE_PLAN.md` (same)

- [ ] **Step 1: Move and document the archive**

```bash
git mv docs/superpowers docs/archive/2026-07-planning
```

`docs/archive/README.md`:
```markdown
# Archive

Working documents from earlier phases, kept for provenance. Their conclusions
were folded into `docs/RESULTS.md`; nothing here is current guidance.

- `2026-07-planning/` — design specs and task plans for the data-acquisition and
  retrieval work of July 2026.
```

- [ ] **Step 2: Strip the personal path**

In `reports/error_analysis.json` line 3 replace the value of `"predictions_file"` with `"outputs/legal-3b-v1/eval/checkpoint-50/predictions.jsonl"`.

- [ ] **Step 3: Renumber the publish script and repoint references**

```bash
git mv scripts/24_publish_hf.py scripts/31_publish_hf.py
grep -rn "24_publish_hf" --exclude-dir=.git . 
```
Change each hit (`tests/test_publish_card.py:24`, README repository map if present, RELEASE_PLAN) to `31_publish_hf.py`. Change the first docstring line of the script to `"""Step 31 — Publish a Nyaya-3B release to the Hugging Face Hub.`

- [ ] **Step 4: Run the suite**

Expected: 348 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: archive July planning docs, strip personal path, renumber publish script to 31"
```

### Task A3: Correct every number and stale claim in the docs

**Files:**
- Modify: `README.md`, `docs/RESULTS.md`, `docs/RELEASE_PLAN.md`, `docs/HANDOFF.md`, `docs/ROADMAP.md`, `data/README.md`, `data/eval/README.md`, `pyproject.toml:9`, `docs/cards/nyaya-3b-v3.md`, `docs/cards/nyaya-statute-db.md`

- [ ] **Step 1: Verify the disputed numbers from the raw files before editing**

```bash
python - <<'EOF'
import json
r = json.load(open("reports/eval_v1_results.json", encoding="utf-8"))
for k in ("base", "nyaya-3b-v3", "nyaya-3b-v5"):
    print(k, r[k]["metrics"]["fact_recall"], r[k]["metrics"]["scored_total"])
c = json.load(open("reports/eval_v1_comparison.json", encoding="utf-8"))
print("v6", c["results"][0]["b"]["mean"], c["results"][0]["ci95"])
EOF
```
Expected: base 0.3427 / v3 0.3294 / v5 0.2397 over 409; v6 0.2344, CI [-0.1396, -0.0775]. These are the only fact-recall numbers allowed anywhere.

- [ ] **Step 2: README edits**

- Line 23: `| **Statute DB** | 16 acts, one row per section, + official IPC↔BNS / CrPC↔BNSS mapping tables | ✅ |` → `| **Statute DB** | 13 acts + the Constitution (2,528 sections), 1,257 official IPC↔BNS / CrPC↔BNSS / IEA↔BSA mappings, 70 guidance notes | ✅ |`
- Line 57: `| v5 (grounded citation data) | 23.4% | ...` → `| v5 (grounded citation data) | 24.0% | **worse**, CI [−13.5, −7.2] |`
- Lines 144-148 ("Claims this project does not make"): replace the paragraph with:
  ```markdown
  One external benchmark has been run: BhashaBench-Legal (1,500-question sample,
  exact MCQ scoring) — base 47.8%, v3 45.2%, tied (`reports/bhashabench_scores.json`).
  No human evaluation has been passed. Nyaya is **not** claimed to be the best
  Indian legal model, and no fine-tune here has beaten its own base model.
  See `docs/RELEASE_PLAN.md`.
  ```
- Line 37: `Measured on Nyaya-Eval-v1 (409 gradeable questions)` → `Measured on Nyaya-Eval-v1 (413 gradeable questions, 409 scored; 4 safety rows are graded separately)`.

- [ ] **Step 3: RESULTS.md**

Section 1's second table (`| base | 173 words | 34.1% |` etc.) must use the same metric as the first table. Replace the fact-recall column with 34.3% / 24.0% / 23.4% and add under the table: `Mean answer length computed over all 413 predictions; fact recall over the 409 scored rows, as in the table above.`

- [ ] **Step 4: RELEASE_PLAN.md and HANDOFF.md**

- RELEASE_PLAN line 3: `**Status:** draft, pending the v6 paired-CI result.` → `**Status:** decided 2026-08-07 (v6 regressed; base + retrieval ships). Checklist tracked below.`
- RELEASE_PLAN line 26 `⏳ v6 pending` → `❌ v6 regressed (RESULTS §1); base Qwen2.5-3B-Instruct is the reader`.
- RELEASE_PLAN pre-publication checklist: convert each `- [ ]` to `- [x] (date)` only when the corresponding task in this plan is done; leave others open. Line 21 "16 acts, IPC↔BNS mapping" → "13 acts + Constitution, 1,257 mappings".
- HANDOFF lines 748-749 (the token line): after the owner confirms rotation (Task B1), replace with `- HF write token and Kaggle API key rotated on <date> after appearing in a chat transcript.` Line 91 "3,785 rows, 16 acts" → "3,785 rows: 13 acts + Constitution, mappings, guidance".

- [ ] **Step 5: Smaller docs**

- `docs/ROADMAP.md`: insert after the title: `> **Historical document (July 2026).** This was the original 12-step plan. What actually happened, including the retirement of the Eval-v0 metric and the finding that base + retrieval is the product, is in `docs/RESULTS.md`.`
- `data/README.md`: replace the `eval/` row with `| eval/ | nyaya_eval_v0.jsonl (frozen, now public and contaminated as a holdout) and nyaya_eval_v1_public.jsonl (public half of Eval-v1; the private half is gitignored) | **yes** |`.
- `data/eval/README.md`: append a section `## Nyaya-Eval-v1 (2026-08-06)` with: built by `scripts/25_build_eval_v1.py` from v0; 413 gradeable, 305 public / 108 private-derived rows; gold answers score 100%; the metric is `nyaya.scoring`; v0's strict metric is retired.
- `pyproject.toml:9`: `description = "Nyaya — an open Indian legal guidance system: statute DB, retriever, reranker, evaluation harness (Qwen2.5-3B-Instruct as the default reader)"`.
- `docs/cards/nyaya-statute-db.md`: heading `## Contents — 3,785 rows` → `## Contents — 3,785 rows: 14 act files, 1 mapping table, 1 guidance file`; replace the schema example's `"replaces": "IPC 302"`, `"punishment_summary": "..."`, `"tags": [...]` with `null` / `[]` and add the sentence `These three fields are reserved and currently empty; old→new bridging is done through law_mappings.jsonl.` (Task D3 fills them and reverses this.)
- `docs/cards/nyaya-3b-v3.md`: handled in Task B2 (text) because it needs the frontmatter work too.

- [ ] **Step 6: Run the suite, commit**

```bash
git add -A
git commit -m "docs: correct v5 number, act count, BhashaBench status, and stale plan statuses"
```

### Task A4: Complete the results record (v6 entry, drop the smoke run, per-candidate comparisons)

**Files:**
- Modify: `scripts/27_compare_runs.py` (add `_save_report`, call it in `main`), `reports/eval_v1_results.json`
- Create: `reports/eval_v1_comparison_nyaya-3b-v3.json`, `reports/eval_v1_comparison_nyaya-3b-v5.json`, `reports/eval_v1_comparison_nyaya-3b-v6.json`

- [ ] **Step 1: Rescore v6 into the results file and remove the 8-question smoke entry**

```bash
python scripts/26_eval_v1_run.py --rescore outputs/eval-v1/nyaya-3b-v6/predictions.jsonl --label nyaya-3b-v6
python - <<'EOF'
import json, pathlib
p = pathlib.Path("reports/eval_v1_results.json")
r = json.loads(p.read_text(encoding="utf-8"))
r.pop("v5-smoke", None)
p.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
print(sorted(r))
EOF
```
Expected: `['base', 'nyaya-3b-v3', 'nyaya-3b-v5', 'nyaya-3b-v6']` and v6 fact_recall 0.2344.

- [ ] **Step 2: Make scripts/27 write one report per candidate**

Add to `scripts/27_compare_runs.py` after `REPORT = ...`:
```python
def _save_report(report: dict, label_b: str) -> Path:
    """One file per candidate so no comparison ever overwrites another."""
    path = ROOT / "reports" / f"eval_v1_comparison_{label_b}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
```
In `main()`, where the report is currently written to `REPORT`, replace that write with `out = _save_report(report, args.b)` and print `out`. Keep writing `REPORT` as well only if you want a "latest" copy; otherwise `git rm reports/eval_v1_comparison.json` after Step 3.

- [ ] **Step 3: Produce the three comparisons**

```bash
python scripts/27_compare_runs.py --a base --b nyaya-3b-v3
python scripts/27_compare_runs.py --a base --b nyaya-3b-v5
python scripts/27_compare_runs.py --a base --b nyaya-3b-v6
```
Expected: v3 CI spans zero; v5 CI ≈ [−0.135, −0.072]; v6 CI ≈ [−0.140, −0.078]. If a CI differs from the README by more than 0.005, the README is what changes.

- [ ] **Step 4: Commit**

```bash
git add reports scripts/27_compare_runs.py
git commit -m "eval: record v6 in results, drop smoke entry, one comparison file per candidate"
```

### Task A5: Trainer passes NEFTune through, warns on dropped keys, survives a broken torch

**Files:**
- Modify: `src/nyaya/trainer.py:23-28,39-45,88-133`, `app.py:45-50`, `tests/test_trainer.py:8,77-78`, `tests/test_dpo.py:8`

- [ ] **Step 1: Failing tests**

Append to `tests/test_trainer.py`:
```python
import warnings
from pathlib import Path

from nyaya.trainer import _filter_to_signature

ROOT = Path(__file__).resolve().parents[1]


def test_neftune_passes_through_when_configured():
    config = load_config(ROOT / "configs" / "train_v3.yaml")
    assert training_kwargs(config)["neftune_noise_alpha"] == 5


def test_neftune_absent_when_not_configured():
    config = load_config(ROOT / "configs" / "smoke.yaml")
    assert training_kwargs(config).get("neftune_noise_alpha") is None


def test_filter_warns_on_every_dropped_key():
    class FakeConfig:
        def __init__(self, output_dir, max_length):
            pass

    with pytest.warns(UserWarning, match="neftune_noise_alpha"):
        out = _filter_to_signature(
            FakeConfig, {"output_dir": "x", "max_seq_length": 4, "neftune_noise_alpha": 5})
    assert out == {"output_dir": "x", "max_length": 4}


def test_filter_is_silent_when_nothing_is_dropped():
    class FakeConfig:
        def __init__(self, output_dir, max_length):
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _filter_to_signature(FakeConfig, {"output_dir": "x", "max_seq_length": 4})
```
Also change line 8 to `ROOT_SMOKE = str(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")` and lines 77-78 to use `ROOT / "configs" / "smoke.yaml"` and `ROOT / "configs" / "train_v1.yaml"` (move the `ROOT` definition above the existing tests).

- [ ] **Step 2: Run, expect failures**

Run: `python -c "import torch, pytest, sys; sys.exit(pytest.main(['-q', 'tests/test_trainer.py']))"`
Expected: `test_neftune_passes_through_when_configured` and `test_filter_warns_on_every_dropped_key` FAIL.

- [ ] **Step 3: Implement**

In `training_kwargs` add after `"eval_steps": t.get("eval_steps"),`:
```python
        # NEFTune (embedding noise) is configured in v3/v4/v5 but was never
        # forwarded until Sept 2026 -- those runs trained WITHOUT it.
        "neftune_noise_alpha": t.get("neftune_noise_alpha"),
```
Replace `_filter_to_signature` with:
```python
def _filter_to_signature(cls, kwargs: dict) -> dict:
    """Keep only kwargs the installed TRL/transformers version accepts.

    Handles the max_seq_length -> max_length rename. Every OTHER dropped key is
    reported with a warning: silently swallowing unknown keys is how NEFTune
    went missing from three training runs."""
    import inspect
    import warnings

    accepted = set(inspect.signature(cls.__init__).parameters)
    out, dropped = {}, []
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in accepted:
            out[key] = value
        elif key == "max_seq_length" and "max_length" in accepted:
            out["max_length"] = value
        else:
            dropped.append(key)
    if dropped:
        warnings.warn(
            f"[trainer] {cls.__name__} does not accept and will IGNORE: {sorted(dropped)}",
            UserWarning, stacklevel=2)
    return out
```
In `_cuda_available` and `_native_bf16` change `except ImportError:` to `except (ImportError, OSError):`. Same in `app.py` `_has_gpu`. In `tests/test_dpo.py` replace `pytest.importorskip("torch")` with:
```python
try:
    import torch  # noqa: F401
except (ImportError, OSError) as exc:  # a broken CUDA DLL raises OSError, not ImportError
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)
```

- [ ] **Step 4: Run the whole suite**

Expected: 352 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nyaya/trainer.py app.py tests/test_trainer.py tests/test_dpo.py
git commit -m "fix(trainer): forward neftune_noise_alpha, warn on dropped keys, tolerate broken torch"
```

### Task A6: Dependency floors, optional extras, lock file, OOM alias, dtype consistency

**Files:**
- Modify: `requirements.txt`, `pyproject.toml`, `scripts/26_eval_v1_run.py:77`, `scripts/01_download_model.py:16`, `scripts/02_run_baseline.py:41`
- Create: `requirements-train.txt`, `requirements.lock`, `tests/test_requirements.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_requirements.py
"""The code calls APIs that only exist above these versions (audit, Sept 2026):
from_pretrained(dtype=) needs transformers>=4.56, SFTTrainer(processing_class=)
needs trl>=0.12, torch.cuda.OutOfMemoryError needs torch>=2.5."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOORS = {"transformers": (4, 56), "trl": (0, 12), "torch": (2, 5)}


def _floor(text: str, pkg: str):
    m = re.search(rf"^{pkg}>=(\d+)\.(\d+)", text, re.MULTILINE)
    assert m, f"{pkg} has no >= floor"
    return int(m.group(1)), int(m.group(2))


def test_training_floors_match_the_apis_the_code_uses():
    text = (ROOT / "requirements-train.txt").read_text(encoding="utf-8")
    for pkg, floor in FLOORS.items():
        assert _floor(text, pkg) >= floor, f"{pkg} floor below {floor}"


def test_core_requirements_do_not_pull_the_training_stack():
    core = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for heavy in ("torch", "trl", "wandb", "gradio", "boto3"):
        assert not re.search(rf"^{heavy}\b", core, re.MULTILINE), f"{heavy} belongs in an extra"


def test_oom_handler_uses_the_portable_exception_name():
    src = (ROOT / "scripts" / "26_eval_v1_run.py").read_text(encoding="utf-8")
    assert "torch.cuda.OutOfMemoryError" in src
    assert "except torch.OutOfMemoryError" not in src
```

- [ ] **Step 2: Run, expect failures** (`requirements-train.txt` missing).

- [ ] **Step 3: Write the requirement files**

`requirements.txt` (core: retrieval, scoring, CLI; standard library does the rest):
```
# Core — retrieval, scoring, data build, CLI. No GPU stack.
# Training/eval on GPU: pip install -r requirements-train.txt   (or pip install -e ".[train]")
pyyaml>=6.0
requests>=2.32
huggingface_hub>=0.24
```
`requirements-train.txt`:
```
# GPU training + evaluation stack (Kaggle T4 tested). Floors are the versions the
# code actually needs: from_pretrained(dtype=) 4.56, processing_class= trl 0.12,
# torch.cuda.OutOfMemoryError 2.5. sentence-transformers 5.x pins transformers<5.
torch>=2.5
transformers>=4.56,<5
datasets>=2.20
accelerate>=0.33
peft>=0.12
trl>=0.12
sentencepiece>=0.2
protobuf>=4.25
sentence-transformers>=3.0
numpy
```
`pyproject.toml`: replace `dependencies = { file = ["requirements.txt"] }` block's neighbours with:
```toml
[project.optional-dependencies]
train = ["torch>=2.5", "transformers>=4.56,<5", "datasets>=2.20", "accelerate>=0.33", "peft>=0.12", "trl>=0.12", "sentencepiece>=0.2", "protobuf>=4.25"]
dense = ["sentence-transformers>=3.0", "numpy"]
data = ["pymupdf>=1.24", "boto3>=1.34"]
demo = ["gradio>=4.0"]
dev = ["pytest>=8", "ruff>=0.5", "datasets>=2.20"]
```
Delete `requirements-dense.txt` and point its two references (README Quickstart, `src/nyaya/dense.py` docstring) at `pip install -e ".[dense]"`.

- [ ] **Step 4: Code fixes**

- `scripts/26_eval_v1_run.py:77`: `except torch.OutOfMemoryError:` → `except torch.cuda.OutOfMemoryError:`
- `scripts/01_download_model.py:16` and `scripts/02_run_baseline.py:41`: `torch_dtype=` → `dtype=`

- [ ] **Step 5: Lock file from an environment that passes the suite**

```bash
python -m venv .venv-lock && .venv-lock/Scripts/pip install -e ".[train,dense,data,demo,dev]" && .venv-lock/Scripts/pip freeze > requirements.lock && .venv-lock/Scripts/python -m pytest -q
```
Expected: all tests pass in the fresh venv; `requirements.lock` committed; `.venv-lock/` added to `.gitignore`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "build: raise dependency floors to the APIs in use, split extras, add lock file, portable OOM alias"
```

### Task A7: Publish script reads the maintained card, allow-lists the eval upload

**Files:**
- Modify: `scripts/31_publish_hf.py` (remove `build_model_card`, add `load_model_card`, fix `--publish-eval`, default version, commit message), `tests/test_publish_card.py`

- [ ] **Step 1: Rewrite the tests to guard the maintained card**

Replace `tests/test_publish_card.py` body from the `cards` fixture down with:
```python
@pytest.fixture(scope="module")
def cards(pub):
    return {v: pub.load_model_card(v) for v in pub.VERSIONS}


def test_every_version_card_loads(cards, pub):
    assert set(cards) == set(pub.VERSIONS)


def test_card_never_claims_apache_for_the_weights(cards):
    for version, card in cards.items():
        frontmatter = card.split("---")[1]
        assert "license: other" in frontmatter, version
        assert "license_name: qwen-research" in frontmatter, version
        assert "apache" not in frontmatter.lower(), version
        assert "non-commercial" in card.lower(), version


def test_card_states_the_tie_and_the_missing_human_eval(cards):
    for version, card in cards.items():
        assert "statistically tied" in card, version
        assert "No human evaluation" in card, version


def test_card_discloses_eval_contamination(cards):
    for version, card in cards.items():
        assert "contaminated" in card, version


def test_card_keeps_the_not_legal_advice_disclaimer(cards):
    for version, card in cards.items():
        assert "Not legal advice" in card, version
        assert "Advocates Act, 1961" in card, version


def test_missing_card_is_fatal(pub):
    with pytest.raises(pub.CardDataError):
        pub.load_model_card("v99")


def test_eval_upload_allowlist_can_never_match_the_private_split(pub):
    assert all("private" not in pattern for pattern in pub.EVAL_ALLOW_PATTERNS)
    assert "nyaya_eval_v1.jsonl" not in pub.EVAL_ALLOW_PATTERNS
    assert "nyaya_eval_v0.jsonl" in pub.EVAL_ALLOW_PATTERNS
```
Update the module docstring to say the card is maintained by hand in `docs/cards/` and these tests guard that file.

- [ ] **Step 2: Run, expect failures** (`load_model_card`, `EVAL_ALLOW_PATTERNS` undefined).

- [ ] **Step 3: Implement in `scripts/31_publish_hf.py`**

- Delete `build_model_card`, `_pct`, `_need`, `BASE_EVAL_KEY`, and the `v4` entry of `VERSIONS` (never published). Keep `CardDataError`.
- Add:
```python
CARDS = ROOT / "docs" / "cards"
EVAL_ALLOW_PATTERNS = ["nyaya_eval_v0.jsonl", "README.md"]


def load_model_card(version: str) -> str:
    """The card is maintained by hand in docs/cards/ (it is prose about
    results, not a template). This only checks the parts that must never
    regress before anything is uploaded."""
    path = CARDS / f"nyaya-3b-{version}.md"
    if not path.exists():
        raise CardDataError(f"no card at {path}")
    card = path.read_text(encoding="utf-8")
    parts = card.split("---")
    if len(parts) < 3:
        raise CardDataError(f"{path.name}: missing YAML frontmatter")
    for must in ("license: other", "license_name: qwen-research"):
        if must not in parts[1]:
            raise CardDataError(f"{path.name}: frontmatter lacks '{must}'")
    return card
```
- In `main()`: `default="v3"`; `card = load_model_card(args.version)`; card-only commit message `f"Update model card ({datetime.date.today().isoformat()})"` with no `commit_description`; in the `--publish-eval` block pass `allow_patterns=EVAL_ALLOW_PATTERNS` to `upload_folder`.

- [ ] **Step 4: Make the maintained card satisfy the new tests**

In `docs/cards/nyaya-3b-v3.md` "Honest status" add the bullet: `- `nyaya-eval-v0` is public, so it is **contaminated** as a held-out benchmark; Eval-v1's private half derives from it.` (The rest of the card is rewritten in Task B2; do that edit there too if B2 runs first.)

- [ ] **Step 5: Run the suite, commit**

```bash
git add scripts/31_publish_hf.py tests/test_publish_card.py docs/cards/nyaya-3b-v3.md
git commit -m "fix(publish): upload the maintained card, allow-list the eval upload, default to v3"
```

### Task A8: Eval scripts record what they ran

**Files:**
- Modify: `scripts/26_eval_v1_run.py:51,119-136,172-173`, `scripts/16_rag_eval.py:186` and its `build_rag_generate_fn`

- [ ] **Step 1: scripts/26**

- Replace the module constant `RESULTS = ROOT / "reports" / "eval_v1_results.json"` with a function used by `_save`:
```python
def _results_path(out_dir: Path) -> Path:
    return out_dir / "reports" / "eval_v1_results.json"
```
  and in `_save` use `results_path = _results_path(out_dir)` everywhere `RESULTS` was used.
- Lines 172-173: replace the label fallback with
```python
    label = args.label or (
        "base" if args.model == MODEL_ID and not adapter
        else Path(args.model).name + (f"+{Path(adapter).name}" if adapter else ""))
```
- `--max-new-tokens` keeps its default of 384 (comparability with committed runs); the shootout passes 768 explicitly.

- [ ] **Step 2: scripts/16**

At line 186 change `"model": MODEL_ID` to record the model id actually loaded (the variable passed to `load_model(..., model_id=...)`; name it `model_id` in `main()` if it is not already). In `build_rag_generate_fn`, wrap the `apply_chat_template` call:
```python
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # templates without a system role (some Gemma builds)
            merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
            text = tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
```
This is what lets the shootout (Task C1) run Gemma-3 without a fork.

- [ ] **Step 3: Test for the label fallback**

Add to a new `tests/test_eval_v1_run.py`:
```python
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("eval_v1_run", ROOT / "scripts" / "26_eval_v1_run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_results_path_follows_out_dir(tmp_path):
    assert _mod()._results_path(tmp_path) == tmp_path / "reports" / "eval_v1_results.json"
```

- [ ] **Step 4: Run the suite, commit**

```bash
git add scripts/26_eval_v1_run.py scripts/16_rag_eval.py tests/test_eval_v1_run.py
git commit -m "fix(eval): honour --out-dir for metrics, label adapter runs, record the real model id, tolerate system-less chat templates"
```

### Task A9: CI, citation, changelog, release tag

**Files:**
- Create: `.github/workflows/tests.yml`, `CITATION.cff`, `CHANGELOG.md`

- [ ] **Step 1: Workflow**

```yaml
# .github/workflows/tests.yml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -e ".[dev]"
      - run: ruff check --select E9,F63,F7,F82 src scripts tests
      - run: python -m pytest -q
```

- [ ] **Step 2: CITATION.cff**

```yaml
cff-version: 1.2.0
message: "If you use the Nyaya statute database, retriever or evaluation harness, please cite it."
title: "Nyaya: an open retrieval system for current Indian law"
type: software
authors:
  - family-names: Jha
    given-names: Jitendra
  - family-names: Nikumbh
    given-names: Siddhant
repository-code: "https://github.com/JitendraJha98/nyaya-model"
license: Apache-2.0
version: 0.3.0
date-released: 2026-09-04
```

- [ ] **Step 3: CHANGELOG.md**

```markdown
# Changelog

## 0.3.0 — 2026-09 (release hardening)
- Removed the decommissioned-cluster manifests; all GPU work now documented for Kaggle T4.
- Eval-v1 record completed: v6 scored, one paired comparison file per candidate.
- Trainer forwards NEFTune (v3/v4/v5 had trained without it) and warns on dropped keys.
- Dependency floors raised to the APIs in use; core install no longer pulls the GPU stack.
- Publish script uploads the maintained card and allow-lists the eval upload.
- Docs: act count corrected to 13 + Constitution; v5 = 24.0%; BhashaBench status stated.

## 0.2.0 — 2026-08-07
- Eval-v1 (gold ceiling 100%), cross-encoder reranker (+12.7 @k=1 on never-audited records),
  BhashaBench-Legal sample run, v5/v6 negative results, model card corrected.
```

- [ ] **Step 4: README badge line** (top of README, under the title): `![tests](https://github.com/JitendraJha98/nyaya-model/actions/workflows/tests.yml/badge.svg) ![code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue) ![weights: qwen-research](https://img.shields.io/badge/weights-qwen--research%20(non--commercial)-orange) ![data: public domain](https://img.shields.io/badge/statutes-public%20domain-green)`

- [ ] **Step 5: Commit; the owner tags after pushing**

```bash
git add .github CITATION.cff CHANGELOG.md README.md
git commit -m "ci: test workflow, citation metadata, changelog"
# owner, after push and green CI:
git tag -a v0.3.0 -m "Release hardening: cluster-free, complete Eval-v1 record" && git push --tags
```

### Task A10: Light install and a `nyaya ask` CLI

**Files:**
- Create: `src/nyaya/cli.py`, `tests/test_cli.py`
- Modify: `pyproject.toml` (`[project.scripts]`)

- [ ] **Step 1: Failing test**

```python
# tests/test_cli.py
from pathlib import Path

from nyaya import cli

ROOT = Path(__file__).resolve().parents[1]


def test_ask_prints_statute_sections(capsys):
    rc = cli.main(["ask", "cheque bounce notice period", "--k", "3",
                   "--canonical-dir", str(ROOT / "data" / "canonical")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Negotiable Instruments Act" in out
    assert "Section" in out
```

- [ ] **Step 2: Implement**

```python
# src/nyaya/cli.py
"""`nyaya` command line: retrieve the statute sections behind a question.

Retrieval only, standard library only. The statute DB is read from
data/canonical when run inside the repo, otherwise downloaded once from the
Hub dataset NyayaLabs98/nyaya-statute-db (~5 MB)."""
import argparse
from pathlib import Path

from .retrieval import format_context, load_statute_index

HUB_DATASET = "NyayaLabs98/nyaya-statute-db"


def _canonical_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    local = Path(__file__).resolve().parents[2] / "data" / "canonical"
    if local.exists():
        return local
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(HUB_DATASET, repo_type="dataset", allow_patterns=["*.jsonl"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nyaya", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="show the sections of current Indian law a question retrieves")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=5)
    ask.add_argument("--canonical-dir", default=None, help="override the statute DB directory")
    args = p.parse_args(argv)

    index = load_statute_index(_canonical_dir(args.canonical_dir))
    rows = index.retrieve(args.question, k=args.k)
    print(format_context(rows))
    print("\n⚖️ Legal information, not legal advice. Consult a licensed advocate; free legal aid: NALSA / DLSA.")
    return 0
```
`pyproject.toml`:
```toml
[project.scripts]
nyaya = "nyaya.cli:main"
```

- [ ] **Step 3: Run test, then try it**

```bash
pip install -e . && nyaya ask "police FIR nahi likh rahi, kya karu?"
```
Expected: BNSS sections 173/175 and the FIR-refusal guidance note.

- [ ] **Step 4: Commit**

```bash
git add src/nyaya/cli.py tests/test_cli.py pyproject.toml
git commit -m "feat: nyaya ask CLI over the standard-library retriever"
```

### Task A11: README restructure

**Files:**
- Modify: `README.md` (full rewrite of structure; every number from Task A3)

- [ ] **Step 1: New outline, in this order**

1. Title + badges (Task A9) + one line: *Ask a legal question in English, Hindi or Hinglish; get a plain-language answer cited to a section of current Indian law (BNS / BNSS / BSA, post-1 July 2024).* Not-legal-advice block.
2. **Try it** — Space link (Task B6), `pip install nyaya` is not on PyPI yet so: `pip install git+https://github.com/JitendraJha98/nyaya-model` then `nyaya ask "..."`; a 20-second GIF of the Space (record after B6).
3. **How it works** — a 5-row table: statute DB → retriever → reranker → reader (Qwen2.5-3B-Instruct, swappable) → citation scorer; one sentence each; one line explaining that the demo runs the base model because v3 is statistically tied with it.
4. **Results** — the reranker table (never-audited slice), the 63.2% / 17.1% retrieval-outcome table, the BhashaBench line, the base-model shootout table once Task C1 lands; each table followed by the path of the report file it comes from.
5. **What we learned** — five fine-tunes tied or worse; the four metric bugs; two paragraphs, link to RESULTS.md.
6. **Coverage and limits** — the 13 + Constitution list, the out-of-coverage finding (a quarter of real questions) with the coverage gate from Task A12, Hindi status.
7. **Reproduce** — light install; `[train]` extra and the Kaggle notebooks for GPU; `26 --rescore` on the committed predictions; the note that Eval-v1 model runs need `scripts/25` first.
8. **Repository map**, **Licensing**, **Citation** (CITATION.cff), **Contributing an act** (points at `configs/acts.yaml` and Task D1's recipe).

- [ ] **Step 2: Remove from the README** the Quickstart line `python scripts/03_build_corpus.py` (the DB is committed; rebuilding is a maintainer action) and the whole "Four measurement bugs" section body (summarise in two lines, link RESULTS §3).

- [ ] **Step 3: Commit** `docs: restructure README around the system; every number linked to reports/`

### Task A12: Guidance relevance floor and coverage gate in the retriever

**Files:**
- Modify: `src/nyaya/retrieval.py` (`retrieve`, new `coverage`), `app.py` (`answer`), `tests/test_retrieval.py`
- Create: `scripts/32_calibrate_retrieval.py`

**Interfaces:**
- Produces: `StatuteIndex.coverage(query) -> dict` with keys `top_statute_score: float`, `covered: bool`; constants `GUIDANCE_FLOOR_RATIO = 0.35`, `COVERAGE_MIN_SCORE` (set by the calibration script).

- [ ] **Step 1: Failing tests** (append to `tests/test_retrieval.py`; the file already builds small synthetic indexes, reuse its helper for rows)

```python
def test_guidance_is_not_appended_when_it_is_irrelevant():
    rows = [
        {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023", "section": "103",
         "title": "Punishment for murder", "text": "Whoever commits murder shall be punished with death or imprisonment for life"},
        {"act_id": "procedures_kb", "act_name": "Official Procedural Guidance (India)", "section": "drunk-driving",
         "title": "Penalties for drunk driving", "text": "Driving under the influence of drink attracts a fine"},
    ]
    index = StatuteIndex(rows, mappings=[])
    hits = index.retrieve("what is the punishment for murder", k=4)
    assert [h["section"] for h in hits] == ["103"]


def test_guidance_is_appended_when_it_matches():
    rows = [
        {"act_id": "bnss_2023", "act_name": "Bharatiya Nagarik Suraksha Sanhita, 2023", "section": "173",
         "title": "Information in cognizable cases", "text": "Every information relating to the commission of a cognizable offence"},
        {"act_id": "procedures_kb", "act_name": "Official Procedural Guidance (India)", "section": "fir-refusal-remedy",
         "title": "What to do when police refuse to register an FIR", "text": "Send the information in writing to the Superintendent of Police"},
    ]
    index = StatuteIndex(rows, mappings=[])
    hits = index.retrieve("police refuse to register my FIR information cognizable", k=4)
    assert "fir-refusal-remedy" in [h["section"] for h in hits]


def test_coverage_reports_uncovered_for_a_query_with_no_statute_signal():
    rows = [{"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023", "section": "103",
             "title": "Punishment for murder", "text": "Whoever commits murder shall be punished"}]
    index = StatuteIndex(rows, mappings=[])
    assert index.coverage("security deposit refund landlord")["covered"] is False
    assert index.coverage("punishment for murder")["covered"] is True
```

- [ ] **Step 2: Implement in `retrieval.py`**

Add constants near `KB_SLOTS`:
```python
# A guidance note only rides along when its own BM25 score is at least this
# fraction of the best statute score. Before Sept 2026 the two KB_SLOTS were
# filled unconditionally: 413/413 eval answers got two notes, e.g. drunk-driving
# penalties appended to "what is the punishment for murder".
GUIDANCE_FLOOR_RATIO = 0.35
# Below this best-statute BM25 score the query is treated as outside the
# database's coverage (about a quarter of real citizen questions: tenancy,
# property, loans, children). Calibrated by scripts/32_calibrate_retrieval.py.
COVERAGE_MIN_SCORE = 4.0
```
In `retrieve`, replace the block from `bm25_order = ...` through the guidance selection so that scores are kept:
```python
        bm25 = self._bm25(query)
        score_of = {i: s for s, i in bm25}
        bm25_order = [i for _s, i in bm25]
        order = (rrf_fuse([bm25_order, self.dense.rank(query)])
                 if self.dense is not None else bm25_order)

        statutes, guidance = [], []
        for i in order:
            row = self.rows[i]
            if f"{row['act_id']}:{row['section'].upper()}" in chosen:
                continue
            (guidance if row["act_id"] == "procedures_kb" else statutes).append((i, row))

        top_statute = max((score_of.get(i, 0.0) for i, _r in statutes), default=0.0)
        floor = GUIDANCE_FLOOR_RATIO * top_statute
        guidance = [(i, r) for i, r in guidance if score_of.get(i, 0.0) >= floor]
        statute_rows = [r for _i, r in statutes]
        guidance_rows = [r for _i, r in guidance]
```
then use `statute_rows` / `guidance_rows` where the old code used `statutes` / `guidance`. Add the method:
```python
    def coverage(self, query: str) -> dict:
        """Is this question inside the acts the database holds?"""
        best = 0.0
        for score, i in self._bm25(query):
            if self.rows[i]["act_id"] != "procedures_kb":
                best = score
                break
        return {"top_statute_score": round(best, 3), "covered": best >= COVERAGE_MIN_SCORE}
```
Exact-citation hits (`picked`) always count as covered: in `coverage`, return `covered=True` early if `self.referenced_keys(query)` is non-empty.

- [ ] **Step 3: Calibration script**

`scripts/32_calibrate_retrieval.py`: loads the index; for every gold-bearing Eval-v1 record whose id is not in `reports/audited_record_ids.json`, records `coverage()["top_statute_score"]`; for every line of `data/raw/citizen_questions.txt` matching the out-of-coverage regex `rent|landlord|tenant|kiraya|makan malik|property|zameen|plot|will\b|inheritance|loan|emi|contract|agreement|pocso|custody|senior citizen|maa baap|dowry|dahej` records the same; prints the 5th percentile of the in-coverage scores and the fraction of out-of-coverage questions below it, and writes `reports/coverage_calibration.json`. Set `COVERAGE_MIN_SCORE` to that 5th percentile, rounded down to one decimal, and record the number in the constant's comment.

- [ ] **Step 4: Wire into the demo**

In `app.py` `answer()`, before retrieval: `cov = index.coverage(question)`; if not covered, return `"The acts in this database do not appear to cover this question. For free legal aid contact NALSA / DLSA (15100)."` and still show the retrieved sections on the right so the user can judge.

- [ ] **Step 5: Measure, then run the suite**

```bash
python scripts/15_retrieval_recall.py --k 1 3 5 8 --skip-phrase-coverage
```
Expected: `full_hit_never_audited` at k=8 unchanged (0.8136) because guidance never occupied statute slots; the report now also prints how many queries received guidance.

- [ ] **Step 6: Commit** `feat(retrieval): relevance floor on guidance notes and a coverage gate`

### Task A13: Query rewriting for Devanagari and Hinglish questions

**Files:**
- Create: `src/nyaya/rewrite.py`, `tests/test_rewrite.py`, `scripts/33_measure_rewrite.py`
- Modify: `scripts/16_rag_eval.py` (`--rewrite` flag), `scripts/26_eval_v1_run.py` (pass-through), `app.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_rewrite.py
from nyaya.rewrite import needs_rewrite, REWRITE_PROMPT


def test_devanagari_and_hinglish_need_rewrite_english_does_not():
    assert needs_rewrite("मकान मालिक डिपॉजिट वापस नहीं कर रहा")
    assert needs_rewrite("makan malik deposit wapas nahi kar raha")
    assert not needs_rewrite("landlord is not returning my security deposit")


def test_prompt_asks_for_statutory_english_only():
    assert "one line" in REWRITE_PROMPT.lower()
    assert "do not answer" in REWRITE_PROMPT.lower()
```

- [ ] **Step 2: Implement**

```python
# src/nyaya/rewrite.py
"""Rewrite a Hindi / Hinglish question into the English statutory vocabulary
the BM25 index is built on. 36% of pure-Devanagari citizen questions retrieved
zero statute sections before this (Sept 2026 audit); the synonym table in
retrieval.py could not keep up by hand."""
import re

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_HINGLISH_MARKERS = re.compile(
    r"\b(kya|kaise|nahi|nahin|hai|hain|karu|karun|mera|meri|mujhe|wapas|raha|rahi|"
    r"kar|ho|gaya|gayi|wala|wale|bhai|bhaiya|police|thana|paisa|paise)\b", re.IGNORECASE)

REWRITE_PROMPT = (
    "Rewrite the following Indian citizen's legal question as ONE LINE of plain English "
    "using the words an Indian statute would use (e.g. 'cheating', 'dishonour of cheque', "
    "'information about a cognizable offence', 'maintenance', 'shared household'). "
    "Do not answer the question. Output only the rewritten line.\n\nQuestion: {question}\nRewritten:"
)


def needs_rewrite(question: str) -> bool:
    if _DEVANAGARI.search(question):
        return True
    return len(_HINGLISH_MARKERS.findall(question)) >= 2


def rewrite_query(question: str, generate) -> str:
    """`generate(prompt: str) -> str` is any text generator (the 3B reader itself)."""
    if not needs_rewrite(question):
        return question
    out = generate(REWRITE_PROMPT.format(question=question)).strip().splitlines()
    rewritten = out[0].strip() if out else ""
    # Keep the original too: exact citations and Devanagari synonyms still match it.
    return f"{question} {rewritten}" if rewritten else question
```
In `scripts/16_rag_eval.py` add `--rewrite` (store_true); when set, `build_rag_generate_fn` calls `rewrite_query(question, plain_generate)` before `index.retrieve`, where `plain_generate` is a one-turn greedy generation of ≤ 40 tokens with the same model. Pass the flag through `scripts/26`.

- [ ] **Step 3: Measure on the real questions (CPU is fine, ~10 s per rewrite)**

`scripts/33_measure_rewrite.py`: for the 53 Devanagari lines of `data/raw/citizen_questions.txt`, count queries with zero statute rows in the top 8 with and without rewriting; print both and write `reports/rewrite_devanagari.json`. Then on Kaggle (Task C1 session 2) run `26 --rewrite` for base and compare with `27`.

- [ ] **Step 4: Commit** `feat(retrieval): rewrite Hindi/Hinglish questions into statutory English before retrieval`

### Task A14: Hindi statute text spike (go/no-go)

**Files:**
- Modify: `docs/RESULTS.md` (record the outcome either way)

- [ ] **Step 1: Fetch and extract one Hindi act**

```bash
python scripts/13_download_raw_assets.py --group hindi_statutes
python - <<'EOF'
import re, sys
sys.path.insert(0, "src")
from nyaya.corpus import extract_pdf_text, slice_act_body, split_sections
text = extract_pdf_text("data/raw/assets/hindi_statutes/bns_2023_hi.pdf")
dev = len(re.findall(r"[ऀ-ॿ]", text)); letters = len(re.findall(r"\w", text))
print("devanagari share of word chars:", round(dev / max(1, letters), 3))
print("sections split:", len(split_sections(slice_act_body(text))))
EOF
```

- [ ] **Step 2: Decide**

- Share ≥ 0.90 and ≥ 300 sections: GO. Open a follow-up task to add a `text_hi` field to the BNS/BNSS/BSA rows (split by the Hindi section pattern `धारा`/numeral) and index `text_hi` tokens alongside English in `StatuteIndex.__init__`.
- Otherwise: NO-GO (legacy fonts or image PDF). Record in RESULTS §5: "Official Hindi PDFs are not text-extractable; Hindi retrieval relies on query rewriting (Task A13)." Do not spend more time here.

- [ ] **Step 3: Commit** whichever RESULTS.md note applies.

---

## Lane B: Hugging Face surfaces (needs the org write token)

### Task B1: Owner-only actions (30 minutes, unblocks everything)

- [ ] Rotate the HF write token (huggingface.co → Settings → Access Tokens → revoke the old one; create a fine-grained token with **write** on the NyayaLabs98 org repos). Hand the new one over as `HF_TOKEN` for Lane B.
- [ ] Rotate the Kaggle API key (kaggle.com → Settings → API → Expire API Token → Create New Token).
- [ ] Delete the Kaggle dataset `jitendrajha98/bhashabench-legal-cache` (it is a redistribution of gated data).
- [ ] With the account that will run the notebooks, accept the terms on `bharatgenai/BhashaBench-Legal`, `google/gemma-3-4b-it`, `meta-llama/Llama-3.2-3B-Instruct`. Add a **read** token as the `HF_TOKEN` secret to each Kaggle notebook.
- [ ] GitHub → repo → About (gear icon): description `Open retrieval stack for current Indian law (BNS/BNSS/BSA): statute DB, reranker, evaluation harness, Qwen2.5-3B reader.`; topics `rag legal-nlp indian-law bns bnss hindi hinglish retrieval evaluation`; remove `qlora`, `fine-tuning`, `qwen`.
- [ ] Tell the executor the rotation date so Task A3 Step 4 can close the HANDOFF line.

### Task B2: Model card for nyaya-3b-v3

**Files:**
- Modify: `docs/cards/nyaya-3b-v3.md`

- [ ] **Step 1: Frontmatter** — replace the existing block with:

```yaml
---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
base_model: Qwen/Qwen2.5-3B-Instruct
base_model_relation: finetune
language: [en, hi]
library_name: transformers
pipeline_tag: text-generation
datasets:
  - NyayaLabs98/nyaya-train-v3
  - NyayaLabs98/nyaya-statute-db
  - NyayaLabs98/nyaya-eval-v0
tags: [legal, india, indian-law, bns, bnss, bsa, retrieval-augmented-generation, qwen2.5, non-commercial]
widget:
  - text: "Police FIR nahi likh rahi, kya karu?"
    example_title: "FIR refusal (Hinglish)"
  - text: "What is the punishment for cheque bounce under the Negotiable Instruments Act?"
    example_title: "Cheque bounce (English)"
model-index:
  - name: nyaya-3b-v3
    results:
      - task: {type: text-generation, name: Legal QA with retrieval}
        dataset: {name: Nyaya-Eval-v1 (409 scored, k=8 RAG), type: NyayaLabs98/nyaya-eval-v0}
        metrics:
          - {type: fact_recall, name: Fact recall, value: 32.9}
          - {type: citation_accuracy, name: Citation accuracy, value: 50.3}
        source: {name: reports/eval_v1_results.json, url: https://github.com/JitendraJha98/nyaya-model/blob/main/reports/eval_v1_results.json}
      - task: {type: multiple-choice, name: Indian legal MCQ}
        dataset: {name: BhashaBench-Legal (1,500-question sample), type: bharatgenai/BhashaBench-Legal}
        metrics:
          - {type: accuracy, name: Accuracy, value: 45.2}
        source: {name: reports/bhashabench_scores.json, url: https://github.com/JitendraJha98/nyaya-model/blob/main/reports/bhashabench_scores.json}
---
```

- [ ] **Step 2: Body edits**

- "Honest status": replace `- **No external benchmark comparison has been run.** No claim is made against any other legal model.` with `- **One external benchmark has been run:** BhashaBench-Legal, 1,500-question sample, exact MCQ scoring: base 47.8%, this model 45.2%, 95% CI on the difference [−6.2, +1.0] → tied. Hindi 38.8% vs English 51.6%. No claim is made against any other legal model.`
- Add the contamination bullet from Task A7 Step 4.
- Replace `- Coverage is limited to the 16 acts in the statute DB.` with `- Coverage is 13 acts plus the Constitution. About a quarter of real citizen questions (rent, property, loans, children) fall outside it; the retriever then returns the nearest section it has. Use the coverage gate in the repository.`
- Replace `**No human evaluation has been passed.**` with `- **No human evaluation has been passed.**` (must contain "No human evaluation" for the test).
- Under "Fine-tuning attempts", add one line: `Note: the v3 config lists NEFTune, but the trainer did not forward it until September 2026; v3 was trained without it.`
- Add a "GGUF" line under Usage once Task B5 is done: `Quantised builds for llama.cpp / Ollama: [NyayaLabs98/nyaya-3b-v3-GGUF](https://huggingface.co/NyayaLabs98/nyaya-3b-v3-GGUF).`

- [ ] **Step 3: Push**

```bash
set HF_TOKEN=hf_...   (PowerShell: $env:HF_TOKEN="hf_...")
python scripts/31_publish_hf.py --version v3 --card-only
```
Expected: `card-only update pushed -> https://huggingface.co/NyayaLabs98/nyaya-3b-v3`; the Hub now shows an "Evaluation results" panel and dataset links.

- [ ] **Step 4: Commit the card** `docs(card): v3 card — BhashaBench, contamination, coverage, model-index`

### Task B3: Organisation card

**Files:**
- Create: `docs/cards/org-README.md`

- [ ] **Step 1: Write it**

```markdown
---
title: NyayaAI
emoji: ⚖️
colorFrom: gray
colorTo: green
sdk: static
pinned: false
---

# NyayaAI

Open tooling for **current Indian law**: the Bharatiya Nyaya Sanhita, Bharatiya Nagarik
Suraksha Sanhita and Bharatiya Sakshya Adhiniyam as in force since 1 July 2024, plus
eleven other acts citizens actually run into, as clean section-level data with the
official IPC↔BNS mappings, a measured retrieval stack, and an evaluation harness that
publishes its own bugs.

- **[nyaya-statute-db](https://huggingface.co/datasets/NyayaLabs98/nyaya-statute-db)** — 13 acts + the Constitution, 2,528 sections, 1,257 official mappings, 70 guidance notes.
- **[nyaya-3b-v3](https://huggingface.co/NyayaLabs98/nyaya-3b-v3)** — the reader model, pre-aligned to the Nyaya prompt format; statistically tied with its base, published for convenience, non-commercial (qwen-research).
- **[nyaya-train-v3](https://huggingface.co/datasets/NyayaLabs98/nyaya-train-v3)** — 6,429 statute-grounded, citation-verified training records.
- Code, retriever, reranker, evaluation: [github.com/JitendraJha98/nyaya-model](https://github.com/JitendraJha98/nyaya-model) (Apache-2.0).

**⚖️ Not legal advice.** Nyaya provides legal information. The practice of law in India is reserved to advocates enrolled under the Advocates Act, 1961. Free legal aid: NALSA / DLSA.
```

- [ ] **Step 2: Publish** — organisation cards live in a Space named `README`:

```bash
python -c "from huggingface_hub import HfApi; api=HfApi(); api.create_repo('NyayaLabs98/README', repo_type='space', space_sdk='static', exist_ok=True); api.upload_file(path_or_fileobj='docs/cards/org-README.md', path_in_repo='README.md', repo_id='NyayaLabs98/README', repo_type='space')"
```
If the Hub rejects the Space creation, use the web UI: organisation page → "Create organization card" and paste the body.

- [ ] **Step 3: Commit** `docs(card): organisation card`

### Task B4: Dataset cards (train-v3 new; eval-v0 notice; statute-db corrections)

**Files:**
- Create: `docs/cards/nyaya-train-v3.md`
- Modify: `docs/cards/nyaya-statute-db.md` (Task A3 edits), and the live `nyaya-eval-v0` card

- [ ] **Step 1: train-v3 card**

```markdown
---
license: apache-2.0
language: [en, hi]
tags: [legal, india, indian-law, bns, bnss, bsa, instruction-tuning, rag, synthetic]
pretty_name: Nyaya-Train-v3 — statute-grounded Indian legal QA (RAFT)
size_categories: [1K<n<10K]
task_categories: [question-answering, text-generation]
---

# Nyaya-Train-v3

6,429 chat-format training records (5,292 train / 280 val / 857 test) for an Indian
legal-information assistant. Every record is a citizen question, the statute sections a
retriever surfaced for it (k=8, plus a deliberate 10% of retrieval-miss demonstrations),
and a teacher answer that cites **only** sections present in that context.

## Provenance
- Statute text: `NyayaLabs98/nyaya-statute-db` (Government of India, public domain).
- Questions: generated from verbatim statute text by Gemma 4 31B-IT (Apache-2.0), then
  regenerated under the inference-time RAG prompt (RAFT).
- Gate: every citation in every answer resolves against the statute DB **and** appears
  in the record's own context; 1,989 of 10,775 teacher outputs failed and were dropped.
- Leakage: 0 records overlap Nyaya-Eval-v0 questions or answers (`reports/v3_dataset_report.json`).
- Splits are grouped by source section, never by row.

## Schema
`{"id", "messages": [{"role": "system"|"user"|"assistant", "content"}], "metadata": {"language", "legal_domain", "task_type", "source_act", "source_sections", "rag": {"context_keys", "is_miss", "question"}, "dataset_version"}}`

## Known result
A LoRA fine-tune of Qwen2.5-3B-Instruct on this data (`NyayaLabs98/nyaya-3b-v3`) is
statistically tied with its base model on Nyaya-Eval-v1. The data is published for
reproducibility and for retriever training (question → gold section pairs), not as a
proven recipe for beating the base model.

## Licence
Apache-2.0 for the generated text. Statutory passages inside prompts are public domain
(Section 52(1)(q), Copyright Act, 1957). Not legal advice.
```
Push: `python -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='docs/cards/nyaya-train-v3.md', path_in_repo='README.md', repo_id='NyayaLabs98/nyaya-train-v3', repo_type='dataset')"`

- [ ] **Step 2: eval-v0 notice** — download the live card (`huggingface-cli download NyayaLabs98/nyaya-eval-v0 README.md --repo-type dataset`), insert after the title: `> **Contaminated as a holdout.** This set is public; do not report numbers on it as held-out. Nyaya-Eval-v1 (`data/eval/nyaya_eval_v1_public.jsonl` in the repository) is the graded successor with a 100% gold ceiling; its private half derives from this file and is therefore also reconstructible.` Push the same way. Save the file as `docs/cards/nyaya-eval-v0.md` so it is versioned.

- [ ] **Step 3: statute-db** — push the Task A3 version of `docs/cards/nyaya-statute-db.md`.

- [ ] **Step 4: Commit** `docs(cards): train-v3 card, eval-v0 contamination notice, statute-db corrections`

### Task B5: GGUF builds and an Ollama Modelfile

- [ ] **Step 1: Quantise with the gguf-my-repo Space** (no GPU, no download): open https://huggingface.co/spaces/ggml-org/gguf-my-repo, sign in as an org member, model `NyayaLabs98/nyaya-3b-v3`, quantisation `Q4_K_M`, output repo `NyayaLabs98/nyaya-3b-v3-GGUF`, tick "private" off. Repeat for `Q8_0` into the same repo.

- [ ] **Step 2: Modelfile** — `docs/cards/Modelfile`:
```
FROM ./nyaya-3b-v3-Q4_K_M.gguf
SYSTEM """You are Nyaya, an Indian legal information model. You provide accurate, plain-language legal guidance for Indian citizens, cite specific sections of current law (BNS/BNSS/BSA and other acts in force), clearly state uncertainty, and recommend consulting a licensed advocate for anything consequential. You provide legal information, not legal advice."""
PARAMETER temperature 0.3
PARAMETER num_ctx 8192
```
Upload with `huggingface-cli upload NyayaLabs98/nyaya-3b-v3-GGUF docs/cards/Modelfile Modelfile`. Add a short README to the GGUF repo: licence (qwen-research, non-commercial), `ollama create nyaya -f Modelfile`, and the line "for accurate answers pair with the retriever; used bare it behaves like Qwen2.5-3B".

- [ ] **Step 3: Verify on Kaggle CPU or locally**: `ollama create nyaya -f Modelfile && ollama run nyaya "Police FIR nahi likh rahi, kya karu?"` answers in Hinglish and cites BNSS.

### Task B6: A CPU Space for the retrieval demo

**Files:**
- Create: `space/app.py`, `space/requirements.txt`, `space/README.md`

- [ ] **Step 1: Space app** — copy `app.py`, then: load the statute DB from the Hub dataset when `data/canonical` is absent (reuse `nyaya.cli._canonical_dir`); default `--no-model` (retrieval only) on the free CPU tier; show `coverage()` result as a banner; keep the disclaimer. `space/requirements.txt`: `gradio>=4.0`, `huggingface_hub>=0.24`, `git+https://github.com/JitendraJha98/nyaya-model`. `space/README.md` frontmatter: `title: Nyaya`, `sdk: gradio`, `sdk_version: "4.44.0"`, `app_file: app.py`, `license: apache-2.0`.

- [ ] **Step 2: Create and push**

```bash
python -c "from huggingface_hub import HfApi; api=HfApi(); api.create_repo('NyayaLabs98/nyaya-demo', repo_type='space', space_sdk='gradio', exist_ok=True); api.upload_folder(repo_id='NyayaLabs98/nyaya-demo', repo_type='space', folder_path='space')"
```
Expected: the Space builds, `Police FIR nahi likh rahi` shows BNSS 173 and the FIR-refusal note in under 2 s.

- [ ] **Step 3: Week 2 upgrade** — once Task C3 publishes `NyayaLabs98/nyaya-reranker-mini-v1`, set it as the default reranker in the Space with depth 20 and measure CPU latency (target ≤ 3 s per query). Week 3: enable the reader with a GGUF via llama-cpp-python if latency allows; otherwise request ZeroGPU.

- [ ] **Step 4: Commit** `feat: Hugging Face Space (retrieval demo)`

---

## Lane C: Kaggle GPU work (owner runs the notebooks)

### Task C1: Base-model shootout under the same retriever

**Files:**
- Create: `scripts/kaggle_shootout.ipynb` (copy `scripts/kaggle_eval_v1.ipynb`, edit cells as below)
- Produces: `outputs/eval-v1/{base-768,nyaya-3b-v3-768,qwen3-4b,gemma-3-4b,llama-3.2-3b,phi-4-mini}/predictions.jsonl`, `reports/eval_v1_results.json` entries, `reports/eval_v1_comparison_*.json`

- [ ] **Step 1: Cell 1 edits** — add `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` before any torch import; replace the pip line with `!pip -q install -r requirements-train.txt`; clone with `--depth 1 --branch main` and print the commit (record it in the results zip).

- [ ] **Step 2: Cell 4 (smoke)** — keep; set `N_MODELS = 3` for session 1, `BUDGET_S = 10 * 3600`. Time the smoke **after** the model is cached: run the smoke twice and use the second timing (HANDOFF §4.4).

- [ ] **Step 3: Session 1 runs** (replace cells 5-7):

```python
COMMON = ["--dense", "--k", "8", "--split", "all", "--max-new-tokens", "768", "--batch-size", "2"]
run([sys.executable, "scripts/26_eval_v1_run.py", "--adapter", "none", "--label", "base-768", *COMMON])
run([sys.executable, "scripts/26_eval_v1_run.py", "--model", "NyayaLabs98/nyaya-3b-v3", "--adapter", "none", "--label", "nyaya-3b-v3-768", *COMMON])
run([sys.executable, "scripts/26_eval_v1_run.py", "--model", "Qwen/Qwen3-4B-Instruct-2507", "--adapter", "none", "--label", "qwen3-4b", *COMMON])
```

- [ ] **Step 4: Session 2 runs** (same notebook, flip a `SESSION = 2` flag):

```python
run([sys.executable, "scripts/26_eval_v1_run.py", "--model", "google/gemma-3-4b-it", "--adapter", "none", "--label", "gemma-3-4b", *COMMON])
run([sys.executable, "scripts/26_eval_v1_run.py", "--model", "meta-llama/Llama-3.2-3B-Instruct", "--adapter", "none", "--label", "llama-3.2-3b", *COMMON])
run([sys.executable, "scripts/26_eval_v1_run.py", "--model", "microsoft/Phi-4-mini-instruct", "--adapter", "none", "--label", "phi-4-mini", *COMMON])
```
Gemma-3-4B in fp16 is ~8.6 GB; if batch 2 OOMs the retry halves it automatically (Task A6 fixed the exception name).

- [ ] **Step 5: Comparisons on CPU after downloading the results zip**

```bash
for b in nyaya-3b-v3-768 qwen3-4b gemma-3-4b llama-3.2-3b phi-4-mini; do python scripts/27_compare_runs.py --a base-768 --b $b; done
```
Commit `outputs/eval-v1/*/predictions.jsonl`, `reports/eval_v1_results.json`, `reports/eval_v1_comparison_*.json`. Add the table to README §4 and RESULTS §1 ("Base-model shootout, same retriever, 768 new tokens"). Whichever model's CI excludes zero **above** base-768 becomes the default reader in `app.py`, `space/app.py` and the README; if that model is Apache-2.0 (Qwen3-4B, Phi-4-mini), say so in the licensing section.

### Task C2: Full BhashaBench-Legal

**Files:**
- Modify: `scripts/kaggle_bhashabench.ipynb`
- Produces: `reports/bhashabench_full.json`, `reports/bhashabench_full_rows.jsonl.gz`

- [ ] **Step 1: Load the gated dataset properly** (replace cells 1-2's parquet glob):

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
from datasets import load_dataset, concatenate_datasets
parts = []
for config in ("English", "Hindi"):          # names as listed on the dataset page; fix if the page differs
    ds = load_dataset("bharatgenai/BhashaBench-Legal", config, split="test", token=os.environ["HF_TOKEN"])
    ds = ds.add_column("lang", [config] * len(ds))
    parts.append(ds)
bench = concatenate_datasets(parts)
SUBSET = 0                                     # all 24,365
print(len(bench), bench.column_names)
```
Delete every reference to `/kaggle/input` and to the cache dataset.

- [ ] **Step 2: One model per session** — cell 3's `run_mcq` unchanged; run `base` in session 1, the C1 winner in session 2. Save per-question rows (`pred`, `gold`, `lang`, and any subject/topic column) gzip-compressed, and the accuracy per language and per topic column into `reports/bhashabench_full.json` with `"n": len(bench)`.

- [ ] **Step 3: Report** — RESULTS §1: replace the 1,500-sample table with the full-set table (keep the sample table below it, labelled "earlier 1,500-question sample"). Submit the results JSON to the BharatGen leaderboard address on their page.

### Task C3: Train a small reranker and a bi-encoder on the project's own pairs

**Files:**
- Create: `scripts/34_build_retriever_pairs.py`, `scripts/kaggle_train_retriever.ipynb`, `tests/test_retriever_pairs.py`
- Produces: `NyayaLabs98/nyaya-reranker-mini-v1`, `NyayaLabs98/nyaya-embed-v1` on the Hub; `reports/retrieval_recall_rerank_mini.json`

- [ ] **Step 1: Pairs script (CPU)** — for each record of `NyayaLabs98/nyaya-train-v3` (download with `snapshot_download`, train split only, skip `metadata.rag.is_miss`): question = the user turn's `Question:` line; positives = `metadata.source_sections`; hard negatives = the top 20 BM25 statute rows that are not gold. Write `data/generated/retriever_pairs.jsonl` as `{"query", "positive_key", "negative_keys": [...]}`. Exclude any query whose gold section appears in `reports/audited_record_ids.json` records' golds? No: exclude any query that is a near-duplicate of an Eval-v1 question (`validators.is_near_duplicate`, threshold 0.85). Print counts. Test: on a 3-record fixture, the output has 3 lines and no negative equals the positive.

- [ ] **Step 2: Notebook** — `pip install -r requirements-train.txt`; build pairs; then:

```python
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import json, random
rows = {f"{r['act_id']}:{r['section'].upper()}": r for r in index.rows}
def passage(k):
    r = rows[k]; return f"{r['act_id'].upper()} Section {r['section']}: {r.get('title','')}\n{(r.get('text') or '')[:1600]}"
examples = []
for line in open("data/generated/retriever_pairs.jsonl", encoding="utf-8"):
    p = json.loads(line)
    examples.append(InputExample(texts=[p["query"], passage(p["positive_key"])], label=1.0))
    for k in random.Random(p["query"]).sample(p["negative_keys"], 3):
        examples.append(InputExample(texts=[p["query"], passage(k)], label=0.0))
ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", num_labels=1, max_length=512)
ce.fit(train_dataloader=DataLoader(examples, shuffle=True, batch_size=32), epochs=1, warmup_steps=200)
ce.save("outputs/nyaya-reranker-mini-v1")
```
Bi-encoder: `SentenceTransformer("intfloat/multilingual-e5-base")` with `MultipleNegativesRankingLoss`, texts prefixed `query: ` / `passage: `, 1 epoch, batch 32; save `outputs/nyaya-embed-v1`.

- [ ] **Step 3: Evaluate on the never-audited slice**

```bash
python scripts/15_retrieval_recall.py --k 1 3 8 --rerank outputs/nyaya-reranker-mini-v1 --rerank-depth 20 --skip-phrase-coverage
python scripts/15_retrieval_recall.py --k 1 3 8 --dense outputs/nyaya-embed-v1 --skip-phrase-coverage
```
Compare `full_hit_never_audited` against 0.4576 (BM25) and 0.5847 (bge-reranker-v2-m3) at k=1. Publish only if the mini reranker beats BM25 at k=1 on that slice; report its CPU latency at depth 20.

- [ ] **Step 4: Publish and wire** — `ce.push_to_hub("NyayaLabs98/nyaya-reranker-mini-v1")`; card with the numbers; `nyaya/rerank.py` `DEFAULT_MODEL` stays bge for eval, and `app.py`/Space use the mini model when no GPU.

### Task C4 (optional, week 3-4): v7 by teacher distillation

Only with a teacher API key and only if the owner wants one more fine-tune. Recipe: point `scripts/20_generate_raft.py` at the API (`TEACHER_BASE_URL`, `NYAYA_TEACHER_API_KEY`, `teacher.model` in `configs/generation.yaml`), regenerate answers for the existing v3 RAFT prompts with the instruction "150–250 words, cover every specific the provisions give"; keep the citation gate; build `configs/train_v7.yaml` from `train_v5.yaml` with `max_examples: 1200`; train on Kaggle with `kaggle_train_v5.ipynb` (rename to `kaggle_train_v7.ipynb`); evaluate with `26` at 768 tokens; compare with `27` against `base-768`. Publish as `nyaya-3b-v7` only if the CI excludes zero on Eval-v1 **and** on the Task D2 holdout.

---

## Lane D: coverage and evaluation data (CPU + one human reviewer)

### Task D1: Add the acts real questions need

**Files:**
- Modify: `configs/acts.yaml`, `data/canonical/*.jsonl` (new files), `reports/corpus_extraction_report.json`, `docs/cards/nyaya-statute-db.md`

- [ ] **Step 1: Per act, in this order** (frequency in the 269 citizen questions): Transfer of Property Act 1882; Indian Contract Act 1872; Protection of Children from Sexual Offences Act 2012; Juvenile Justice (Care and Protection of Children) Act 2015; Hindu Succession Act 1956; Indian Succession Act 1925; Model Tenancy Act 2021; Guardians and Wards Act 1890; Dowry Prohibition Act 1961; Maintenance and Welfare of Parents and Senior Citizens Act 2007; Payment of Gratuity Act 1972; Maternity Benefit Act 1961; Limitation Act 1963; Legal Services Authorities Act 1987.

- [ ] **Step 2: Recipe for one act** (same as the 13 existing entries):
  1. On indiacode.nic.in search the act; copy the handle page URL (`/handle/123456789/<id>`) and the English PDF bitstream URL; read `expected_sections` from the Arrangement of Sections.
  2. Append to `configs/acts.yaml`:
     ```yaml
       - act_id: tpa_1882
         act_name: "Transfer of Property Act, 1882"
         enabled: true
         url: "<bitstream pdf url>"
         handle_url: "<handle url>"
         expected_sections: 137
         effective_date: "1882-07-01"
     ```
     and add the alias family to `src/nyaya/validators.py` `ACT_ALIASES` (`"tpa": ["tpa", "transfer of property act", "संपत्ति अंतरण अधिनियम"]`) and `_ACT_ID_FAMILY` (`"tpa": "tpa"`), plus `retrieval._FAMILY_TO_ACT_ID`.
  3. `python scripts/03_build_corpus.py --act tpa_1882`
  4. Gate: `clean_fraction ≥ 0.98` in `reports/corpus_extraction_report.json`; open five random rows and compare with the PDF.
  5. `python -m pytest -q tests/test_corpus.py tests/test_validators_db.py`; commit `data: add Transfer of Property Act, 1882 (137 sections, 100% clean)`.

- [ ] **Step 3: After every act** re-run the citizen-question probe and record the zero-statute-hit count and the out-of-coverage count in `reports/coverage_probe.json` (write `scripts/35_coverage_probe.py`: the bucket regexes from the audit, counts per bucket, count of questions with `coverage()["covered"] is False`).

- [ ] **Step 4: Card and README** — update the contents table in `docs/cards/nyaya-statute-db.md` and push (Task B4 Step 3 command); update the coverage sentence everywhere.

### Task D2: A true holdout from the citizen questions

**Files:**
- Create: `scripts/36_select_holdout.py`, `data/eval/holdout_v1_draft.jsonl` (gitignored), `docs/HOLDOUT_REVIEW.md`
- Modify: `.gitignore`, `scripts/26_eval_v1_run.py` `EVAL_FILES` (add `"holdout": ROOT / "data" / "eval" / "nyaya_holdout_v1_private.jsonl"`)

- [ ] **Step 1: Select 180 questions** stratified: 15 per bucket for the 11 buckets in the audit probe, the rest from the uncategorised set; keep the original Hindi/Hinglish text; write records in the `EvalRecord` shape with empty `expected_answer`, `required_facts`, `forbidden_facts`, and `"source": "citizen_questions.txt"`.

- [ ] **Step 2: Reviewer brief** (`docs/HOLDOUT_REVIEW.md`): a law student or advocate fills `expected_answer` (3–5 sentences), `required_facts` (2–4 short quotable phrases, at least one `Section <n> <Act>`), `forbidden_facts` (repealed provisions "as current law"); every fact must survive `nyaya.scoring.lint_fact` (run `python -c "..."` on the file and fix every flagged fact); mark questions outside all indexed acts with `"legal_domain": "out_of_coverage"` and `required_facts: ["not covered"]`.

- [ ] **Step 3: Gold ceiling check** as in `scripts/25`: score every `expected_answer` against its own facts; require 100%.

- [ ] **Step 4: Run** `26 --split holdout` for `base-768`, the C1 winner, and v7 if it exists; `27` between them; report in RESULTS §1 as "Holdout-v1 (180 real citizen questions, never published)". This is also the human-evaluation gate: the reviewer additionally grades 100 answers of the best system for correctness / completeness / safety on a 3-point scale; report the distribution.

### Task D3: Populate `replaces`, `tags` and `punishment_summary`

**Files:**
- Create: `scripts/37_enrich_statute_db.py`, `tests/test_enrich.py`
- Modify: `data/canonical/bns_2023.jsonl`, `bnss_2023.jsonl`, `bsa_2023.jsonl` (fields only), `docs/cards/nyaya-statute-db.md`

- [ ] **Step 1: Failing test**

```python
# tests/test_enrich.py
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("enrich", ROOT / "scripts" / "37_enrich_statute_db.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_replaces_comes_from_the_official_mapping():
    m = _mod()
    mappings = [{"old_act": "IPC", "old_section": "302", "new_act": "BNS", "new_section": "103", "note": None}]
    row = {"act_id": "bns_2023", "section": "103", "text": "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine."}
    out = m.enrich(row, mappings)
    assert out["replaces"] == ["IPC 302"]
    assert out["punishment_summary"] == "death or imprisonment for life, and shall also be liable to fine"


def test_rows_without_a_penalty_clause_keep_null():
    m = _mod()
    row = {"act_id": "bns_2023", "section": "2", "text": "In this Sanhita, unless the context otherwise requires"}
    assert m.enrich(row, [])["punishment_summary"] is None
```

- [ ] **Step 2: Implement**

```python
# scripts/37_enrich_statute_db.py
"""Fill the reserved statute-DB fields from data the repo already has:
replaces <- law_mappings.jsonl; punishment_summary <- the first penalty clause."""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "data" / "canonical"
NEW_ACT_TO_ID = {"BNS": "bns_2023", "BNSS": "bnss_2023", "BSA": "bsa_2023"}
_PENALTY = re.compile(r"shall be punish(?:ed|able) with ([^.;]{5,160})", re.IGNORECASE)


def enrich(row: dict, mappings: list[dict]) -> dict:
    out = dict(row)
    olds = [f"{m['old_act']} {m['old_section']}" for m in mappings
            if NEW_ACT_TO_ID.get(m["new_act"]) == row["act_id"] and m["new_section"].upper() == row["section"].upper()]
    out["replaces"] = olds or None
    m = _PENALTY.search(row.get("text") or "")
    out["punishment_summary"] = m.group(1).strip() if m else None
    return out


def main() -> None:
    mappings = [json.loads(l) for l in (CANON / "law_mappings.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for act_id in NEW_ACT_TO_ID.values():
        path = CANON / f"{act_id}.jsonl"
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [enrich(r, mappings) for r in rows]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(act_id, sum(1 for r in rows if r["replaces"]), "rows with replaces;",
              sum(1 for r in rows if r["punishment_summary"]), "with punishment_summary")


if __name__ == "__main__":
    main()
```
Note `replaces` becomes a list of strings (the schema in `schemas.py` says dict; update the dataclass comment to `list[str] | None` and the card example to `"replaces": ["IPC 302"]`).

- [ ] **Step 3: Run, verify, re-upload** — `python scripts/37_enrich_statute_db.py`; run the suite (the retrieval tests read these files); spot-check BNS 103 → `["IPC 302"]`, BNS 318 → the IPC 415–420 range; `huggingface-cli upload NyayaLabs98/nyaya-statute-db data/canonical . --repo-type dataset --include "*.jsonl"`; restore the populated example in the card and remove the "reserved and currently empty" sentence.

- [ ] **Step 4: Commit** `data: populate replaces and punishment_summary from official mappings and penalty clauses`

---

## Credentials and what each unlocks

| Credential | Who provides | Unlocks | Needed by |
|---|---|---|---|
| HF **write** token, fine-grained, scoped to NyayaLabs98 repos | owner, after rotating the leaked one | Tasks B2–B6, D1 Step 4, D3 Step 3, C3 Step 4 (cards, org page, GGUF repo, Space, dataset re-uploads) | Day 1 |
| HF **read** token as Kaggle secret `HF_TOKEN`, on an account that accepted the gated terms | owner | Tasks C1 (Gemma, Llama), C2 (BhashaBench) | Day 2 |
| Kaggle: nothing programmatic. Owner runs notebooks in the browser. Optional `kaggle.json` if the executor should push notebooks with `kaggle kernels push` | owner (optional) | convenience only | — |
| GitHub: nothing programmatic. Executor commits locally; owner pushes and sets About/topics in the UI. Optional PAT (repo scope) if the executor should set About/topics via API | owner (optional) | convenience only | — |
| Teacher API key (Anthropic, OpenAI or Gemini; any OpenAI-compatible chat endpoint) | owner | Task C4 (v7 distillation) and the strong-model judge on the holdout | Week 3, only if C4 is wanted |

Nothing else. The BhashaBench leaderboard is an email submission; India Code and NCRB downloads need no key.

## Self-review against the audit

- Part III rows: HF model card → B2; org page → B3; train-v3 / eval-v0 / statute-db cards → B4, A3; GGUF → B5; GitHub About → B1; README → A3, A11; pyproject → A3, A10; RELEASE_PLAN / HANDOFF → A3, B1; notebooks' private inputs → B1, C2; k8s / namespace / registry → A1; personal path → A2; docs/superpowers → A2; CI / CITATION / CHANGELOG / duplicate 24 / naming → A9, A2; data READMEs / ROADMAP → A3; app.py explanation → A11, A12.
- Part IV items 5–10: coverage gate → A12, D1; Hindi → A13, A14; guidance floor → A12; learned retrieval → C3; shippable reranker → C3, B6; holdout → D2.
- Part V items 15–21: NEFTune → A5; floors / OOM / dtype → A6; eval leak → A7; v5/v6 gate bypass and reproducibility → recorded in A3 (RESULTS) and CHANGELOG, data re-upload in C4 if v7 happens; card divergence → A7; smaller defects → A8 (26/16), A2 (numbering), A5 (test paths). Not scheduled: scripts/12 byte-cap double count, scripts/15 zero-division, scripts/28 shallow-clone, scripts/29 metadata — these do not affect any public surface; file them as GitHub issues at the end of week 1.
- Part VI tracks A–D and Part VIII weeks 1–4 map onto lanes A–D and the day plan above.
