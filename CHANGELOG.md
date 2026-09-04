# Changelog

## 0.3.0 — 2026-09 (release hardening)

- Removed the manifests for the decommissioned training cluster; every GPU step is
  now documented for Kaggle T4. A guard test keeps references from coming back.
- Eval-v1 record completed: v6 scored into `reports/eval_v1_results.json`, an
  8-question smoke entry removed, and one paired-comparison file per candidate
  (`reports/eval_v1_comparison_<label>.json`).
- Trainer forwards `neftune_noise_alpha` and warns on every dropped config key.
  v3, v4 and v5 had trained **without** NEFTune although their configs listed it.
- Dependency floors raised to the APIs the code calls (transformers ≥ 4.56,
  TRL ≥ 0.12, torch ≥ 2.5); core install no longer pulls the GPU stack; extras
  `train`, `dense`, `data`, `demo`, `dev`; `requirements.lock` added.
- Publish script uploads the maintained card from `docs/cards/` and allow-lists
  the eval upload so the private Eval-v1 split can never be pushed.
- Eval scripts: metrics follow `--out-dir`, adapter runs get a real label, the
  recorded model id is the one loaded, chat templates without a system role work.
- Docs: act count corrected to 13 + the Constitution; v5 fact recall 24.0%;
  BhashaBench-Legal status stated; stale plan statuses closed.
- `nyaya ask` command-line entry point over the standard-library retriever.
- Retrieval: relevance floor on guidance notes; calibrated coverage gate
  (`StatuteIndex.coverage`); Hindi/Hinglish query rewriting (`nyaya.rewrite`,
  `--rewrite`) that cuts zero-hit Devanagari questions from 19/53 to 3/53.
- Statute DB: `replaces` (from the official mapping tables) and
  `punishment_summary` populated; re-uploaded to the Hub.
- Static browser-side demo (`space-static/`, a JavaScript port of the retriever,
  parity-checked on 389 questions) at huggingface.co/spaces/NyayaLabs98/nyaya-demo;
  GGUF builds (Q4_K_M, Q8_0, Ollama Modelfile) at NyayaLabs98/nyaya-3b-v3-GGUF.
- Holdout-v1 tooling: stratified selection of 180 unpublished citizen questions
  and a reviewer brief; `--split holdout` in the Eval-v1 runner.
- Recomputed and corrected: fact recall by retrieval outcome (63.2% / 20.3%),
  the v5/v6 retriever-version caveat, coverage probe over real questions.
- Statute DB grown from 14 to 27 acts + the Constitution (3,736 sections) through
  the India Code DSpace API (`nyaya.indiacode`, `scripts/42`); real citizen
  questions with no act in the database fell from 25% to 3%.
- Retriever models trained on the project's own pairs and published:
  `NyayaLabs98/nyaya-embed-v1` (default dense stage; never-audited recall@8
  88.1% vs 81.4% for BM25; +3.9 points of reader fact recall on the 3B reader)
  and `NyayaLabs98/nyaya-reranker-mini-v1` (118M, +5.9 at k=1).
- Default reader is `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0): 50.6% fact recall
  vs 35.8% for Qwen2.5-3B under the same retriever; the default configuration
  scores 52.0% / 77.1% citation accuracy, +16.2 points over the system as first
  published (paired 95% CI [+12.7, +19.9]). Llama-3.2-3B ties the 3B base;
  Gemma-3-4B is unscoreable in fp16 on a T4.
- External check: BhashaBench-Legal, 3,000 paired questions, letter-logit scoring —
  Qwen3-4B 52.5% vs 49.6% (+3.0, CI [+1.0, +5.0]); the generation-parsed pass is
  kept as a documented failure.
- Teacher test (C4) without a paid API: a served Qwen2.5-14B-AWQ scored 45.0%,
  above the 3B reader but below Qwen3-4B, so no v7 was trained; its 1,456
  citation-verified answers are published as `NyayaLabs98/nyaya-train-v7-raft`.
- Eval runner: `--dense-model` with per-model vector caches, `--endpoint` for any
  OpenAI-compatible server, unlabelled runs named after the model.
- `NyayaLabs98/nyaya-3b-v3` Hub metadata made loadable by transformers 4.x
  (tokenizer config and `rope_theta`); repository history rewritten to remove a
  collaborator's employer address; work merged to `main`.

## 0.2.0 — 2026-08-07

- Eval-v1 (gold ceiling 100%), cross-encoder reranker (+12.7 points at k=1 on
  never-audited records), BhashaBench-Legal sample run, v5/v6 negative results,
  model card corrected to the qwen-research licence.

## 0.1.0 — 2026-07-20

- Statute DB (13 acts + Constitution), Nyaya-Eval-v0, v1–v4 fine-tunes,
  `NyayaLabs98/nyaya-3b-v3` published.
