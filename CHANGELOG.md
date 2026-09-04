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

## 0.2.0 — 2026-08-07

- Eval-v1 (gold ceiling 100%), cross-encoder reranker (+12.7 points at k=1 on
  never-audited records), BhashaBench-Legal sample run, v5/v6 negative results,
  model card corrected to the qwen-research licence.

## 0.1.0 — 2026-07-20

- Statute DB (13 acts + Constitution), Nyaya-Eval-v0, v1–v4 fine-tunes,
  `NyayaLabs98/nyaya-3b-v3` published.
