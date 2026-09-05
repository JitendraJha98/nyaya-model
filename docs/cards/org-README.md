---
title: Nyaya
emoji: ⚖️
colorFrom: gray
colorTo: green
sdk: static
pinned: false
---

# Nyaya

Open tooling for **current Indian law**: the Bharatiya Nyaya Sanhita, Bharatiya
Nagarik Suraksha Sanhita and Bharatiya Sakshya Adhiniyam as in force since
1 July 2024, plus the Constitution and twenty-four other acts citizens actually run into,
as clean section-level data with the official IPC↔BNS mappings, a measured
retrieval stack, and an evaluation harness that publishes its own bugs.

**Try it in the browser, nothing to install:** [nyaya-demo](https://huggingface.co/spaces/NyayaLabs98/nyaya-demo)
— the retriever runs client-side over the statute database; type a question in English,
Hindi or Hinglish and see the sections of current law it resolves to.

- **[nyaya-statute-db](https://huggingface.co/datasets/NyayaLabs98/nyaya-statute-db)** —
  27 acts + the Constitution, 3,736 sections, 1,257 official mappings, 70 procedural guidance notes.
- **[nyaya-embed-v1](https://huggingface.co/NyayaLabs98/nyaya-embed-v1)** — bi-encoder
  fine-tuned on the project's question–section pairs; lifts the base reader's fact recall
  by +3.9 points (95% CI [+0.9, +7.0]), the only end-to-end gain with an interval clear of
  zero. MIT.
- **[nyaya-reranker-mini-v1](https://huggingface.co/NyayaLabs98/nyaya-reranker-mini-v1)** —
  118M cross-encoder, +5.9 points at k=1 over BM25 on never-tuned records. Apache-2.0.
- **[nyaya-3b-v3](https://huggingface.co/NyayaLabs98/nyaya-3b-v3)** — the project's fine-tuned
  reader, kept for the record: statistically tied with its base model; non-commercial
  (qwen-research). GGUF builds for llama.cpp / Ollama: [nyaya-3b-v3-GGUF](https://huggingface.co/NyayaLabs98/nyaya-3b-v3-GGUF).
  The system's default reader is `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0): 50.6% fact
  recall against 35.8% for the 3B base under the same retriever, paired CI [+11.4, +18.3].
- **[nyaya-train-v3](https://huggingface.co/datasets/NyayaLabs98/nyaya-train-v3)** —
  6,429 statute-grounded, citation-verified training records.
- **[nyaya-train-v7-raft](https://huggingface.co/datasets/NyayaLabs98/nyaya-train-v7-raft)** —
  1,456 citation-verified answers from a served 14B teacher; published, not trained on
  (the teacher scores below the default reader).
- Code, retriever, reranker, evaluation:
  [github.com/JitendraJha98/nyaya-model](https://github.com/JitendraJha98/nyaya-model) (Apache-2.0).

**What we found:** five fine-tunes of a 3B model never beat the base model once
the benchmark could tell the difference. Retrieval did: with the right section
in context the base model reaches 63% fact recall; without it, 20%. The work is
in finding the right law.

**Where it stands (2026-09-04):** the default configuration — `Qwen/Qwen3-4B-Instruct-2507`
reading sections found by BM25 + `nyaya-embed-v1` — scores **52.0% fact recall and 77.1%
citation accuracy** on Nyaya-Eval-v1, +16.2 points over the system as first published
(paired 95% CI [+12.7, +19.9]). Every prediction behind every number is committed in the
repository and re-scorable on a CPU. On the external BhashaBench-Legal set (3,000 paired
questions, no retrieval) the same reader scores 52.5% against 49.6% for the 3B base,
+3.0 points (CI [+1.0, +5.0]), most of it on Hindi.

**⚖️ Not legal advice.** Nyaya provides legal information. The practice of law
in India is reserved to advocates enrolled under the Advocates Act, 1961. Free
legal aid: NALSA / DLSA.
