---
title: NyayaAI
emoji: ⚖️
colorFrom: gray
colorTo: green
sdk: static
pinned: false
---

# NyayaAI

Open tooling for **current Indian law**: the Bharatiya Nyaya Sanhita, Bharatiya
Nagarik Suraksha Sanhita and Bharatiya Sakshya Adhiniyam as in force since
1 July 2024, plus the Constitution and twenty-four other acts citizens actually run into,
as clean section-level data with the official IPC↔BNS mappings, a measured
retrieval stack, and an evaluation harness that publishes its own bugs.

- **[nyaya-statute-db](https://huggingface.co/datasets/NyayaLabs98/nyaya-statute-db)** —
  27 acts + the Constitution, 3,736 sections, 1,257 official mappings, 70 procedural guidance notes.
- **[nyaya-3b-v3](https://huggingface.co/NyayaLabs98/nyaya-3b-v3)** — the reader model,
  pre-aligned to the Nyaya prompt format. Statistically tied with its base model;
  published for convenience; non-commercial (qwen-research).
- **[nyaya-embed-v1](https://huggingface.co/NyayaLabs98/nyaya-embed-v1)** — bi-encoder
  fine-tuned on the project's question–section pairs; lifts the base reader's fact recall
  by +3.9 points (95% CI [+0.9, +7.0]), the only end-to-end gain with an interval clear of
  zero. MIT.
- **[nyaya-reranker-mini-v1](https://huggingface.co/NyayaLabs98/nyaya-reranker-mini-v1)** —
  118M cross-encoder, +5.9 points at k=1 over BM25 on never-tuned records. Apache-2.0.
- **[nyaya-train-v3](https://huggingface.co/datasets/NyayaLabs98/nyaya-train-v3)** —
  6,429 statute-grounded, citation-verified training records.
- Code, retriever, reranker, evaluation:
  [github.com/JitendraJha98/nyaya-model](https://github.com/JitendraJha98/nyaya-model) (Apache-2.0).

**What we found:** five fine-tunes of a 3B model never beat the base model once
the benchmark could tell the difference. Retrieval did: with the right section
in context the base model reaches 63% fact recall; without it, 20%. The work is
in finding the right law.

**⚖️ Not legal advice.** Nyaya provides legal information. The practice of law
in India is reserved to advocates enrolled under the Advocates Act, 1961. Free
legal aid: NALSA / DLSA.
