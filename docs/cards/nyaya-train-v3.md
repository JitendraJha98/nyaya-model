---
license: apache-2.0
language:
- en
- hi
tags:
- legal
- india
- indian-law
- bns
- bnss
- bsa
- instruction-tuning
- rag
- synthetic
pretty_name: Nyaya-Train-v3 — statute-grounded Indian legal QA (RAFT)
size_categories:
- 1K<n<10K
task_categories:
- question-answering
- text-generation
---

# Nyaya-Train-v3

6,429 chat-format training records (5,292 train / 280 val / 857 test) for an
Indian legal-information assistant. Every record is a citizen question, the
statute sections a retriever surfaced for it (k=8, plus a deliberate 10% of
retrieval-miss demonstrations), and a teacher answer that cites **only**
sections present in that context.

## Provenance

- **Statute text:** [`NyayaLabs98/nyaya-statute-db`](https://huggingface.co/datasets/NyayaLabs98/nyaya-statute-db)
  (Government of India material, public domain under Section 52(1)(q) of the
  Copyright Act, 1957).
- **Questions and answers:** generated from verbatim statute text by Gemma 4
  31B-IT (Apache-2.0), then regenerated under the inference-time RAG prompt
  (RAFT). The teacher never cited from memory: its prompt held the statute text
  and nothing else.
- **Gate:** every citation in every answer resolves against the statute DB
  **and** appears in the record's own context. 1,989 of 10,775 teacher outputs
  failed that gate and were dropped.
- **Leakage:** 0 records overlap a Nyaya-Eval-v0 question or answer
  (`reports/v3_dataset_report.json` in the repository).
- **Splits** are grouped by source section, never by row, so a section's
  questions never straddle train and test.

## Schema

```json
{
  "id": "raft_gen_000123_ab12cd34_01",
  "messages": [
    {"role": "system", "content": "You are Nyaya, an Indian legal information model. ..."},
    {"role": "user", "content": "Relevant provisions of current Indian law ... Question: ..."},
    {"role": "assistant", "content": "Under Section 173 of the Bharatiya Nagarik Suraksha Sanhita, 2023 ..."}
  ],
  "metadata": {
    "language": "hinglish",
    "legal_domain": "bnss_2023",
    "task_type": "grounded_qa",
    "source_act": "bnss_2023",
    "source_sections": ["bnss_2023:173"],
    "rag": {"context_keys": ["bnss_2023:173", "bnss_2023:175", "..."], "is_miss": false, "question": "..."},
    "dataset_version": "nyaya_instruct_v3"
  }
}
```

## Known result — read before training on this

A LoRA fine-tune of Qwen2.5-3B-Instruct on this data
([`NyayaLabs98/nyaya-3b-v3`](https://huggingface.co/NyayaLabs98/nyaya-3b-v3))
is **statistically tied with its base model** on Nyaya-Eval-v1 (fact recall
32.9% vs 34.3%, 95% CI on the paired difference spans zero). The data is
published for reproducibility and for **retriever training** — each record is a
citizen question paired with its gold statute sections, which is a contrastive
training set — not as a proven recipe for beating the base model.

## Licence

Apache-2.0 for the generated text. Statutory passages inside prompts are public
domain. **Not legal advice**: the answers are training targets for a legal
*information* assistant, not advice; consult a licensed advocate (Advocates Act,
1961). Built with
[github.com/JitendraJha98/nyaya-model](https://github.com/JitendraJha98/nyaya-model).
