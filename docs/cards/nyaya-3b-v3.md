---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
base_model: Qwen/Qwen2.5-3B-Instruct
base_model_relation: finetune
language:
- en
- hi
library_name: transformers
pipeline_tag: text-generation
datasets:
- NyayaLabs98/nyaya-train-v3
- NyayaLabs98/nyaya-statute-db
- NyayaLabs98/nyaya-eval-v0
tags:
- legal
- india
- indian-law
- bns
- bnss
- bsa
- retrieval-augmented-generation
- qwen2.5
- non-commercial
widget:
- text: "Police FIR nahi likh rahi, kya karu?"
  example_title: "FIR refusal (Hinglish)"
- text: "What is the punishment for cheque bounce under the Negotiable Instruments Act?"
  example_title: "Cheque bounce (English)"
model-index:
- name: nyaya-3b-v3
  results:
  - task:
      type: text-generation
      name: Legal QA with retrieval (Nyaya-Eval-v1)
    dataset:
      type: NyayaLabs98/nyaya-eval-v0
      name: Nyaya-Eval-v1 (409 scored, k=8 RAG; base model scores 34.3 / 52.8)
    metrics:
    - type: fact_recall
      name: Fact recall
      value: 32.9
    - type: citation_accuracy
      name: Citation accuracy
      value: 50.3
    source:
      name: reports/eval_v1_results.json
      url: https://github.com/JitendraJha98/nyaya-model/blob/main/reports/eval_v1_results.json
  - task:
      type: multiple-choice
      name: Indian legal MCQ
    dataset:
      type: bharatgenai/BhashaBench-Legal
      name: BhashaBench-Legal (1,500-question sample; base model scores 47.8)
    metrics:
    - type: accuracy
      name: Accuracy
      value: 45.2
    source:
      name: reports/bhashabench_scores.json
      url: https://github.com/JitendraJha98/nyaya-model/blob/main/reports/bhashabench_scores.json
---

# Nyaya-3B — the model component of the Nyaya legal guidance system

> **⚖️ Not legal advice.** Nyaya provides legal *information*. The practice of law
> in India is reserved to advocates enrolled under the Advocates Act, 1961.
> Consult a licensed advocate for anything consequential. Free legal aid is
> available through NALSA / DLSA (Legal Services Authorities Act, 1987).

> **📋 Non-commercial licence.** The base model `Qwen/Qwen2.5-3B-Instruct` is
> released under the **Qwen Research License** — *not* Apache-2.0. The 3B is one
> of the Qwen2.5 sizes carrying the restricted licence, and these merged weights
> inherit it: **research / non-commercial use only.**

## What this is, stated plainly

This is the model that reads retrieved statute sections and writes a cited
answer. **On the project's own benchmark its accuracy is statistically tied
with the base model it was fine-tuned from.** The system's measured gains come
from **retrieval**, not from these weights.

It is published because a single download that works — model, prompt format and
citation style already aligned with the Nyaya retriever — is more useful than
assembling the pieces yourself. It is **not** published as an improvement over
`Qwen/Qwen2.5-3B-Instruct`, because it is not one.

**Use it with [`NyayaLabs98/nyaya-statute-db`](https://huggingface.co/datasets/NyayaLabs98/nyaya-statute-db)
and the retriever from the [repository](https://github.com/JitendraJha98/nyaya-model).**
Used bare, without retrieval, it behaves close to the base model.

## Evaluation

Nyaya-Eval-v1, 413 gradeable questions (409 scored; 4 safety rows graded
separately), paired comparison, 10,000-round bootstrap. Same retriever, same
questions — only the weights differ.

| | fact recall | citation accuracy |
|---|---|---|
| `Qwen2.5-3B-Instruct` + RAG | **34.3%** | **52.8%** |
| **Nyaya-3B-v3** + RAG | 32.9% | 50.3% |

95% CI on the paired difference **spans zero** → statistically indistinguishable.

Re-measured on 2026-09-04 with 768 new tokens and the tokenizer/config files
corrected (the Hub copy was written by transformers 5.12), same retriever as the
base run: 33.8% vs 35.8% fact recall, CI [−5.2, +1.2] (tied); citation recall
48.6% vs 55.6%, CI [−13.9, −0.4] (worse). Under that same setup
`Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0) reaches **50.6%** fact recall and
72.2% citation accuracy, so the repository now uses it as the default reader.
If you want the best small reader for Indian statutes, use that model with the
Nyaya retriever; use this one only where the Nyaya prompt alignment matters.

### Where the accuracy actually comes from

| retrieval outcome (base model) | n | fact recall |
|---|---|---|
| every gold statute retrieved | 94 | **63.2%** |
| a gold statute missed | 51 | **20.3%** |

That 43-point gap is why this project's effort moved to retrieval
(`reports/eval_v1_retrieval_outcome.json`). Adding a
cross-encoder reranker put the correct section in the **top result** for 58.5%
of questions, up from 45.8% — validated on records never used for tuning.
A bi-encoder fine-tuned on the project's own pairs
([`nyaya-embed-v1`](https://huggingface.co/NyayaLabs98/nyaya-embed-v1)) lifts this very
reader's fact recall from 35.8% to 39.7% (paired 95% CI [+0.9, +7.0]) with no change to
the weights (`reports/eval_v1_comparison_base-768-embed-v1.json`).

### Fine-tuning attempts, for the record

| | fact recall | vs base |
|---|---|---|
| base | 34.3% | — |
| v3 (this model, RAFT) | 32.9% | tied |
| v5 (grounded citation data) | 24.0% | **worse**, CI [−13.5, −7.2] |
| v6 (v5 + answer-style fix) | 23.4% | **worse**, CI [−14.0, −7.8] |

None beat base. v5 and v6 regressed because training on short templated targets
shortened the answers (173 → 90 → 57 words), and shorter answers carry fewer of
the facts being scored. Two notes for the record: v5 and v6 ran under a retriever
with an extended vocabulary table, so "same retriever" is exact only for base vs
v3; and the v3 config lists NEFTune, but the trainer did not forward it until
September 2026 — v3 was trained without it.

## Honest status

- **One external benchmark has been run:** BhashaBench-Legal, 1,500-question
  sample, exact MCQ scoring — base 47.8%, this model 45.2%, 95% CI on the
  difference [−6.2, +1.0] → tied. Hindi 38.8% vs English 51.6%. No claim is made
  against any other legal model.
- **No human evaluation has been passed.**
- The project's earlier benchmark (Eval-v0) scored its own gold answers at
  10.7%, so **any accuracy figure derived from it is meaningless** — including
  numbers previously shown on this card.
- `nyaya-eval-v0` is public, so it is **contaminated** as a held-out benchmark;
  Eval-v1's private half derives from it and is reconstructible.
- Coverage is 27 acts plus the Constitution. Of 269 real citizen questions,
  3% fall in domains with no act at all; the retriever then returns the nearest
  section it has. Use the coverage gate in the repository.

## Usage

Quantised builds for llama.cpp / Ollama (Q4_K_M 1.9 GB, Q8_0 3.3 GB, with a
Modelfile carrying the Nyaya system prompt):
[`NyayaLabs98/nyaya-3b-v3-GGUF`](https://huggingface.co/NyayaLabs98/nyaya-3b-v3-GGUF).
Retrieval-only demo: [`NyayaLabs98/nyaya-demo`](https://huggingface.co/spaces/NyayaLabs98/nyaya-demo).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("NyayaLabs98/nyaya-3b-v3")
model = AutoModelForCausalLM.from_pretrained("NyayaLabs98/nyaya-3b-v3", device_map="auto")

system = (
    "You are Nyaya, an Indian legal information model. You provide accurate, "
    "plain-language legal guidance for Indian citizens, cite specific sections of "
    "current law (BNS/BNSS/BSA and other acts in force), clearly state uncertainty, "
    "and recommend consulting a licensed advocate for anything consequential. "
    "You provide legal information, not legal advice."
)
# Prepend retrieved statute passages to the user turn — that is where the
# accuracy comes from. See the repo's nyaya.retrieval.build_rag_prompt.
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": "Police FIR nahi likh rahi, kya karu?"},
]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=512)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

## Licence & attribution

**Research / non-commercial only.** Merged LoRA derivative of
[`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
(`qwen-research`). The training/eval **code** is Apache-2.0; that licence does
not extend to these weights.

Statutory text is Government of India material, public domain under Section
52(1)(q) of the Copyright Act, 1957.

Code, evaluation harness, and the full record of what did and did not work:
[github.com/JitendraJha98/nyaya-model](https://github.com/JitendraJha98/nyaya-model)
