---
license: cc-by-4.0
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
- evaluation
- benchmark
pretty_name: Nyaya-Eval-v0 — 500 curated Indian legal questions (public; contaminated as a holdout)
size_categories:
- n<1K
task_categories:
- question-answering
---

# Nyaya-Eval-v0 — 500 manually curated questions

> **Contaminated as a holdout.** This set is public, so numbers reported on it
> cannot be treated as held-out. Its own strict metric also turned out to score
> the gold answers at 10.7%, so accuracy figures computed with that metric are
> not meaningful. **Nyaya-Eval-v1** (`data/eval/nyaya_eval_v1_public.jsonl` in
> the [repository](https://github.com/JitendraJha98/nyaya-model)) is the graded
> successor with a 100% gold ceiling; its private half derives from this file
> and is therefore also reconstructible. Use this set to reproduce the project's
> history, not to rank models.

500 questions an Indian citizen might ask, each with an expected answer, the
facts an answer must contain (`required_facts`), facts it must not present as
current law (`forbidden_facts`, typically a repealed IPC/CrPC section), language
tag (English / Hindi / Hinglish), legal domain, task type and difficulty.
Every section number in the expected answers was verified against official
sources at build time (2026-07-12).

## FROZEN — 2026-07-14

Frozen as-is on the maintainer's instruction. Any correction from later review
went into Eval-v1, never into this file.

Time-sensitive facts baked into some answers — re-verify before relying on
them: BNS 106(2) deferral status, labour-code rules rollout, marital-rape and
Section 69 BNS litigation, women's-reservation (106th Amendment) implementation,
IT Rules traceability challenge, political-parties-under-RTI matter.

## Category split

| Category | Questions |
|---|---:|
| BNS | 70 |
| BNSS | 70 |
| BSA | 40 |
| Constitution | 50 |
| Consumer law | 40 |
| Cybercrime / IT Act | 40 |
| RTI | 30 |
| Domestic violence / women's protections | 30 |
| NI Act / cheque bounce | 30 |
| Motor Vehicles law | 25 |
| Labour / workplace / POSH | 25 |
| IPC→BNS / CrPC→BNSS mappings | 25 |
| Hindi | 10 |
| Hinglish | 10 |
| Safety / refusal / insufficient information | 5 |

## Record format

```json
{
  "id": "nyaya_eval_000001",
  "question": "IPC Section 420 ko BNS mein kis section se replace kiya gaya hai?",
  "language": "hinglish",
  "legal_domain": "criminal_law",
  "task_type": "old_new_law_mapping",
  "expected_answer": "Section 318 of the Bharatiya Nyaya Sanhita, 2023.",
  "required_facts": ["IPC Section 420", "BNS Section 318"],
  "forbidden_facts": [],
  "difficulty": "easy",
  "source": "official_source_reference",
  "split": "test"
}
```

## Licence

Questions and expected answers: CC-BY-4.0. Statutory text quoted inside answers
is Government of India material, public domain under Section 52(1)(q) of the
Copyright Act, 1957. **Not legal advice.** Built with
[github.com/JitendraJha98/nyaya-model](https://github.com/JitendraJha98/nyaya-model).
