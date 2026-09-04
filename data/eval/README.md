# Nyaya-Eval-v0 — 500 manually curated questions (BUILD THIS FIRST)

This is roadmap Step 4, and it comes **before** creating any training data.
The file `nyaya_eval_v0.jsonl` is frozen once created: never train on it,
never edit it mid-experiment (make a v1 instead).

## FROZEN — 2026-07-14

Frozen as-is on the maintainer's instruction (built 2026-07-12; every
section number in expected answers was verified against official sources
at build time; validated by `tests/test_eval_set.py`). Any correction from
later review goes into **Eval-v1**, never into this file.

Time-sensitive facts baked into some answers — re-verify when building v1:
BNS 106(2) deferral status, labour-code rules rollout, marital-rape and
Section 69 BNS litigation, women's-reservation (106th Amdt) implementation,
IT Rules traceability challenge, political-parties-under-RTI matter.

## Target category split (can evolve)

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

## Record format (see `src/nyaya/schemas.py::EvalRecord`)

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

## Nyaya-Eval-v1 (2026-08-06) — the graded successor

Built by `scripts/25_build_eval_v1.py` from the frozen v0 file: 413 gradeable
records (87 need manual curation and are kept but not scored), 665 facts kept
and 400 quarantined, a public / private split of 305 / 195 rows by id hash.
Gold answers score 100% on every metric by construction. The scorer is
`nyaya.scoring` (strict citations, partial-credit substance); v0's strict
metric, which scored its own gold answers at 10.7%, is retired.

Only `nyaya_eval_v1_public.jsonl` is versioned. Because v0 is public, the
private half is reconstructible and is damage limitation, not a true holdout.
