# Nyaya-Eval-v0 draft — review before freezing

`nyaya_eval_v0_draft.jsonl` (500 records) is a **draft**. The eval set is frozen
forever once blessed (see README.md), so the freeze must be a deliberate human
act: review, then rename the file to `nyaya_eval_v0.jsonl` and delete this note.
The schema/count/split tests in `tests/test_eval_set.py` validate whichever of
the two files exists (frozen wins).

## How it was built (2026-07-12)

- Category counts follow README.md's target split exactly (validated by test).
  Counting rule: `language` hindi/hinglish rows count as the Hindi/Hinglish
  categories, `task_type=safety_abstention` as Safety, everything else by
  `legal_domain`.
- **Verification discipline:** section numbers appear in `expected_answer` /
  `required_facts` only where verified against official/authoritative sources
  on 2026-07-12 (BPRD/MHA IPC–BNS / CrPC–BNSS comparison tables, PIB, Gazette
  reporting, e.g.: BNS 103/318/304/85/80/64/69/106/111-113/152; BNSS
  173/35(3)/144/163/479/482/251/258/193/105; BSA 63/26/23; CPA jurisdiction
  Rules 2021 (50L/2Cr); MV Act 2019 fine amounts; labour codes in force
  21-Nov-2025; BNS 106(2) still deferred). Everything else is deliberately
  concept-level so a wrong section number cannot poison grading.
- Each record's `source` field says which basis it rests on
  ("verified 2026-07-12" vs "stable law" = settled pre-2024 law).

## What a human reviewer should check before freezing

1. **Spot-check ~10% of section citations** against indiacode.nic.in (the
   roadmap's ≥98%-clean gate). Highest-risk rows: BNS/BNSS punishments and the
   25 mapping rows.
2. **Time-sensitive rows** — re-verify at freeze time if months have passed:
   BNS 106(2) deferral status, labour-code rules rollout, marital-rape and
   Section 69 BNS litigation, women's-reservation (106th Amdt) implementation,
   IT Rules traceability challenge, political-parties-under-RTI matter.
3. **Hindi/Hinglish phrasing** — native-speaker read for naturalness.
4. **Safety rows (5)** — confirm the expected behaviour matches project policy.
5. Difficulty labels are the author's judgment; adjust freely (they are
   metadata, not graded).

Run `python -m pytest tests/test_eval_set.py` after any edit.
