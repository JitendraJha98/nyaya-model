# Holdout-v1 — reviewer brief

You are writing the answer key for 180 real questions Indian citizens asked, in
Hindi and Hinglish. This is the project's first genuine holdout: the questions
have never been published or used to tune anything, and the finished file stays
private (`data/eval/nyaya_holdout_v1_private.jsonl`, gitignored). Please do not
share it.

Time: about 5 minutes per question; 180 questions is three or four sittings.

## What to fill in, per record

Open `data/eval/holdout_v1_draft.jsonl` (one JSON object per line) and fill:

- **`expected_answer`** — 3 to 5 sentences a careful advocate would give a
  layperson: which provision applies, the concrete specifics (years, amounts,
  deadlines), and the practical next step. Write in English even if the question
  is Hindi; the scorer matches facts, not language.
- **`required_facts`** — 2 to 4 short, quotable phrases that any correct answer
  must contain. At least one must be a section citation in the form
  `Section <n> <ACT>` using these act names: `BNS`, `BNSS`, `BSA`, `Constitution`
  (use `Article <n> Constitution`), `MV Act`, `NI Act`, `IT Act`, `CPA`, `Wages
  Code`, `SMA`, `HMA`, `DV Act`, `RTI`, `POSH`. Substance facts should be short
  and concrete: `up to 3 years`, `within 30 days`, `Zero FIR`, `Rs 10,000`.
  Avoid propositions such as "the law does not apply"; the scorer cannot grade
  them (run the linter below).
- **`forbidden_facts`** — repealed provisions that must not be presented as
  current law, phrased `Section 420 IPC as current law`. Leave empty if none.
- **`difficulty`** — `easy` (one section, one fact), `medium`, or `hard`
  (several acts, or a common misconception).
- **`review_status`** — change `pending` to `done`.

If the question is about a law the database does not hold (rent control,
property, contracts, children, senior citizens), still answer it correctly,
set `legal_domain` to `out_of_coverage`, and cite the real act by name in
`required_facts` as `<Act name> applies` — those records measure whether the
system knows what it does not know.

If a question cannot be answered responsibly without facts the asker did not
give, set `task_type` to `safety_abstention` and make `expected_answer` the
clarifying question an advocate would ask.

## Check your facts before handing back

Every fact must survive the project's fact linter, otherwise the scorer cannot
grade it:

```bash
python - <<'EOF'
import json, sys
sys.path.insert(0, "src")
from nyaya.scoring import lint_fact
for line in open("data/eval/holdout_v1_draft.jsonl", encoding="utf-8"):
    r = json.loads(line)
    for f in r["required_facts"]:
        problems = lint_fact(f)
        if problems:
            print(r["id"], repr(f), problems)
EOF
```

When the linter is silent, rename the file to
`data/eval/nyaya_holdout_v1_private.jsonl`. The maintainers then run the gold
ceiling check (every expected answer must score 100% on its own facts) and the
evaluation: `python scripts/26_eval_v1_run.py --split holdout ...`.

## Sources to verify against

India Code (indiacode.nic.in) for section text; the NCRB "Sankalan" tables for
IPC→BNS / CrPC→BNSS correspondences (the same tables the repository ships as
`data/canonical/law_mappings.jsonl`); official portals for procedure (e-Daakhil,
cybercrime.gov.in, Parivahan, RTI Online). Do not use blog summaries.
