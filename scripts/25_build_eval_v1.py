"""Step 25 — Build Nyaya-Eval-v1 from the frozen v0 set.

WHY
---
Eval-v0's strict metric scores the eval set's own gold `expected_answer` at
~10.7%. A metric that fails its own reference answers 89% of the time has no
headroom: base / v3 / v4 all landed within a 2-answer spread, so it could not
show whether training helped. Part of that was a matcher bug (fixed in
nyaya.evaluation) and a substring-only rule (replaced by nyaya.scoring); the
rest is that some required_facts are simply not satisfied by their own gold
answer, and no scorer can fix a fact like that.

THE CURATION RULE
-----------------
A required fact that its OWN gold answer does not satisfy is a broken fact.
That is objective and automatable, so this script applies it:

  score(fact, expected_answer) == 1.0  -> KEEP, graded in v1
  score(fact, expected_answer)  < 1.0  -> QUARANTINE, not graded, reported
                                          for manual rewrite

Nothing is deleted. Quarantined facts travel with the record in
`quarantined_facts` (with the reason) so a human can rewrite them and promote
them back. Records left with no gradeable fact are marked `needs_curation` and
excluded from the scored core.

By construction the resulting set has a gold-answer ceiling of 100%: a perfect
answer scores 1.0. That is the property v0 lacked.

PUBLIC / PRIVATE SPLIT — AND WHAT IT DOES NOT BUY
-------------------------------------------------
`NyayaLabs98/nyaya-eval-v0` was published, so v0 is contaminated: anyone can
train on every one of its questions. v1 is split anyway —

  public  (60%) — publishable, so others can reproduce numbers
  private (40%) — not published, not committed (see .gitignore)

— but BE HONEST ABOUT WHAT THIS IS. v1 is derived from v0, whose questions are
already public, and the split is deterministic, so the private half is fully
reconstructible by anyone with this script. It is NOT a cryptographically
held-out set, and a salt would not fix that: the questions themselves are out.

What the split actually buys: it stops *our own* accidental reuse, gives a
clean public benchmark others can cite, and limits casual leakage. It does not
restore held-out status.

A genuinely held-out benchmark requires NEW questions that have never been
published (Eval-v2). That is human authoring work, not something this script
can manufacture. Until then, treat every v1 number as potentially optimistic
for any model that could have seen v0.

The split is deterministic (sha256 of the record id, not RNG state) and
stratified by domain x difficulty x language, so both halves stay
representative and the assignment is stable across reruns and machines.

Usage:
    python scripts/25_build_eval_v1.py                 # write v1 + report
    python scripts/25_build_eval_v1.py --dry-run       # report only
"""

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nyaya.scoring import (  # noqa: E402
    aggregate, forbidden_present, lint_fact, score_fact, score_record,
)

EVAL_V0 = ROOT / "data" / "eval" / "nyaya_eval_v0.jsonl"
EVAL_V1 = ROOT / "data" / "eval" / "nyaya_eval_v1.jsonl"
EVAL_V1_PUBLIC = ROOT / "data" / "eval" / "nyaya_eval_v1_public.jsonl"
EVAL_V1_PRIVATE = ROOT / "data" / "eval" / "nyaya_eval_v1_private.jsonl"
REPORT = ROOT / "reports" / "eval_v1_curation.json"

PUBLIC_FRACTION = 0.6


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _bucket_key(record: dict) -> tuple:
    return (record.get("legal_domain"), record.get("difficulty"), record.get("language"))


def _stable_rank(record_id: str) -> int:
    """Deterministic per-record ordinal — stable across runs, machines, Python
    versions. PYTHONHASHSEED-dependent hash() would silently reshuffle the
    private set between runs, which would quietly leak held-out records."""
    return int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:16], 16)


def curate(records: list[dict]) -> tuple[list[dict], dict]:
    """Apply the self-consistency rule. Returns (v1_records, curation_report)."""
    out, quarantine_log, forbidden_log = [], [], []
    kept_facts = dropped_facts = 0

    for record in records:
        gold = record["expected_answer"]
        keep, quarantined = [], []

        for fact in record.get("required_facts", []):
            result = score_fact(fact, gold)
            if result["score"] == 1.0:
                keep.append(fact)
                kept_facts += 1
                continue
            quarantined.append({
                "fact": fact,
                "gold_score": result["score"],
                "reason": result["reason"],
                "lint": lint_fact(fact),
            })
            dropped_facts += 1
            quarantine_log.append({
                "id": record["id"],
                "legal_domain": record.get("legal_domain"),
                "fact": fact,
                "gold_score": result["score"],
                "reason": result["reason"],
                "lint": lint_fact(fact),
                "expected_answer": gold,
            })

        # Forbidden facts get linted too. A proposition like "sedition under
        # 124A is still in force" is undetectable by any string metric — it
        # never fires, so the record silently tests nothing. Flag for rewrite
        # into the detectable citation form ("Section 124A IPC as current law").
        # Same self-consistency rule, applied to forbidden facts: the gold
        # answer is correct by construction, so if it TRIPS a forbidden fact,
        # the forbidden fact is wrong — not the answer. Those are quarantined
        # too, which is what makes the 100% gold ceiling hold by construction.
        forbidden_keep, forbidden_flagged = [], []
        for fact in record.get("forbidden_facts", []):
            problems = lint_fact(fact)
            fires_on_gold = forbidden_present(fact, gold)
            if problems or fires_on_gold:
                forbidden_flagged.append({
                    "fact": fact, "lint": problems, "fires_on_gold": fires_on_gold,
                })
                forbidden_log.append({
                    "id": record["id"],
                    "fact": fact,
                    "lint": problems,
                    "fires_on_gold": fires_on_gold,
                    "suggestion": (
                        "fires on its own gold answer — the forbidden fact is "
                        "mis-specified" if fires_on_gold else
                        "rewrite as '<Section N Act> as current law' so it is detectable"
                    ),
                })
            else:
                forbidden_keep.append(fact)

        new_record = dict(record)
        new_record["required_facts"] = keep
        new_record["quarantined_facts"] = quarantined
        new_record["forbidden_facts"] = forbidden_keep
        new_record["forbidden_needs_rewrite"] = forbidden_flagged
        # A record with nothing gradeable cannot contribute signal; keep it in
        # the file (so the rewrite work is visible) but flag it out of scoring.
        new_record["needs_curation"] = not keep
        new_record["source_eval"] = "nyaya_eval_v0"
        out.append(new_record)

    gradeable = [r for r in out if not r["needs_curation"]]
    report = {
        "records_in": len(records),
        "records_gradeable": len(gradeable),
        "records_needing_curation": len(out) - len(gradeable),
        "facts_kept": kept_facts,
        "facts_quarantined": dropped_facts,
        "quarantine_by_reason": dict(collections.Counter(
            q["reason"].split(":")[0] for q in quarantine_log)),
        "quarantine_by_domain": dict(collections.Counter(
            q["legal_domain"] for q in quarantine_log)),
        "forbidden_needing_rewrite": len(forbidden_log),
        "quarantined": quarantine_log,
        "forbidden_to_rewrite": forbidden_log,
    }
    return out, report


def split_public_private(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic, stratified public/private assignment."""
    buckets = collections.defaultdict(list)
    for record in records:
        buckets[_bucket_key(record)].append(record)

    public, private = [], []
    for _, rows in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        rows.sort(key=lambda r: _stable_rank(r["id"]))
        cutoff = round(len(rows) * PUBLIC_FRACTION)
        for i, record in enumerate(rows):
            tagged = dict(record)
            tagged["visibility"] = "public" if i < cutoff else "private"
            (public if i < cutoff else private).append(tagged)
    return public, private


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Report only, write nothing")
    args = p.parse_args()

    records = _load(EVAL_V0)
    v1, report = curate(records)
    public, private = split_public_private(v1)

    # Validate the whole point of the exercise: the gold answers must now score
    # ~1.0. If they do not, the curation rule did not hold and v1 is not sound.
    scored_gold = [score_record(r, r["expected_answer"])
                   for r in v1 if not r["needs_curation"]]
    ceiling = aggregate(scored_gold)
    report["gold_ceiling"] = {
        "fact_recall": ceiling["fact_recall"],
        "citation_accuracy": ceiling["citation_accuracy"],
        "all_facts_accuracy": ceiling["all_facts_accuracy"],
    }
    report["split"] = {
        "public": len(public),
        "private": len(private),
        "public_fraction": PUBLIC_FRACTION,
    }

    print(f"[eval-v1] records in            : {report['records_in']}")
    print(f"[eval-v1] gradeable             : {report['records_gradeable']}")
    print(f"[eval-v1] need manual curation  : {report['records_needing_curation']}")
    print(f"[eval-v1] facts kept/quarantined: {report['facts_kept']} / {report['facts_quarantined']}")
    print(f"[eval-v1] split public/private  : {len(public)} / {len(private)}")
    print()
    print(f"[eval-v1] GOLD CEILING fact_recall      : {ceiling['fact_recall']:.1%}")
    print(f"[eval-v1] GOLD CEILING citation accuracy: {ceiling['citation_accuracy']:.1%}")
    print(f"[eval-v1] GOLD CEILING all_facts        : {ceiling['all_facts_accuracy']:.1%}")

    if ceiling["fact_recall"] < 0.999:
        print("\n[eval-v1] WARNING: gold ceiling is below 100% — the curation rule "
              "did not fully hold. Inspect reports/eval_v1_curation.json.")

    if args.dry_run:
        print("\n[eval-v1] --dry-run: nothing written.")
        return

    _write(EVAL_V1, v1)
    _write(EVAL_V1_PUBLIC, public)
    _write(EVAL_V1_PRIVATE, private)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[eval-v1] wrote {EVAL_V1}")
    print(f"[eval-v1] wrote {EVAL_V1_PUBLIC}")
    print(f"[eval-v1] wrote {EVAL_V1_PRIVATE}   <- NEVER publish this file")
    print(f"[eval-v1] wrote {REPORT}")


if __name__ == "__main__":
    main()
