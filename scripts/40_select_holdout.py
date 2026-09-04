"""Step 40 — Select the questions for Holdout-v1, the project's first true holdout.

Nyaya-Eval-v0 is public, so Eval-v1's private half is reconstructible. The 269
citizen questions in data/raw/citizen_questions.txt were collected by the
maintainers, never published and never used for tuning: they are the only
genuinely held-out material the project has. This picks a stratified 180 of
them (up to 15 per legal-domain bucket, the rest from the uncategorised pool)
and writes them as EvalRecord drafts with EMPTY gold fields for a human
reviewer to fill (docs/HOLDOUT_REVIEW.md).

The draft and the finished file are gitignored: publishing them would burn the
holdout exactly as publishing v0 did.

Usage:
    python scripts/40_select_holdout.py                 # -> data/eval/holdout_v1_draft.jsonl
    python scripts/40_select_holdout.py --n 180 --seed 0
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "raw" / "citizen_questions.txt"
OUT = ROOT / "data" / "eval" / "holdout_v1_draft.jsonl"
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Same buckets as scripts/35_coverage_probe.py; the legal_domain values match the
# Eval-v1 vocabulary where one exists so per-domain reporting lines up.
BUCKETS = {
    "tenancy_rent": r"rent|landlord|tenant|kiraya|kirayedaar|makan malik|मकान मालिक|किराया|pg owner|deposit wapas|security deposit|खाली|khali karne",
    "property_inheritance": r"property|zameen|plot|jameen|wasiyat|\bwill\b|inheritance|hissa|batwara|संपत्ति|जमीन|registry|mutation|succession",
    "contract_loan": r"agreement|contract|\bloan\b|\bemi\b|recovery agent|guarantor|udhaar|udhar|कर्ज|लोन|byaj|interest rate",
    "labour_law": r"notice period|\bpf\b|gratuity|resign|terminate|fired|salary|tankhwah|वेतन|naukri|offer letter|\bbond\b",
    "womens_protection": r"dowry|dahej|दहेज|domestic violence|maar|मारपीट|harass",
    "senior_citizens": r"parents|maa baap|buzurg|senior citizen|बुजुर्ग|माँ-बाप|old age",
    "children": r"\bchild|bachch|baccha|\bminor\b|17 saal|16 saal|nabalig|बच्च|school|custody",
    "motor_vehicles": r"challan|helmet|\bdl\b|licence|license|gaadi|bike|scooty|\bcar\b|rc book|parking|drink and drive|pollution|toll|traffic",
    "bnss": r"\bfir\b|police|thana|थाने|bail|zamanat|arrest|giraftar|\bncr\b|chargesheet",
    "consumer_law": r"flipkart|amazon|refund|defective|warranty|consumer|product|delivery|online order|zomato|swiggy|insurance",
    "cheque_bounce_cyber": r"cheque|check bounce|\bupi\b|\botp\b|bank|scam|fraud|paisa kat|paise kat",
    "family": r"divorce|talaq|तलाक|shaadi|marriage|wife|husband|पति|पत्नी",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=180)
    p.add_argument("--per-bucket", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    questions = [l.strip() for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(args.seed)
    by_bucket = defaultdict(list)
    for q in questions:
        ql = q.lower()
        hit = next((b for b, pat in BUCKETS.items() if re.search(pat, ql)), "uncategorised")
        by_bucket[hit].append(q)

    chosen = []
    for bucket, qs in sorted(by_bucket.items()):
        if bucket == "uncategorised":
            continue
        rng.shuffle(qs)
        chosen += [(bucket, q) for q in qs[: args.per_bucket]]
    rest = by_bucket["uncategorised"][:]
    rng.shuffle(rest)
    chosen += [("uncategorised", q) for q in rest[: max(0, args.n - len(chosen))]]
    chosen = chosen[: args.n]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for i, (bucket, q) in enumerate(chosen, 1):
            fh.write(json.dumps({
                "id": f"nyaya_holdout_{i:06d}",
                "question": q,
                "language": "hindi" if _DEVANAGARI.search(q) else "hinglish",
                "legal_domain": bucket,
                "task_type": "scenario",
                "expected_answer": "",
                "required_facts": [],
                "forbidden_facts": [],
                "difficulty": "medium",
                "source": "citizen_questions.txt (maintainer-collected, unpublished)",
                "split": "holdout",
                "review_status": "pending",
            }, ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for b, _q in chosen:
        counts[b] += 1
    print(f"[holdout] wrote {len(chosen)} drafts -> {OUT}")
    for b, c in sorted(counts.items()):
        print(f"  {b:<22} {c}")
    print("\nNext: a reviewer fills expected_answer / required_facts / forbidden_facts per docs/HOLDOUT_REVIEW.md.")


if __name__ == "__main__":
    main()
