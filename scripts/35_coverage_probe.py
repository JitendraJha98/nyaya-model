"""Step 35 — How much of what citizens actually ask does the statute DB cover?

Runs the retriever over data/raw/citizen_questions.txt (269 real questions
collected by the maintainers, never published, never used for tuning) and
reports three things Eval-v1 cannot show:

  1. questions per legal-domain bucket, and which buckets have NO act in the DB
     (tenancy, property, contracts, children, parents ...) -- the retriever
     still returns eight confident sections for these;
  2. how many questions retrieve zero statute sections (guidance notes only),
     split by script, because Devanagari questions fail far more often;
  3. once nyaya.retrieval.StatuteIndex.coverage exists, how many the coverage
     gate would flag as outside the database.

Writes reports/coverage_probe.json. Re-run after every act added to
configs/acts.yaml; the numbers should move.

Usage:
    python scripts/35_coverage_probe.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.retrieval import load_statute_index  # noqa: E402

QUESTIONS = ROOT / "data" / "raw" / "citizen_questions.txt"
REPORT = ROOT / "reports" / "coverage_probe.json"
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# bucket -> (regex over the lowercased question, is any act for it in the DB?)
BUCKETS = {
    "tenancy_rent": (r"rent|landlord|tenant|kiraya|kirayedaar|makan malik|मकान मालिक|किराया|pg owner|deposit wapas|security deposit|खाली|khali karne", False),
    # Transfer of Property Act added Sept 2026 (inheritance acts still missing)
    "property_inheritance": (r"property|zameen|plot|jameen|wasiyat|\bwill\b|inheritance|hissa|batwara|संपत्ति|जमीन|registry|mutation|succession", True),
    # Indian Contract Act added Sept 2026
    "contract_loan": (r"agreement|contract|\bloan\b|\bemi\b|recovery agent|guarantor|udhaar|udhar|कर्ज|लोन|byaj|interest rate", True),
    "employment": (r"notice period|\bpf\b|gratuity|resign|terminate|fired|salary|tankhwah|वेतन|naukri|offer letter|\bbond\b", True),
    "dowry": (r"dowry|dahej|दहेज", False),
    "senior_citizens": (r"parents|maa baap|buzurg|senior citizen|बुजुर्ग|माँ-बाप|old age", False),
    # POCSO 2012 and Juvenile Justice Act 2015 added Sept 2026
    "children": (r"\bchild|bachch|baccha|\bminor\b|17 saal|16 saal|nabalig|बच्च|school", True),
    "traffic": (r"challan|helmet|\bdl\b|licence|license|gaadi|bike|scooty|\bcar\b|rc book|parking|drink and drive|pollution|toll|traffic", True),
    "police_fir_bail": (r"\bfir\b|police|thana|थाने|bail|zamanat|arrest|giraftar|\bncr\b|chargesheet", True),
    "consumer": (r"flipkart|amazon|refund|defective|warranty|consumer|product|delivery|online order|zomato|swiggy|insurance claim", True),
    "cheque_bank_fraud": (r"cheque|check bounce|\bupi\b|\botp\b|bank|scam|fraud|paisa kat|paise kat", True),
}


def main() -> None:
    questions = [l.strip() for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    index = load_statute_index(ROOT / "data" / "canonical")
    has_coverage = hasattr(index, "coverage")

    per_bucket = Counter()
    outside = set()
    uncategorised = 0
    zero_hits = Counter()
    n_by_script = Counter()
    flagged = Counter()
    for q in questions:
        ql = q.lower()
        hit_any = False
        for name, (pattern, in_db) in BUCKETS.items():
            if re.search(pattern, ql):
                per_bucket[name] += 1
                hit_any = True
                if not in_db:
                    outside.add(q)
        uncategorised += not hit_any

        script = "devanagari" if _DEVANAGARI.search(q) else "latin"
        n_by_script[script] += 1
        rows = index.retrieve(q, k=8)
        if not any(r["act_id"] != "procedures_kb" for r in rows):
            zero_hits[script] += 1
        if has_coverage and not index.coverage(q)["covered"]:
            flagged[script] += 1

    report = {
        "questions": len(questions),
        "acts_in_db": sorted({r["act_id"] for r in index.rows if r["act_id"] != "procedures_kb"}),
        "per_bucket": dict(per_bucket),
        "buckets_without_an_act": [b for b, (_p, in_db) in BUCKETS.items() if not in_db],
        "questions_in_buckets_without_an_act": len(outside),
        "uncategorised": uncategorised,
        "zero_statute_hits": {s: {"n": n_by_script[s], "zero": zero_hits[s]} for s in n_by_script},
        "coverage_gate_flagged": ({s: flagged[s] for s in n_by_script} if has_coverage else None),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[probe] {len(questions)} real citizen questions; "
          f"{len(outside)} ({len(outside) / len(questions):.0%}) fall in buckets with no act in the DB")
    for s, d in report["zero_statute_hits"].items():
        print(f"[probe] {s:<10} {d['zero']:>3}/{d['n']:<3} retrieve zero statute sections")
    if has_coverage:
        print(f"[probe] coverage gate flags: {dict(flagged)}")
    print(f"[probe] wrote {REPORT}")


if __name__ == "__main__":
    main()
