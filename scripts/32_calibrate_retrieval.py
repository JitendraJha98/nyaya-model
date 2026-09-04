"""Step 32 — Calibrate the coverage gate (retrieval.COVERAGE_MIN_SCORE).

Three populations, one number:
  eval_in_coverage   Eval-v1 public records whose facts name a section AND whose
                     id is not in reports/audited_record_ids.json (never used to
                     tune retrieval) -- must stay covered;
  real_in_coverage   citizen questions (data/raw/citizen_questions.txt) in
                     domains the DB holds (traffic, FIR/bail, consumer, cheque,
                     divorce, RTI, domestic violence) -- must stay covered;
  real_out_of_cover  citizen questions in domains with NO act in the DB
                     (tenancy, property, loans, children, parents) -- should be
                     flagged.

The eval questions are written in statutory English and score far higher than
real Hinglish questions, so calibrating on them alone (as the first attempt did,
threshold 15.7) flags almost half of real in-coverage questions. The threshold
chosen here is the largest candidate that keeps >= 90% of BOTH in-coverage
populations. Prints the score percentiles, a threshold table, and writes
reports/coverage_calibration.json. Set COVERAGE_MIN_SCORE to the printed value.

Usage:
    python scripts/32_calibrate_retrieval.py
"""
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.retrieval import load_statute_index  # noqa: E402

EVAL = ROOT / "data" / "eval" / "nyaya_eval_v1_public.jsonl"
AUDITED = ROOT / "reports" / "audited_record_ids.json"
QUESTIONS = ROOT / "data" / "raw" / "citizen_questions.txt"
REPORT = ROOT / "reports" / "coverage_calibration.json"
CANDIDATES = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0)
KEEP_RATE = 0.90

IN_COVERAGE = re.compile(
    r"challan|helmet|\bdl\b|licence|license|gaadi|bike|scooty|\bcar\b|rc book|parking|drink and drive|pollution|traffic"
    r"|\bfir\b|police|thana|थाने|bail|zamanat|arrest|giraftar|\bncr\b"
    r"|flipkart|amazon|refund|defective|warranty|consumer|cheque|check bounce|\bupi\b|\botp\b|scam|fraud"
    r"|divorce|talaq|तलाक|\brti\b|domestic violence|dowry|dahej", re.IGNORECASE)
# Domains with no act in the DB as of Sept 2026 (after 14 acts were added from
# the India Code API): rent control, tax, passports/visas, intellectual property,
# arbitration, insurance regulation.
OUT_OF_COVERAGE = re.compile(
    r"rent control|income tax|\bgst\b|passport|visa|immigration|trademark|copyright|patent"
    r"|arbitration|irdai|insurance ombudsman|property tax|stamp duty", re.IGNORECASE)


def _score(index, text: str) -> float:
    cov = index.coverage(text)
    return float("inf") if cov["top_statute_score"] is None else cov["top_statute_score"]


def _pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[max(0, min(len(values) - 1, int(round(q * (len(values) - 1)))))]


def _summary(values: list[float]) -> dict:
    return {"n": len(values), "p05": _pct(values, .05), "p25": _pct(values, .25), "median": _pct(values, .5)}


def main() -> None:
    index = load_statute_index(ROOT / "data" / "canonical")
    audited = set(json.loads(AUDITED.read_text(encoding="utf-8"))["audited"])

    eval_in = []
    for r in (json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()):
        if r["id"] in audited or r.get("task_type") == "safety_abstention":
            continue
        gold = set()
        for fact in r.get("required_facts", []):
            gold.update(index.referenced_keys(fact, domain=r.get("legal_domain")))
        if gold:
            eval_in.append(_score(index, r["question"]))
    eval_in = [s for s in eval_in if s != float("inf")]  # explicit citations are always covered

    questions = [l.strip() for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    real_in = [_score(index, q) for q in questions if IN_COVERAGE.search(q) and not OUT_OF_COVERAGE.search(q)]
    real_out = [_score(index, q) for q in questions if OUT_OF_COVERAGE.search(q) and not IN_COVERAGE.search(q)]
    real_in = [s for s in real_in if s != float("inf")]
    real_out = [s for s in real_out if s != float("inf")]

    table = []
    for thr in CANDIDATES:
        table.append({
            "threshold": thr,
            "eval_in_coverage_kept": sum(s >= thr for s in eval_in) / max(1, len(eval_in)),
            "real_in_coverage_kept": sum(s >= thr for s in real_in) / max(1, len(real_in)),
            "real_out_of_coverage_flagged": (sum(s < thr for s in real_out) / len(real_out)) if real_out else None,
        })
    ok = [row for row in table if row["eval_in_coverage_kept"] >= KEEP_RATE and row["real_in_coverage_kept"] >= KEEP_RATE]
    chosen = max(ok, key=lambda row: row["threshold"]) if ok else table[0]

    report = {
        "populations": {"eval_in_coverage": _summary(eval_in), "real_in_coverage": _summary(real_in),
                        "real_out_of_coverage": _summary(real_out)},
        "table": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()} for row in table],
        "rule": f"largest threshold keeping >= {KEEP_RATE:.0%} of both in-coverage populations",
        "chosen_threshold": chosen["threshold"],
        "note": "BM25 top score is a weak separator on real questions; the durable fixes are adding the "
                "missing acts and a reranker-based gate.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'threshold':>9} {'eval kept':>10} {'real kept':>10} {'out flagged':>12}")
    for row in table:
        print(f"{row['threshold']:>9} {row['eval_in_coverage_kept']:>10.0%} {row['real_in_coverage_kept']:>10.0%} "
              f"{row['real_out_of_coverage_flagged']:>12.0%}")
    print(f"\n[calibrate] set COVERAGE_MIN_SCORE = {chosen['threshold']} in src/nyaya/retrieval.py; wrote {REPORT}")


if __name__ == "__main__":
    main()
