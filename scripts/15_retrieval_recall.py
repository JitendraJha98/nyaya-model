"""Step 13 (v2) — Offline retrieval recall@k on the frozen eval set.

The GPU gate: v2 = RAG only makes sense if the retriever can actually surface
the sections the frozen eval requires. This measures that with zero GPU —
gold sections are parsed from each record's required_facts with the same
reference parser the retriever uses, then checked against the top-k
retrieved for the raw question.

Metrics per k:
  any_hit@k  — at least one gold section retrieved (partial grounding)
  full_hit@k — every gold section retrieved (complete grounding)

Records whose required_facts cite no statute section (pure-concept answers,
safety abstentions) are excluded from recall and counted separately.

Output: reports/retrieval_recall.json

Usage:
    python scripts/15_retrieval_recall.py
    python scripts/15_retrieval_recall.py --k 1 3 5 8
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Devanagari on cp1252 Windows consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.evaluation import load_eval_records
from nyaya.retrieval import CITATION_PATTERN, load_statute_index


def gold_keys(record: dict, index) -> tuple[set[str], list[str]]:
    """(resolvable gold section keys, citation-bearing facts that resolved to
    nothing — DB coverage gaps or parser misses)."""
    gold, unresolved = set(), []
    for fact in record.get("required_facts", []):
        keys = index.referenced_keys(fact, domain=record.get("legal_domain"))
        if keys:
            gold.update(keys)
        elif CITATION_PATTERN.search(fact):
            unresolved.append(fact)
    return gold, unresolved


def phrase_coverage(records, index, k: int) -> dict:
    """For records with NO resolvable citation gold (the phrase-only set):
    fraction whose top-k retrieved text contains at least one required_fact
    verbatim (case-insensitive). The KB-coverage metric — statute recall@k
    cannot see these records at all."""
    covered = total = 0
    for rec in records:
        facts = [f for f in rec.get("required_facts", []) if f.strip()]
        if not facts:
            continue
        gold, _ = gold_keys(rec, index)
        if gold:
            continue
        total += 1
        blob = " ".join(
            f"{h.get('title') or ''} {h.get('text') or ''}"
            for h in index.retrieve(rec["question"], k=k)).lower()
        if any(f.lower() in blob for f in facts):
            covered += 1
    return {"n": total,
            "any_fact_in_topk": round(covered / total, 4) if total else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 8])
    parser.add_argument("--canonical-dir", default=str(ROOT / "data" / "canonical"))
    args = parser.parse_args()

    index = load_statute_index(args.canonical_dir)
    records = load_eval_records()
    max_k = max(args.k)

    scored, no_gold, unresolved_facts = [], 0, []
    for rec in records:
        gold, unresolved = gold_keys(rec, index)
        unresolved_facts.extend(unresolved)
        if not gold:
            no_gold += 1
            continue
        hits = index.retrieve(rec["question"], k=max_k)
        hit_keys = [f"{h['act_id']}:{h['section'].upper()}" for h in hits]
        scored.append((rec, gold, hit_keys))

    report = {
        "eval_records": len(records),
        "gold_bearing": len(scored),
        "no_citation_gold": no_gold,
        "unresolved_fact_count": len(unresolved_facts),
        "unresolved_facts_sample": sorted(set(unresolved_facts))[:25],
        "recall": {},
        "by_language": {},
        "by_domain": {},
    }
    for k in sorted(args.k):
        any_hit = sum(1 for _, gold, hk in scored if gold & set(hk[:k]))
        full_hit = sum(1 for _, gold, hk in scored if gold <= set(hk[:k]))
        report["recall"][f"k={k}"] = {
            "any_hit": round(any_hit / len(scored), 4),
            "full_hit": round(full_hit / len(scored), 4),
        }

    buckets = {"by_language": "language", "by_domain": "legal_domain"}
    for out_key, field in buckets.items():
        agg = defaultdict(lambda: [0, 0])
        for rec, gold, hk in scored:
            cell = agg[rec.get(field, "unknown")]
            cell[0] += 1
            cell[1] += 1 if gold <= set(hk[:max_k]) else 0
        report[out_key] = {
            name: {"n": n, f"full_hit@{max_k}": round(hit / n, 4)}
            for name, (n, hit) in sorted(agg.items())
        }

    report["phrase_coverage"] = phrase_coverage(records, index, max_k)

    report["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = ROOT / "reports" / "retrieval_recall.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
