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
    """Content coverage for records with NO resolvable citation gold.

    Statute recall@k cannot see these records at all — their required facts are
    phrases ("up to 3 years", "Zero FIR"), not section references — so this is
    the only measure of whether retrieval surfaced the substance they need.

    It originally tested for the fact appearing VERBATIM in the retrieved text,
    which is the same defect that made Eval-v0 score its own gold answers at
    10.7%: statutes say "which may extend to three years", the eval says "up to
    3 years", and a substring test calls that a miss. That reported 5.1%
    coverage and pointed the whole project at a retrieval problem that was
    substantially a measurement problem — the paraphrase-tolerant score is 64%.

    Scored with nyaya.scoring so retrieval and answers are judged on the same
    basis. Both numbers are reported: `any_fact_verbatim` for continuity with
    earlier reports, `any_fact_scored` / `mean_best_fact` as the real signal.
    """
    from nyaya.scoring import score_fact

    verbatim = scored = total = 0
    best_scores = []
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
            for h in index.retrieve(rec["question"], k=k))
        verbatim += any(f.lower() in blob.lower() for f in facts)
        best = max(score_fact(f, blob)["score"] for f in facts)
        best_scores.append(best)
        scored += best > 0

    def pct(x):
        return round(x / total, 4) if total else 0.0

    return {
        "n": total,
        "any_fact_verbatim": pct(verbatim),
        "any_fact_scored": pct(scored),
        "mean_best_fact": round(sum(best_scores) / total, 4) if total else 0.0,
        # kept so older reports/dashboards keep resolving
        "any_fact_in_topk": pct(scored),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 8])
    parser.add_argument("--canonical-dir", default=str(ROOT / "data" / "canonical"))
    parser.add_argument("--dense", nargs="?",
                        const="intfloat/multilingual-e5-base",
                        default=None, metavar="MODEL",
                        help="enable the hybrid dense stage (requires "
                             "requirements-dense.txt); writes a separate "
                             "retrieval_recall_dense.json. Bare --dense uses "
                             "e5-base — the model behind the committed "
                             "recall numbers — via the shared .npy cache")
    parser.add_argument("--rerank", nargs="?", const=None, default=False,
                        metavar="MODEL",
                        help="enable the cross-encoder second stage "
                             "(nyaya.rerank). Bare --rerank uses the default "
                             "multilingual checkpoint. Scores are cached on "
                             "disk so repeat sweeps are near-free.")
    parser.add_argument("--rerank-depth", type=int, default=50,
                        help="candidates pulled from stage 1 before reranking; "
                             "this is the ceiling on what reranking can recover")
    parser.add_argument("--limit", type=int,
                        help="only score the first N eval records (fast iteration)")
    args = parser.parse_args()

    index = load_statute_index(args.canonical_dir)
    reranker = None
    if args.rerank is not False:
        from nyaya.rerank import DEFAULT_MODEL, CachedReranker, CrossEncoderReranker
        model_name = args.rerank or DEFAULT_MODEL
        inner = CrossEncoderReranker(model_name=model_name, depth=args.rerank_depth)
        reranker = CachedReranker(
            inner, ROOT / "data" / "generated" / "rerank_cache.json")
        index.set_reranker(reranker)
        print(f"[recall] reranking with {model_name} (depth {args.rerank_depth})")
    if args.dense:
        from nyaya.dense import DEFAULT_ATTACH_MODEL, attach_dense_index
        attach_dense_index(
            index, model_name=args.dense,
            # the named .npy cache only matches the e5-base doc vectors
            cache_path=(ROOT / "data" / "generated" / "e5_doc_vectors.npy"
                        if args.dense == DEFAULT_ATTACH_MODEL else None))
    records = load_eval_records()
    if args.limit:
        records = records[: args.limit]
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
        "dense_model": args.dense,
        "eval_records": len(records),
        "gold_bearing": len(scored),
        "no_citation_gold": no_gold,
        "unresolved_fact_count": len(unresolved_facts),
        "unresolved_facts_sample": sorted(set(unresolved_facts))[:25],
        "recall": {},
        "by_language": {},
        "by_domain": {},
    }
    # The synonym table was written by reading the failures of 32 specific
    # records (scripts/28). Recall that includes them is tuned-on and
    # optimistic by construction -- it showed +16pts overall while the
    # never-audited slice moved +0.9pts. Report both, always, so the honest
    # number cannot be lost by whoever reads this next.
    audited_path = ROOT / "reports" / "audited_record_ids.json"
    audited = set()
    if audited_path.exists():
        try:
            audited = set(json.loads(audited_path.read_text(encoding="utf-8"))["audited"])
        except (ValueError, KeyError):
            audited = set()
    clean = [row for row in scored if row[0]["id"] not in audited]
    report["audited_excluded"] = len(scored) - len(clean)

    for k in sorted(args.k):
        any_hit = sum(1 for _, gold, hk in scored if gold & set(hk[:k]))
        full_hit = sum(1 for _, gold, hk in scored if gold <= set(hk[:k]))
        cell = {
            "any_hit": round(any_hit / len(scored), 4),
            "full_hit": round(full_hit / len(scored), 4),
        }
        if clean and len(clean) != len(scored):
            cell["full_hit_never_audited"] = round(
                sum(1 for _, gold, hk in clean if gold <= set(hk[:k])) / len(clean), 4)
            cell["n_never_audited"] = len(clean)
        report["recall"][f"k={k}"] = cell

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
    # the dense run goes to a separate file so the canonical BM25-only report
    # is never overwritten by an experimental hybrid measurement
    if reranker is not None:
        reranker.flush()          # persist scores so repeat sweeps are near-free
    # Separate files per configuration so an experimental run can never
    # overwrite the canonical BM25-only numbers.
    if reranker is not None:
        name = "retrieval_recall_rerank.json"
    elif args.dense:
        name = "retrieval_recall_dense.json"
    else:
        name = "retrieval_recall.json"
    out = ROOT / "reports" / name
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
