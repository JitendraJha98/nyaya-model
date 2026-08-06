"""Step 28 — Does a retrieval change GENERALISE, or did it fit the eval set?

The trap this exists to catch
-----------------------------
Vocabulary fixes are found by auditing failures. That makes the headline
number self-fulfilling: you inspect the questions that fail, add terms that
fix exactly those, then report the improvement on the same set. It looks like
retrieval got better. It may only mean the eval got memorised.

The split here is the cheapest honest check available without writing new
questions:

    AUDITED       records whose gold was unreachable BEFORE the change --
                  i.e. the ones whose failures were actually inspected
    NEVER AUDITED every other gold-bearing record, never looked at

A change that generalises improves both. A change that was fitted to the eval
improves only the audited group, and the aggregate number is then an artefact
of how many failures happened to be inspected.

Result for the LEGAL_SYNONYMS expansion (commits 5c3d843, 57d89ec):

    AUDITED        n=32    full@8   0.0% -> 71.9%   (+71.9)
    NEVER AUDITED  n=118   full@8  80.5% -> 81.4%   (+0.8)

The aggregate 63.3% -> 79.3% is almost entirely the audited group. On
questions never inspected the gain is +0.8 points, which is noise. So the
synonym table is a PATCH for observed failures, not a general improvement, and
79.3% must not be quoted as expected performance on new questions.

This is not an argument for deleting the synonyms -- fixing 32 real failures
is worth having, and the mappings are legitimate. It is an argument that
enumerating citizen phrasings does not scale, and that the generalising fixes
are the semantic ones: cross-encoder reranking, stronger multilingual
embeddings, and LLM query expansion, none of which depend on having seen the
phrasing before.

Usage:
    python scripts/28_validate_generalization.py --baseline-ref <git-ref>
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import nyaya.retrieval as R  # noqa: E402
from nyaya.evaluation import load_eval_records  # noqa: E402

REPORT = ROOT / "reports" / "retrieval_generalization.json"


def _gold_keys_fn():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "r15", ROOT / "scripts" / "15_retrieval_recall.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gold_keys


def baseline_synonym_keys(ref: str) -> set[str]:
    """LEGAL_SYNONYMS keys as they were at `ref` — the pre-change vocabulary."""
    # encoding must be explicit: the synonym table contains Devanagari and the
    # Windows default (cp1252) cannot decode it.
    src = subprocess.run(
        ["git", "show", f"{ref}:src/nyaya/retrieval.py"],
        capture_output=True, text=True, encoding="utf-8",
        cwd=ROOT, check=True).stdout
    start = src.find("LEGAL_SYNONYMS = {")
    end = src.find("\n}", start)
    return set(re.findall(r'^\s+"([^"]+)":', src[start:end + 2], re.M))


def _install(synonyms: dict) -> None:
    """Swap the live synonym table and recompile its patterns."""
    R.LEGAL_SYNONYMS.clear()
    R.LEGAL_SYNONYMS.update(synonyms)
    R._SYNONYM_PATTERNS[:] = [
        (re.compile(rf"\b{re.escape(p)}\b" if p[0].isascii() else re.escape(p)), e)
        for p, e in synonyms.items()
    ]


def measure(synonyms: dict, records, gold_keys, k: int, depth: int) -> dict:
    _install(synonyms)
    index = R.load_statute_index(str(ROOT / "data" / "canonical"))
    out = {}
    for rec in records:
        gold, _ = gold_keys(rec, index)
        if not gold:
            continue
        keys = [f"{h['act_id']}:{h['section'].upper()}"
                for h in index.retrieve(rec["question"], k=depth)]
        out[rec["id"]] = (gold <= set(keys[:k]), gold <= set(keys))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-ref", default="503d0df",
                   help="git ref holding the pre-change LEGAL_SYNONYMS")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--depth", type=int, default=100)
    args = p.parse_args()

    gold_keys = _gold_keys_fn()
    records = load_eval_records()
    full = dict(R.LEGAL_SYNONYMS)
    keys = baseline_synonym_keys(args.baseline_ref)
    base = {k: v for k, v in full.items() if k in keys}

    print(f"[generalization] baseline {args.baseline_ref}: {len(base)} synonyms, "
          f"current: {len(full)} (+{len(full) - len(base)})")

    before = measure(base, records, gold_keys, args.k, args.depth)
    after = measure(full, records, gold_keys, args.k, args.depth)
    _install(full)  # leave the process in the real configuration

    audited = {i for i, (_h, reachable) in before.items() if not reachable}
    never = set(before) - audited

    rows = {}
    for name, group in (("audited", audited), ("never_audited", never)):
        if not group:
            continue
        b = sum(before[i][0] for i in group) / len(group)
        a = sum(after[i][0] for i in group) / len(group)
        rows[name] = {"n": len(group), "before": round(b, 4),
                      "after": round(a, 4), "delta": round(a - b, 4)}

    overall_b = sum(v[0] for v in before.values()) / len(before)
    overall_a = sum(v[0] for v in after.values()) / len(after)
    rows["overall"] = {"n": len(before), "before": round(overall_b, 4),
                       "after": round(overall_a, 4),
                       "delta": round(overall_a - overall_b, 4)}

    print(f"\n{'group':<16}{'n':>6}{'before':>10}{'after':>9}{'delta':>10}")
    for name in ("audited", "never_audited", "overall"):
        r = rows.get(name)
        if r:
            print(f"{name:<16}{r['n']:>6}{r['before']:>10.1%}"
                  f"{r['after']:>9.1%}{r['delta']:>+10.1%}")

    never_delta = rows.get("never_audited", {}).get("delta", 0.0)
    generalises = never_delta >= 0.03
    print(f"\nVERDICT: {'generalises' if generalises else 'DID NOT GENERALISE'} "
          f"— never-audited delta {never_delta:+.1%}")
    if not generalises:
        print("The aggregate gain is an artefact of auditing. Do not quote the\n"
              "overall number as expected performance on new questions.")

    rows["generalises"] = generalises
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n[generalization] wrote {REPORT}")

    # Persist the audited set so EVERY later retrieval measurement can report
    # the never-audited slice as its headline. Tuning was done against these
    # 32 records; any number that includes them is optimistic by construction,
    # and that should be the default view rather than a thing to remember.
    audited_path = ROOT / "reports" / "audited_record_ids.json"
    audited_path.write_text(json.dumps({
        "note": "Records whose retrieval failures were inspected while writing "
                "LEGAL_SYNONYMS. Recall including these is tuned-on and "
                "optimistic; report the never-audited slice instead.",
        "baseline_ref": args.baseline_ref,
        "audited": sorted(audited),
    }, indent=2), encoding="utf-8")
    print(f"[generalization] wrote {audited_path} ({len(audited)} records)")


if __name__ == "__main__":
    main()
