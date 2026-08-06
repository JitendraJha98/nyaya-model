"""Step 27 — Paired comparison between two Eval-v1 runs.

Unpaired means hide small effects. Both runs answer the SAME questions, so
compare per question and bootstrap the paired difference: that removes
question difficulty as a source of variance and is the only way a ~1-point
delta at n=409 can be honestly called real or not.

Reports a 95% CI on the difference. If it spans zero, the runs are
statistically indistinguishable and no "better than" claim is defensible --
which is exactly the claim a model card is tempted to make.

Usage:
    python scripts/27_compare_runs.py --a base --b nyaya-3b-v3
"""

import argparse
import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.scoring import score_record  # noqa: E402

RUNS = ROOT / "outputs" / "eval-v1"
REPORT = ROOT / "reports" / "eval_v1_comparison.json"
BOOTSTRAP_ROUNDS = 10000


def load_run(label: str) -> dict:
    path = RUNS / label / "predictions.jsonl"
    if not path.exists():
        sys.exit(f"[compare] {path} not found")
    return {r["id"]: r for r in
            (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def paired_scores(a: dict, b: dict, metric: str) -> list[tuple[float, float]]:
    pairs = []
    for rid in sorted(set(a) & set(b)):
        sa = score_record(a[rid]["record"], a[rid]["response"])
        sb = score_record(b[rid]["record"], b[rid]["response"])
        if sa["is_safety_row"]:
            continue
        va, vb = sa.get(metric), sb.get(metric)
        if va is None or vb is None:
            continue
        pairs.append((va, vb))
    return pairs


def bootstrap_ci(diffs: list[float], rounds: int = BOOTSTRAP_ROUNDS, seed: int = 0):
    """Percentile bootstrap CI on the mean paired difference."""
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(rounds):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(rounds * 0.025)], means[int(rounds * 0.975)]


def compare(label_a: str, label_b: str, metric: str) -> dict:
    a, b = load_run(label_a), load_run(label_b)
    pairs = paired_scores(a, b, metric)
    n = len(pairs)
    if not n:
        sys.exit(f"[compare] no comparable rows for metric '{metric}'")

    mean_a = sum(x for x, _ in pairs) / n
    mean_b = sum(y for _, y in pairs) / n
    diffs = [y - x for x, y in pairs]
    lo, hi = bootstrap_ci(diffs)
    indistinguishable = lo < 0 < hi

    return {
        "metric": metric,
        "n": n,
        "a": {"label": label_a, "mean": round(mean_a, 4)},
        "b": {"label": label_b, "mean": round(mean_b, 4)},
        "delta": round(mean_b - mean_a, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "indistinguishable": indistinguishable,
        "b_better_on": sum(1 for d in diffs if d > 0),
        "b_worse_on": sum(1 for d in diffs if d < 0),
        "tied_on": sum(1 for d in diffs if d == 0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", default="base", help="baseline run label")
    p.add_argument("--b", default="nyaya-3b-v3", help="candidate run label")
    p.add_argument("--metrics", nargs="+",
                   default=["fact_recall", "citation_recall", "substance_recall"])
    args = p.parse_args()

    results = [compare(args.a, args.b, m) for m in args.metrics]

    print(f"\nPaired comparison — {args.b} vs {args.a}\n")
    print(f"{'metric':<20}{args.a:>10}{args.b:>14}{'delta':>10}{'95% CI':>22}  verdict")
    for r in results:
        ci = f"[{r['ci95'][0]:+.2%}, {r['ci95'][1]:+.2%}]"
        verdict = "TIED (CI spans 0)" if r["indistinguishable"] else (
            "b BETTER" if r["delta"] > 0 else "b WORSE")
        print(f"{r['metric']:<20}{r['a']['mean']:>9.1%}{r['b']['mean']:>14.1%}"
              f"{r['delta']:>+10.2%}{ci:>22}  {verdict}")

    head = results[0]
    print(f"\nper-question: {args.b} better on {head['b_better_on']}, "
          f"worse on {head['b_worse_on']}, tied on {head['tied_on']} (n={head['n']})")
    if head["indistinguishable"]:
        print(f"\nCONCLUSION: no defensible 'better than' claim. The confidence "
              f"interval on {head['metric']} includes zero, so {args.b} and "
              f"{args.a} are statistically indistinguishable on this benchmark.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {"baseline": args.a, "candidate": args.b,
         "bootstrap_rounds": BOOTSTRAP_ROUNDS, "results": results},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[compare] wrote {REPORT}")


if __name__ == "__main__":
    main()
