"""Step 29 — RAG-grounded citation training data (Milestone 2).

The measured failure this targets
---------------------------------
On Nyaya-Eval-v1, with the gold statute ALREADY in the model's context, base
Qwen2.5-3B still missed 71 required facts:

    29  wrong or missing citation   (41%)
    15  missing the exact number    (21%)
    18  paraphrased the fact away   (25%)
     9  partial                     (13%)

So the bottleneck is not knowledge and not retrieval — it is that the model
will not COPY what it was handed. It reads "shall be punished with
imprisonment for a term which may extend to three years" and writes "a few
years"; it sees Section 103(2) in context and cites Section 103.

Why v3's training could not fix this
------------------------------------
v3 taught "cite sections in the right format" from questions that NAMED the
section ("What is the punishment under Section 13 of the BNS?"). Under RAG the
base model already does that — the statute is right there — which is exactly
why v3 measured as a statistical tie with base.

This generator trains the skill that is actually missing: given a realistic
retrieved context containing the right section AND competing distractors,
select the correct one and reproduce its number verbatim.

Contamination
-------------
Sections reachable from any frozen-eval record are excluded, reusing
scripts/19's eval_excluded_keys. Distractors are drawn from the real retriever,
so a distractor could in principle be eval-reachable — it is only ever context,
never the answer, but the gold section itself is always eval-free.

Usage:
    python scripts/29_build_grounding_data.py
    python scripts/29_build_grounding_data.py --k 8 --cap-per-act 400
"""

import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.evaluation import load_eval_records  # noqa: E402
from nyaya.prompts import NYAYA_SYSTEM_PROMPT  # noqa: E402
from nyaya.retrieval import build_rag_prompt, load_statute_index  # noqa: E402

OUT = ROOT / "data" / "generated" / "grounding_v1.jsonl"
REPORT = ROOT / "reports" / "grounding_dataset.json"

# The two things the model demonstrably fails to copy.
_YEARS = re.compile(
    r"(?:imprisonment|term)[^.]{0,80}?may extend to (\w+(?:[- ]\w+)?) years?", re.I)
_MIN_YEARS = re.compile(
    r"not be less than (\w+(?:[- ]\w+)?) years?", re.I)
_FINE = re.compile(r"fine which may extend to (?:rupees\s+)?([\w,]+)(?:\s+rupees)?", re.I)
_DAYS = re.compile(r"within (\w+(?:[- ]\w+)?) days", re.I)


def _one(pattern: re.Pattern, text: str) -> str | None:
    """A value only if it is UNAMBIGUOUS in the section.

    Two different durations in one section means the correct answer depends on
    sub-clauses this generator cannot parse, and a wrong label teaches the
    model to guess. Precision over volume.
    """
    found = {m.group(1).strip().lower() for m in pattern.finditer(text)}
    return found.pop() if len(found) == 1 else None


def _eval_excluded(index, records) -> set[str]:
    spec = importlib.util.spec_from_file_location(
        "gen19", ROOT / "scripts" / "19_generate_extraction_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_excluded_keys(index, records)


def _act_phrase(row: dict) -> str:
    return f"Section {row['section']} of the {row['act_name']}"


def _questions(row: dict) -> list[tuple[str, str, str]]:
    """(question, answer_fact, subtype) for whatever this section states clearly.

    Questions deliberately do NOT name the section — that is the whole point.
    The model must locate it in the retrieved context.
    """
    text = row.get("text") or ""
    title = (row.get("title") or "").strip().rstrip(".")
    if not title:
        return []
    out = []

    years = _one(_YEARS, text)
    if years:
        out.append((f"What is the maximum imprisonment for {title.lower()}?",
                    f"imprisonment which may extend to {years} years", "max_years"))
    minimum = _one(_MIN_YEARS, text)
    if minimum:
        out.append((f"Is there a minimum sentence for {title.lower()}?",
                    f"imprisonment of not less than {minimum} years", "min_years"))
    fine = _one(_FINE, text)
    if fine:
        out.append((f"What fine applies for {title.lower()}?",
                    f"a fine which may extend to {fine}", "fine"))
    days = _one(_DAYS, text)
    if days:
        out.append((f"What is the time limit regarding {title.lower()}?",
                    f"within {days} days", "deadline"))

    # Citation selection — the single largest failure mode (29 of 71 missed
    # facts, 41%). Needs no numeric pattern, so it applies to EVERY section
    # rather than the ~7% that state an unambiguous duration or fine. The
    # answer's fact is the section's own opening clause, kept verbatim, since
    # copying is precisely the skill being taught.
    opening = _first_clause(text)
    if opening:
        out.append((f"Which provision of Indian law covers {title.lower()}?",
                    opening, "which_section"))
    return out


_SENT_END = re.compile(r"(?<=[.;])\s+")


def _first_clause(text: str, max_words: int = 45) -> str | None:
    """The section's opening clause, verbatim and trimmed to a usable length."""
    body = (text or "").strip()
    if not body:
        return None
    clause = _SENT_END.split(body)[0].strip().rstrip(".;")
    words = clause.split()
    if len(words) < 4:
        return None
    if len(words) > max_words:
        clause = " ".join(words[:max_words])
    return clause


def build(index, excluded: set[str], k: int, cap_per_act: int,
          seed: int = 0) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    rows = [r for r in index.rows if r.get("act_id") != "procedures_kb"]
    rng.shuffle(rows)

    records, per_act, stats = [], Counter(), Counter()
    for row in rows:
        key = f"{row['act_id']}:{row['section'].upper()}"
        if key in excluded:
            stats["skipped_eval_reachable"] += 1
            continue
        if per_act[row["act_id"]] >= cap_per_act:
            continue

        for question, fact, subtype in _questions(row):
            hits = index.retrieve(question, k=k)
            hit_keys = [f"{h['act_id']}:{h['section'].upper()}" for h in hits]

            # The gold section MUST be in context — this teaches selection
            # among distractors, not recall from parameters. If the retriever
            # cannot surface it, that is a retrieval problem, not a training
            # example, and a fabricated context would train on a distribution
            # the model never sees at serving time.
            if key not in hit_keys:
                stats["skipped_gold_not_retrieved"] += 1
                continue
            if len(hits) < 2:
                stats["skipped_no_distractors"] += 1
                continue

            # Lowercase the first letter of a verbatim clause so it reads as a
            # sentence continuation ("Under Section 92 ..., when any person
            # ..."), while leaving acronyms and proper nouns alone.
            body = fact
            if body[:1].isupper() and not body.split()[0].isupper():
                body = body[0].lower() + body[1:]
            answer = (f"Under {_act_phrase(row)} ({row['title']}), {body}. "
                      f"This is legal information, not legal advice — consult a "
                      f"licensed advocate for anything consequential.")
            records.append({
                "id": f"grnd_{len(records):05d}_{subtype}",
                "messages": [
                    {"role": "system", "content": NYAYA_SYSTEM_PROMPT},
                    {"role": "user", "content": build_rag_prompt(question, hits)},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "language": "english",
                    "legal_domain": row["act_id"],
                    "task_type": "rag_grounding",
                    "subtype": subtype,
                    "source_act": row["act_id"],
                    "source_sections": [key],
                    "distractors": len(hits) - 1,
                    "generator": "rule_grounding_v1",
                    "verified": True,
                },
            })
            per_act[row["act_id"]] += 1
            stats[subtype] += 1

    return records, {
        "total": len(records),
        "by_subtype": dict(stats),
        "by_act": dict(per_act),
        "k": k,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--cap-per-act", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    index = load_statute_index(str(ROOT / "data" / "canonical"))
    excluded = _eval_excluded(index, load_eval_records())
    print(f"[grounding] {len(excluded)} eval-reachable sections excluded")

    records, report = build(index, excluded, args.k, args.cap_per_act, args.seed)

    # Leakage assertion: no generated answer may cite an eval-reachable section.
    leaked = [r for r in records if set(r["metadata"]["source_sections"]) & excluded]
    if leaked:
        sys.exit(f"[grounding] LEAK: {len(leaked)} records cite eval sections")
    report["leakage_checked"] = True

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[grounding] wrote {OUT}")


if __name__ == "__main__":
    main()
