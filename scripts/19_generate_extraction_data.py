"""Step 19 — rule-based extraction-QA training data (v3).

v2 showed extraction is a binding constraint: with the gold section IN
context, strict accuracy tops out ~21% — the model paraphrases instead of
copying the exact duration/fine/date. These examples teach verbatim copying:
question + the section, answer restating the exact fact with the citation.

No LLM anywhere (zero budget, deterministic). Precision over volume: a section
whose text matches a pattern more than once with different values is skipped.
Sections reachable from ANY frozen-eval record (gold facts, forbidden facts,
or the question itself) are excluded — over-exclusion is the correct failure
direction.

Usage:
    python scripts/19_generate_extraction_data.py
    python scripts/19_generate_extraction_data.py --cap-per-act 150
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.evaluation import load_eval_records
from nyaya.prompts import NYAYA_SYSTEM_PROMPT
from nyaya.retrieval import _FAMILY_TO_ACT_ID, load_statute_index

DATASET_VERSION = "extraction_v1"

_PUNISH = re.compile(
    r"imprisonment[^.;]{0,120}?may extend to ([a-z\- ]+?) years?", re.I)
_DAYS = re.compile(r"within ([a-z\-]+) days", re.I)

WORD2NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
            "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30,
            "forty-five": 45, "sixty": 60, "ninety": 90}


def _unique_match(pattern: re.Pattern, text: str) -> str | None:
    """The single distinct captured value, or None (absent or ambiguous)."""
    values = {m.group(1).strip().lower() for m in pattern.finditer(text)}
    return values.pop() if len(values) == 1 else None


def extract_punishment_years(text: str) -> str | None:
    return _unique_match(_PUNISH, text)


def extract_deadline_days(text: str) -> str | None:
    return _unique_match(_DAYS, text)


def eval_excluded_keys(index, records) -> set[str]:
    """Every statute key any frozen-eval record can reach — never train
    extraction on these."""
    excluded: set[str] = set()
    for rec in records:
        domain = rec.get("legal_domain")
        for fact in rec.get("required_facts", []) + rec.get("forbidden_facts", []):
            excluded.update(index.referenced_keys(fact, domain=domain))
        excluded.update(index.referenced_keys(rec["question"], domain=domain))
    return excluded


def _mk(record_id, question, answer, task_type, language, act_id, section):
    return {
        "id": record_id,
        "messages": [
            {"role": "system", "content": NYAYA_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "language": language, "legal_domain": act_id.rsplit("_", 1)[0],
            "task_type": task_type, "source_act": act_id,
            "source_sections": [f"{act_id}:{section}"],
            "generator": "rule_extraction_v1", "verified": True,
            "dataset_version": DATASET_VERSION,
        },
    }


def _punishment_records(row, n):
    act, sec, title = row["act_name"], row["section"], row["title"]
    years = extract_punishment_years(row.get("text") or "")
    if years is None:
        return
    yield _mk(
        f"extq_{n:05d}_en_pun",
        f"What is the maximum imprisonment under Section {sec} of the {act}?",
        f"Under Section {sec} of the {act} ({title}), imprisonment may extend "
        f"to {years} years. Always check the full provision, since fines or "
        f"alternative punishments may also apply, and consult a licensed "
        f"advocate for anything consequential.",
        "extraction_qa", "english", row["act_id"], sec)
    num = WORD2NUM.get(years)
    if num is not None:
        yield _mk(
            f"extq_{n:05d}_hg_pun",
            f"Section {sec} {act} ke under maximum kitne saal ki saza ho sakti hai?",
            f"Section {sec} of the {act} ({title}) ke under imprisonment {num} "
            f"years tak extend ho sakti hai. Poora provision zaroor check karein "
            f"— fine ya alternative punishment bhi ho sakta hai. Kisi bhi "
            f"important case ke liye licensed advocate se consult karein.",
            "extraction_qa", "hinglish", row["act_id"], sec)


def _deadline_records(row, n):
    act, sec, title = row["act_name"], row["section"], row["title"]
    days = extract_deadline_days(row.get("text") or "")
    if days is None:
        return
    yield _mk(
        f"extq_{n:05d}_en_ddl",
        f"What is the time limit mentioned in Section {sec} of the {act}?",
        f"Section {sec} of the {act} ({title}) sets a time limit of {days} "
        f"days. Missing statutory time limits can be fatal to a claim, so act "
        f"promptly and consult a licensed advocate if the deadline is near.",
        "extraction_qa", "english", row["act_id"], sec)


def _mapping_records(mappings, index, excluded, start_n):
    act_names = {r["act_id"]: r["act_name"] for r in index.rows}
    n = start_n
    for m in mappings:
        new_family = m["new_act"].lower()
        new_act_id = _FAMILY_TO_ACT_ID.get(new_family)
        if not new_act_id or new_act_id not in act_names:
            continue
        key = f"{new_act_id}:{m['new_section'].upper()}"
        if key not in index.by_key or key in excluded:
            continue
        full = act_names[new_act_id]
        n += 1
        yield _mk(
            f"extq_{n:05d}_en_map",
            f"Which section of the {m['new_act']} corresponds to "
            f"{m['old_act']} Section {m['old_section']}?",
            f"{m['old_act']} Section {m['old_section']} corresponds to Section "
            f"{m['new_section']} of the {full}. The {m['old_act']} has been "
            f"repealed, so current matters should cite Section {m['new_section']} "
            f"of the {full}.",
            "law_mapping", "english", new_act_id, m["new_section"].upper())


def generate_records(index, mappings, excluded: set[str],
                     cap_per_act: int = 150) -> list[dict]:
    out, per_act = [], Counter()
    for n, row in enumerate(index.rows):
        key = f"{row['act_id']}:{row['section'].upper()}"
        if key in excluded or row["act_id"] == "procedures_kb":
            continue
        for rec in list(_punishment_records(row, n)) + list(_deadline_records(row, n)):
            if per_act[row["act_id"]] >= cap_per_act:
                break
            per_act[row["act_id"]] += 1
            out.append(rec)
    out.extend(_mapping_records(mappings, index, excluded, len(index.rows)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-per-act", type=int, default=150)
    parser.add_argument("--out", default=str(ROOT / "data" / "generated"
                                             / "extraction_qa_v1.jsonl"))
    args = parser.parse_args()

    index = load_statute_index(ROOT / "data" / "canonical")
    mappings_path = ROOT / "data" / "canonical" / "law_mappings.jsonl"
    with mappings_path.open(encoding="utf-8") as fh:
        mappings = [json.loads(line) for line in fh if line.strip()]

    excluded = eval_excluded_keys(index, load_eval_records())
    records = generate_records(index, mappings, excluded, args.cap_per_act)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_type = Counter(r["metadata"]["task_type"] for r in records)
    by_lang = Counter(r["metadata"]["language"] for r in records)
    print(json.dumps({"total": len(records), "excluded_eval_sections": len(excluded),
                      "by_task_type": dict(by_type), "by_language": dict(by_lang),
                      "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
