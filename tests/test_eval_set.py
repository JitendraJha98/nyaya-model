"""Validates Nyaya-Eval-v0 against the contract in data/eval/README.md.

Applies to the draft (nyaya_eval_v0_draft.jsonl) while under review and to the
frozen file (nyaya_eval_v0.jsonl) once blessed — whichever exists, frozen wins.

Category counting rule (one category per record, matching the README table):
language 'hindi'/'hinglish' rows count as the Hindi/Hinglish categories,
task_type 'safety_abstention' counts as the Safety category, and every other
record counts under its legal_domain.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "eval" / "nyaya_eval_v0.jsonl"
DRAFT = ROOT / "data" / "eval" / "nyaya_eval_v0_draft.jsonl"

# data/eval/README.md target category split
EXPECTED_COUNTS = {
    "bns": 70,
    "bnss": 70,
    "bsa": 40,
    "constitutional_law": 50,
    "consumer_law": 40,
    "cyber_law": 40,
    "rti": 30,
    "womens_protection": 30,
    "cheque_bounce": 30,
    "motor_vehicles": 25,
    "labour_law": 25,
    "old_new_law_mapping": 25,
    "hindi": 10,
    "hinglish": 10,
    "safety": 5,
}

LANGUAGES = {"english", "hindi", "hinglish"}
DIFFICULTIES = {"easy", "medium", "hard"}
TASK_TYPES = {
    "legal_qa",
    "procedural_guidance",
    "scenario",
    "terminology",
    "old_new_law_mapping",
    "safety_abstention",
}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def category(record):
    if record["language"] == "hindi":
        return "hindi"
    if record["language"] == "hinglish":
        return "hinglish"
    if record["task_type"] == "safety_abstention":
        return "safety"
    return record["legal_domain"]


@pytest.fixture(scope="module")
def records():
    path = FROZEN if FROZEN.exists() else DRAFT
    if not path.exists():
        pytest.fail(f"eval set not found: expected {FROZEN.name} or {DRAFT.name}")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_total_count(records):
    assert len(records) == 500


def test_ids_unique_and_well_formed(records):
    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"nyaya_eval_\d{6}", i) for i in ids)


def test_schema_fields(records):
    for r in records:
        assert isinstance(r["question"], str) and r["question"].strip()
        assert isinstance(r["expected_answer"], str) and r["expected_answer"].strip()
        assert r["language"] in LANGUAGES
        assert isinstance(r["legal_domain"], str) and r["legal_domain"]
        assert r["task_type"] in TASK_TYPES
        assert r["difficulty"] in DIFFICULTIES
        assert r["split"] == "test"
        assert isinstance(r["required_facts"], list)
        assert isinstance(r["forbidden_facts"], list)
        assert isinstance(r["source"], str) and r["source"].strip()


def test_category_split_matches_readme(records):
    counts = {}
    for r in records:
        counts[category(r)] = counts.get(category(r), 0) + 1
    assert counts == EXPECTED_COUNTS


def test_questions_unique(records):
    normalized = [" ".join(r["question"].lower().split()) for r in records]
    assert len(set(normalized)) == len(normalized)


def test_required_facts_present_except_safety(records):
    for r in records:
        if r["task_type"] != "safety_abstention":
            assert r["required_facts"], f"{r['id']} has no required_facts"


def test_language_script_consistency(records):
    for r in records:
        has_devanagari = bool(DEVANAGARI.search(r["question"]))
        if r["language"] == "hindi":
            assert has_devanagari, f"{r['id']} marked hindi but not Devanagari"
        elif r["language"] == "hinglish":
            assert not has_devanagari, f"{r['id']} marked hinglish but uses Devanagari"
