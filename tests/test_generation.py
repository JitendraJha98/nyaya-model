"""Tests for the grounded-generation pipeline's pure logic (no teacher calls)."""

import json

import pytest

from nyaya.generation import (
    build_generation_plan,
    parse_teacher_response,
)

STATUTE_ROWS = [
    {
        "act_id": "bns_2023",
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section": str(n),
        "title": f"Offence {n}",
        "text": f"Whoever does the thing {n} shall be punished with imprisonment. " * 8,
        "chapter": "CHAPTER I",
    }
    for n in range(1, 41)
] + [
    {
        "act_id": "rti_2005",
        "act_name": "Right to Information Act, 2005",
        "section": str(n),
        "title": f"RTI provision {n}",
        "text": f"Every citizen shall have the right to information {n}. " * 8,
        "chapter": "CHAPTER II",
    }
    for n in range(1, 21)
]

MAPPINGS = [
    {"old_act": "IPC", "old_section": "420", "new_act": "BNS", "new_section": "3", "note": None},
    {"old_act": "IPC", "old_section": "302", "new_act": "BNS", "new_section": "5", "note": None},
]

COMPOSITION = {
    "grounded_qa": {"count": 12, "language": "english"},
    "hindi_qa": {"count": 4, "language": "hindi"},
    "hinglish_qa": {"count": 4, "language": "hinglish"},
    "law_mapping": {"count": 2, "language": "english"},
    "safety_abstention": {"count": 2, "language": "english"},
}


class TestBuildGenerationPlan:
    def _plan(self, seed=7):
        return build_generation_plan(
            STATUTE_ROWS, MAPPINGS, COMPOSITION, seed=seed
        )

    def test_counts_match_composition(self):
        plan = self._plan()
        by_type = {}
        for t in plan:
            by_type[t["task_type"]] = by_type.get(t["task_type"], 0) + 1
        assert by_type == {k: v["count"] for k, v in COMPOSITION.items()}

    def test_grounded_tasks_carry_verbatim_statute_text(self):
        plan = self._plan()
        grounded = [t for t in plan if t["task_type"] == "grounded_qa"]
        for t in grounded:
            assert t["source_sections"], "grounded task must record its split key"
            # verbatim statute text present in the prompt
            key = t["source_sections"][0]
            act_id, section = key.split(":")
            row = next(r for r in STATUTE_ROWS if r["act_id"] == act_id and r["section"] == section)
            assert row["text"][:60] in t["prompt"]

    def test_language_directives(self):
        plan = self._plan()
        assert any("Hinglish" in t["prompt"] for t in plan if t["task_type"] == "hinglish_qa")
        assert all(t["language"] == "hindi" for t in plan if t["task_type"] == "hindi_qa")

    def test_mapping_tasks_reference_both_acts(self):
        plan = self._plan()
        for t in plan:
            if t["task_type"] == "law_mapping":
                assert "IPC" in t["prompt"] and "BNS" in t["prompt"]

    def test_deterministic_for_seed(self):
        a = [t["task_id"] for t in self._plan(seed=7)]
        b = [t["task_id"] for t in self._plan(seed=7)]
        c = [t["task_id"] for t in self._plan(seed=8)]
        assert a == b
        assert a != c

    def test_safety_tasks_are_not_statute_grounded(self):
        plan = self._plan()
        for t in plan:
            if t["task_type"] == "safety_abstention":
                assert t["source_sections"] == []

    def test_task_ids_unique(self):
        ids = [t["task_id"] for t in self._plan()]
        assert len(set(ids)) == len(ids)


class TestParseTeacherResponse:
    def _task(self):
        return {
            "task_id": "gen_000001",
            "task_type": "grounded_qa",
            "language": "english",
            "source_act": "bns_2023",
            "source_sections": ["bns_2023:318"],
        }

    def test_parses_json_array(self):
        raw = json.dumps([
            {"question": "What is cheating?", "answer": "Section 318 BNS covers cheating."},
            {"question": "Kya hoga?", "answer": "Saza hogi."},
        ])
        records = parse_teacher_response(raw, self._task(), dataset_version="v1")
        assert len(records) == 2
        r = records[0]
        assert r["messages"][0]["role"] == "system"
        assert r["messages"][1] == {"role": "user", "content": "What is cheating?"}
        assert r["messages"][2]["role"] == "assistant"
        assert r["metadata"]["source_sections"] == ["bns_2023:318"]
        assert r["metadata"]["task_type"] == "grounded_qa"
        assert r["metadata"]["dataset_version"] == "v1"
        assert r["id"].startswith("gen_000001")

    def test_parses_fenced_json(self):
        raw = "Here you go:\n```json\n" + json.dumps(
            [{"question": "Q1?", "answer": "A1."}]
        ) + "\n```\nHope that helps!"
        records = parse_teacher_response(raw, self._task(), dataset_version="v1")
        assert len(records) == 1

    def test_rejects_garbage(self):
        assert parse_teacher_response("I cannot help with that.", self._task(), "v1") == []

    def test_skips_items_missing_fields(self):
        raw = json.dumps([
            {"question": "ok?", "answer": "yes"},
            {"question": "missing answer"},
            {"answer": "missing question"},
        ])
        records = parse_teacher_response(raw, self._task(), "v1")
        assert len(records) == 1
