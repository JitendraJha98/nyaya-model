"""Rule-based extraction-QA generator: precision over volume, zero eval leakage."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "extraction_gen",
    Path(__file__).resolve().parents[1] / "scripts" / "19_generate_extraction_data.py")
extraction_gen = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(extraction_gen)

from nyaya.retrieval import StatuteIndex
from nyaya.validators import validate_example

ROWS = [
    {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023",
     "section": "318", "title": "Cheating",
     "text": "Whoever commits cheating shall be punished with imprisonment of either "
     "description for a term which may extend to three years, or with fine, or with both.",
     "chapter": "XVII"},
    {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023",
     "section": "999", "title": "Multi-clause provision",
     "text": "shall be punished with imprisonment which may extend to three years; "
     "and in aggravated cases with imprisonment which may extend to seven years.",
     "chapter": "X"},
    {"act_id": "rti_2005", "act_name": "Right to Information Act, 2005",
     "section": "7", "title": "Disposal of request",
     "text": "The request shall be disposed of within thirty days of the receipt "
     "of the request.", "chapter": "II"},
]
MAPPINGS = [
    {"old_act": "IPC", "old_section": "420", "new_act": "BNS",
     "new_section": "318", "note": None},
]


class TestFactExtraction:
    def test_punishment_years(self):
        assert extraction_gen.extract_punishment_years(ROWS[0]["text"]) == "three"

    def test_ambiguous_punishment_skipped(self):
        assert extraction_gen.extract_punishment_years(ROWS[1]["text"]) is None

    def test_deadline_days(self):
        assert extraction_gen.extract_deadline_days(ROWS[2]["text"]) == "thirty"

    def test_no_fact_returns_none(self):
        assert extraction_gen.extract_punishment_years("No numbers here.") is None


class TestGeneration:
    def test_generates_valid_records(self):
        index = StatuteIndex(ROWS, MAPPINGS)
        records = extraction_gen.generate_records(index, MAPPINGS, excluded=set())
        assert records, "generator produced nothing from extractable rows"
        # ipc whitelisted like the real pipeline (load_statute_db include_old_law=True)
        # — mapping answers legitimately mention the repealed section
        db = {"bns": {"318", "999"}, "rti": {"7"}, "ipc": {"420"}}
        for rec in records:
            ok, reasons = validate_example(rec, db, eval_records=[])
            assert ok, (rec["id"], reasons)

    def test_excluded_sections_produce_nothing(self):
        index = StatuteIndex(ROWS, MAPPINGS)
        excluded = {"bns_2023:318", "bns_2023:999", "rti_2005:7"}
        records = extraction_gen.generate_records(index, MAPPINGS, excluded=excluded)
        assert all(
            not (set(r["metadata"]["source_sections"]) & excluded) for r in records)
        # mapping records source from the NEW section — 318 excluded kills them too
        assert not any(r["metadata"]["task_type"] == "law_mapping" for r in records)

    def test_ids_deterministic(self):
        index = StatuteIndex(ROWS, MAPPINGS)
        a = extraction_gen.generate_records(index, MAPPINGS, excluded=set())
        b = extraction_gen.generate_records(index, MAPPINGS, excluded=set())
        assert [r["id"] for r in a] == [r["id"] for r in b]
