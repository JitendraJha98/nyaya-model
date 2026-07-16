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


class TestEvalExclusion:
    def test_bare_old_law_reference_excludes_mapped_new_section(self):
        # eval facts often write the repealed section WITHOUT a "Section" marker
        # ("IPC 124A", "CrPC 41A"); the citation parser needs the marker, so
        # eval_excluded_keys must also resolve bare old-law forms through the
        # official mapping — else a mapping record leaks a section the eval tests.
        rows = [{"act_id": "bns_2023",
                 "act_name": "Bharatiya Nyaya Sanhita, 2023", "section": "151",
                 "title": "Acts endangering sovereignty",
                 "text": "Whoever endangers sovereignty shall be punished.",
                 "chapter": "VII"}]
        mappings = [{"old_act": "IPC", "old_section": "124A", "new_act": "BNS",
                     "new_section": "151", "note": None}]
        idx = StatuteIndex(rows, mappings)
        recs = [{"legal_domain": "bns",
                 "required_facts": ["IPC 124A sedition deleted"],
                 "forbidden_facts": [], "question": "What happened to IPC 124A?"}]
        excl = extraction_gen.eval_excluded_keys(idx, recs)
        assert "bns_2023:151" in excl

    def test_reversed_bare_old_law_reference_excluded(self):
        # the number-then-act order ("498A IPC") also appears unmarked in eval
        # forbidden_facts — it must be caught too
        rows = [{"act_id": "bns_2023",
                 "act_name": "Bharatiya Nyaya Sanhita, 2023", "section": "85",
                 "title": "Cruelty by husband or relative",
                 "text": "Whoever subjects a woman to cruelty shall be punished.",
                 "chapter": "XI"}]
        mappings = [{"old_act": "IPC", "old_section": "498A", "new_act": "BNS",
                     "new_section": "85", "note": None}]
        idx = StatuteIndex(rows, mappings)
        recs = [{"legal_domain": "bns", "required_facts": [],
                 "forbidden_facts": ["498A IPC as current law"],
                 "question": "Is 498A IPC still the law?"}]
        excl = extraction_gen.eval_excluded_keys(idx, recs)
        assert "bns_2023:85" in excl

    def test_bare_reference_blocks_the_mapping_record(self):
        rows = [{"act_id": "bns_2023",
                 "act_name": "Bharatiya Nyaya Sanhita, 2023", "section": "151",
                 "title": "Acts endangering sovereignty",
                 "text": "Whoever endangers sovereignty shall be punished.",
                 "chapter": "VII"}]
        mappings = [{"old_act": "IPC", "old_section": "124A", "new_act": "BNS",
                     "new_section": "151", "note": None}]
        idx = StatuteIndex(rows, mappings)
        recs = [{"legal_domain": "bns", "required_facts": ["IPC 124A"],
                 "forbidden_facts": [], "question": "IPC 124A?"}]
        excluded = extraction_gen.eval_excluded_keys(idx, recs)
        out = extraction_gen.generate_records(idx, mappings, excluded=excluded)
        assert not any(r["metadata"]["task_type"] == "law_mapping" for r in out)


class TestMappingCap:
    def test_mapping_records_capped_per_new_act(self):
        rows = [{"act_id": "bns_2023",
                 "act_name": "Bharatiya Nyaya Sanhita, 2023", "section": str(s),
                 "title": f"Provision {s}", "text": "Whoever offends is punished.",
                 "chapter": "X"} for s in range(1, 21)]
        mappings = [{"old_act": "IPC", "old_section": str(100 + s), "new_act": "BNS",
                     "new_section": str(s), "note": None} for s in range(1, 21)]
        idx = StatuteIndex(rows, mappings)
        out = extraction_gen.generate_records(idx, mappings, excluded=set(),
                                              cap_per_act=5)
        mapped = [r for r in out if r["metadata"]["task_type"] == "law_mapping"]
        assert len(mapped) == 5  # capped, not all 20

    def test_constitution_deadline_skipped(self):
        rows = [{"act_id": "constitution_1950", "act_name": "Constitution of India",
                 "section": "352", "title": "Proclamation of Emergency",
                 "text": "The Proclamation shall be laid before each House within "
                 "thirty days.", "chapter": "XVIII"}]
        idx = StatuteIndex(rows, [])
        out = extraction_gen.generate_records(idx, [], excluded=set())
        assert not any("time limit" in r["messages"][1]["content"] for r in out)
