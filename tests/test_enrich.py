"""scripts/38_enrich_statute_db.py fills the statute-DB fields the card advertises:
`replaces` from the official mapping table, `punishment_summary` from the first
penalty clause. Until Sept 2026 both were null in every one of 2,528 rows."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("enrich", ROOT / "scripts" / "38_enrich_statute_db.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_replaces_comes_from_the_official_mapping():
    m = _mod()
    mappings = [{"old_act": "IPC", "old_section": "302", "new_act": "BNS", "new_section": "103", "note": None},
                {"old_act": "IPC", "old_section": "303", "new_act": "BNS", "new_section": "104", "note": None}]
    row = {"act_id": "bns_2023", "section": "103",
           "text": "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine."}
    out = m.enrich(row, mappings)
    assert out["replaces"] == ["IPC 302"]
    assert out["punishment_summary"] == "death or imprisonment for life, and shall also be liable to fine"


def test_several_old_sections_collapse_into_one_list():
    m = _mod()
    mappings = [{"old_act": "IPC", "old_section": s, "new_act": "BNS", "new_section": "318", "note": None}
                for s in ("415", "416", "417", "418", "419", "420")]
    out = m.enrich({"act_id": "bns_2023", "section": "318", "text": "Cheating."}, mappings)
    assert out["replaces"] == ["IPC 415", "IPC 416", "IPC 417", "IPC 418", "IPC 419", "IPC 420"]


def test_rows_without_a_penalty_clause_or_mapping_keep_null():
    m = _mod()
    out = m.enrich({"act_id": "bns_2023", "section": "2", "text": "In this Sanhita, unless the context otherwise requires"}, [])
    assert out["punishment_summary"] is None
    assert out["replaces"] is None


def test_other_acts_are_untouched_by_mappings():
    m = _mod()
    mappings = [{"old_act": "IPC", "old_section": "302", "new_act": "BNS", "new_section": "103", "note": None}]
    out = m.enrich({"act_id": "mv_act_1988", "section": "103", "text": "No penalty here"}, mappings)
    assert out["replaces"] is None
