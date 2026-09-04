"""Pure-logic tests for the India Code API client (no network)."""
from nyaya.indiacode import clean_title, section_rows, strip_section_html, tidy_section_text, to_iso_date


def test_tidy_removes_repeated_heading_and_footnote_numbers():
    text = "1 [106. Duration of certain leases. -- (1) In the absence of a contract 2 [or local law] the lease"
    out = tidy_section_text(text, "106", "Duration of certain leases")
    assert out.startswith("(1) In the absence of a contract [or local law] the lease")
    # numbers that are part of the text survive: "(1)" and a genuine figure
    assert tidy_section_text("Whoever fails 3 times pays 10 rupees", "1", "X") == "Whoever fails 3 times pays 10 rupees"


def test_strip_section_html_keeps_text_and_fixes_footnote_period():
    raw = ('<span style="margin-left: 15px;"></span>(1) Whoever commits murder shall be punished with death, '
           'and shall also be liable to fine.<sup>1</sup>.<br/><hr style="border: none;"/>(2) When a group')
    out = strip_section_html(raw)
    assert out.startswith("(1) Whoever commits murder")
    assert "liable to fine. (2) When a group" in out
    assert "<" not in out and "&" not in out


def test_clean_title_drops_trailing_period():
    assert clean_title("Punishment for murder.") == "Punishment for murder"
    assert clean_title("  Cheating. ") == "Cheating"


def test_dates_are_normalised():
    assert to_iso_date("1-7-2024") == "2024-07-01"
    assert to_iso_date("2023-12-25") == "2023-12-25"
    assert to_iso_date(None) is None
    assert to_iso_date("not a date") is None


def test_section_rows_follow_act_order_and_drop_repealed():
    sections = [
        {"section": "3", "order": 3, "title": "Interpretation clause", "text": "In this Act ...", "repealed": False},
        {"section": "1", "order": 1, "title": "Short title", "text": "This Act may be called ...", "repealed": False},
        {"section": "2", "order": 2, "title": "[Repealed.]", "text": "", "repealed": True},
    ]
    rows = section_rows("tpa_1882", "Transfer of Property Act, 1882", sections, "1882-07-01", "https://indiacode.gov.in/handle/x")
    assert [r["section"] for r in rows] == ["1", "3"]
    assert rows[0]["act_id"] == "tpa_1882" and rows[0]["effective_date"] == "1882-07-01"
    assert set(rows[0]) == {"act_id", "act_name", "section", "title", "text", "chapter", "subsection",
                            "effective_date", "replaces", "punishment_summary", "tags", "source_url"}
