"""Tests for NCRB corresponding-section table parsing (pure logic)."""

from nyaya.corpus import extract_mapping_pairs

# Mimics rows from fitz find_tables() on the NCRB Sankalan "Corresponding
# Section Table": (new-act cell, old-act cell); cells are multi-line, rows can
# be headers, chapter markers, continuations (empty left cell), or contain
# "New Section" markers with no old-law counterpart.
ROWS = [
    ("Bharatiya Nyaya Sanhita,\n2023", "Indian Penal Code, 1860"),
    ("CHAPTER I – PRELIMINARY", "CHAPTER I – INTRODUCTION"),
    (
        "1. Short title, commencement and\napplication\n1(1)\n1(2)\n1(3)",
        "1. Title and extent of operation of the\nCode.\nNew Section\n2. Punishment of offences committed\nwithin India.",
    ),
    ("2. Definitions.", ""),
    ("", "33. “Act”. “Omission”\n47. “Animal”."),
    ("103. Punishment for murder.", "302. Punishment for murder."),
    ("111. Organised crime.", "New Section"),
    (
        "115. Voluntarily causing hurt.",
        "321. Voluntarily causing hurt.\n323. Punishment for voluntarily causing\nhurt.",
    ),
    # Consolidating sections appear as subsection-level lines in the PDF:
    # "318 (1)" / "319(1)" set the current section just like "318." does.
    ("318. Cheating.", ""),
    ("318 (1)", "415. Cheating."),
    ("318 (4)", "420. Cheating and dishonestly inducing delivery of property."),
    ("319(1)", "416. Cheating by personation."),
    ("319(2)", "419. Punishment of cheating by personation."),
    # Right-column numbers whose title wrapped to the next line are bare
    # "N." lines with nothing after the period.
    ("85. Husband or relative of husband of", "498A."),
    # Letter-suffixed old sections sometimes omit the period entirely,
    # and left cells can carry several subsection markers on one line.
    ("35(3), 35(4) 35(5), 35(6)", "41A Notice of appearance before"),
]


class TestExtractMappingPairs:
    def test_simple_pair(self):
        pairs = extract_mapping_pairs(ROWS)
        assert ("103", "302") in pairs

    def test_multiple_old_sections_fan_out(self):
        pairs = extract_mapping_pairs(ROWS)
        assert ("115", "321") in pairs
        assert ("115", "323") in pairs

    def test_continuation_rows_attach_to_previous_section(self):
        pairs = extract_mapping_pairs(ROWS)
        assert ("2", "33") in pairs
        assert ("2", "47") in pairs

    def test_new_sections_produce_no_pair(self):
        pairs = extract_mapping_pairs(ROWS)
        assert not any(new == "111" for new, _ in pairs)

    def test_header_and_chapter_rows_ignored(self):
        pairs = extract_mapping_pairs(ROWS)
        new_sections = {new for new, _ in pairs}
        assert "2023" not in new_sections
        assert all(not n.startswith("CHAPTER") for n in new_sections)

    def test_multiline_cell_uses_first_number_as_section(self):
        pairs = extract_mapping_pairs(ROWS)
        # left cell "1. Short title...\n1(1)\n1(2)" is BNS section 1
        assert ("1", "1") in pairs
        assert ("1", "2") in pairs

    def test_subsection_markers_set_current_section(self):
        pairs = extract_mapping_pairs(ROWS)
        assert ("318", "415") in pairs
        assert ("318", "420") in pairs

    def test_subsection_marker_switches_section(self):
        # "319(1)" must switch context away from 318 so 416/419 attach to 319.
        pairs = extract_mapping_pairs(ROWS)
        assert ("319", "416") in pairs
        assert ("319", "419") in pairs
        assert ("318", "416") not in pairs

    def test_bare_number_line_pairs(self):
        assert ("85", "498A") in extract_mapping_pairs(ROWS)

    def test_letter_suffix_without_period(self):
        assert ("35", "41A") in extract_mapping_pairs(ROWS)

    def test_plain_number_without_period_is_not_a_section(self):
        # "41 Notice" (no period, no letter suffix) must not match — too
        # easily confused with prose/wrapped titles.
        pairs = extract_mapping_pairs([("103. Murder.", "302 some wrapped title text")])
        assert pairs == []
