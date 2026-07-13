"""Tests for the statute-corpus extraction pipeline (pure logic, no network)."""

from nyaya.corpus import slice_act_body, split_sections, validate_sections

# Mimics India Code gazette layout after PDF text extraction: ToC first
# (must be excluded), enacting formula, chapters, sections with ".——" title
# separators, subsections, an Explanation, and a letter-suffixed section.
FIXTURE = """\
THE TEST ACT, 2023
ARRANGEMENT OF SECTIONS
SECTIONS
1. Short title.
2. Definitions.
3. Offence of testing.
3A. Aggravated testing.
4. Punishment.

THE TEST ACT, 2023
ACT NO. 99 OF 2023
An Act to test extraction.
BE it enacted by Parliament in the Seventy-fourth Year of the Republic of India as follows:——
CHAPTER I
PRELIMINARY
1. Short title.——(1) This Act may be called the Test Act, 2023.
(2) It shall come into force at once.
2. Definitions.——In this Act, "test" means a trial of quality.
CHAPTER II
OFFENCES
3. Offence of testing.——Whoever tests without authority commits an offence.
Explanation.——Testing includes re-testing.
3A. Aggravated testing.——(1) Whoever tests repeatedly shall be liable to enhanced punishment.
(2) The punishment may extend to five years.
4. Punishment of person guilty of one of several offences, judgment stating that it is doubtful
of which.—Whoever commits an offence under section 3 shall be punished with fine.
5. False information given to a co-operative testing authority—Whoever gives false information
commits an offence under this section.
6.—Repeated testing without disclosure.—Whoever tests repeatedly without disclosure shall be
punished with community service.
THE SCHEDULE
[See section 4]
Forms and procedures.
"""


class TestSliceActBody:
    def test_starts_at_enacting_formula(self):
        body = slice_act_body(FIXTURE)
        assert body.lstrip().startswith("BE it enacted")

    def test_toc_excluded(self):
        body = slice_act_body(FIXTURE)
        assert "ARRANGEMENT OF SECTIONS" not in body

    def test_schedule_excluded(self):
        body = slice_act_body(FIXTURE)
        assert "THE SCHEDULE" not in body
        assert "Forms and procedures" not in body


class TestSplitSections:
    def _sections(self):
        return split_sections(slice_act_body(FIXTURE))

    def test_all_sections_found(self):
        assert [s["section"] for s in self._sections()] == ["1", "2", "3", "3A", "4", "5", "6"]

    def test_titles_extracted(self):
        sections = {s["section"]: s for s in self._sections()}
        assert sections["1"]["title"] == "Short title"
        assert sections["3A"]["title"] == "Aggravated testing"

    def test_line_wrapped_title_with_single_dash(self):
        # Real gazette PDFs wrap long titles across lines and sometimes use a
        # single em-dash separator.
        sections = {s["section"]: s for s in self._sections()}
        assert sections["4"]["title"].endswith("doubtful of which")
        assert sections["4"]["text"].startswith("Whoever commits")

    def test_gazette_typo_missing_period_before_dash(self):
        # BNS s.217 in the official PDF: "...another person—Whoever" (no period).
        # The hyphen in "co-operative" must NOT terminate the title.
        sections = {s["section"]: s for s in self._sections()}
        assert sections["5"]["title"].endswith("co-operative testing authority")
        assert sections["5"]["text"].startswith("Whoever gives false information")

    def test_gazette_typo_dash_after_number(self):
        # BNS s.255 in the official PDF: "255.—Public servant...".
        sections = {s["section"]: s for s in self._sections()}
        assert sections["6"]["title"] == "Repeated testing without disclosure"
        assert sections["6"]["text"].startswith("Whoever tests repeatedly")

    def test_body_text_verbatim_with_subsections(self):
        sections = {s["section"]: s for s in self._sections()}
        assert "(1) This Act may be called the Test Act, 2023." in sections["1"]["text"]
        assert "(2) It shall come into force at once." in sections["1"]["text"]

    def test_explanation_stays_inside_its_section(self):
        sections = {s["section"]: s for s in self._sections()}
        assert "Explanation" in sections["3"]["text"]
        assert "re-testing" in sections["3"]["text"]

    def test_chapters_attached(self):
        sections = {s["section"]: s for s in self._sections()}
        assert sections["1"]["chapter"].startswith("CHAPTER I")
        assert sections["3"]["chapter"].startswith("CHAPTER II")

    def test_toc_style_lines_do_not_create_sections(self):
        # A ToC line like "3. Offence of testing." has no em-dash separator —
        # feeding un-sliced text must not double-count sections.
        sections = split_sections(FIXTURE)
        assert [s["section"] for s in sections] == ["1", "2", "3", "3A", "4", "5", "6"]


class TestValidateSections:
    def test_clean_extraction_passes(self):
        report = validate_sections(split_sections(slice_act_body(FIXTURE)), expected_count=7)
        assert report["extracted"] == 7
        assert report["expected"] == 7
        assert report["monotonic"] is True
        assert report["numbering_gaps"] == []
        assert report["empty_or_short"] == []
        assert report["clean_fraction"] == 1.0

    def test_gap_detected(self):
        sections = [s for s in split_sections(slice_act_body(FIXTURE)) if s["section"] != "2"]
        report = validate_sections(sections, expected_count=7)
        assert report["numbering_gaps"] == ["2"]
        assert report["clean_fraction"] < 1.0

    def test_out_of_order_detected(self):
        sections = split_sections(slice_act_body(FIXTURE))
        sections[0], sections[1] = sections[1], sections[0]
        report = validate_sections(sections)
        assert report["monotonic"] is False

    def test_short_text_flagged(self):
        sections = split_sections(slice_act_body(FIXTURE))
        sections[0]["text"] = "stub"
        report = validate_sections(sections)
        assert report["empty_or_short"] == ["1"]
