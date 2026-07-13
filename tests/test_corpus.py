"""Tests for the statute-corpus extraction pipeline (pure logic, no network)."""

from nyaya.corpus import (
    slice_act_body,
    split_articles,
    split_sections,
    validate_sections,
)

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

    def test_pre_constitution_enacting_formula(self):
        # Pre-1950 acts (NI Act 1881) enact via "It is hereby enacted as
        # follows" — not "BE it enacted by Parliament". The ToC must still
        # be trimmed or its entries poison section detection.
        text = (
            "ARRENGMENT OF SECTIONS\n"
            "1. Short title.\n"
            "21. “At sight”.\n"
            "Preamble.—Whereas it is expedient to define the law; "
            "It is hereby enacted as follows:—\n"
            "1. Short title.—This Act may be called the Test Act, 1881.\n"
        )
        body = slice_act_body(text)
        assert "ARRENGMENT" not in body
        assert body.startswith("It is hereby enacted")

    def test_bracketed_schedule_heading_terminates(self):
        # IT Act 2000: an inserted-by-amendment schedule prints as
        # "[THE FIRST SCHEDULE" — leading bracket must not defeat the anchor.
        text = (
            "BE it enacted by Parliament as follows:—\n"
            "1. Short title.—This Act may be called the Test Act.\n"
            "[THE FIRST SCHEDULE\n[See section 1]\nSchedule content here.\n"
        )
        body = slice_act_body(text)
        assert "Schedule content" not in body

    def test_statement_of_objects_excluded(self):
        # India Code "updated" PDFs append the Statement of Objects and
        # Reasons after the last section (Code on Wages 2019).
        text = (
            "BE it enacted by Parliament as follows:—\n"
            "1. Short title.—This Code may be called the Test Code.\n"
            "STATEMENT OF OBJECTS AND REASONS\n"
            "The salient features of the Code, inter alia, are as follows.\n"
        )
        body = slice_act_body(text)
        assert "salient features" not in body
        assert "Short title" in body


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

    def test_horizontal_bar_separator(self):
        # Older acts re-typeset by India Code (SMA 1954, HMA 1955 "As on"
        # snapshots) use U+2015 HORIZONTAL BAR, not an em dash, after titles:
        # "1. Short title, extent and commencement.―(1) This Act..."
        text = (
            "BE it enacted by Parliament as follows:―\n"
            "1. Short title.―(1) This Act may be called the Test Act.\n"
            "2. Definitions.―In this Act, unless the context otherwise requires,―\n"
            "(a) a term means what it says.\n"
        )
        sections = split_sections(text)
        assert [s["section"] for s in sections] == ["1", "2"]
        assert sections[0]["title"] == "Short title"
        assert sections[1]["text"].startswith("In this Act")

    def test_omitted_section_and_following_section(self):
        # HMA 1955 s.6: "6. [Guardianship in marriage.] Omitted by ..." — no
        # dash separator. Must extract as its own section AND must not swallow
        # the following section's line into a phantom title.
        text = (
            "BE it enacted by Parliament as follows:—\n"
            "5. Conditions.—A marriage may be solemnized.\n"
            "6. [Guardianship in marriage.] Omitted by the Amendment Act, 1978 "
            "(2 of 1978), s. 6 and Sch. (w.e.f. 1-10-1978). \n"
            "7. Ceremonies.—(1) A marriage may be solemnized in accordance "
            "with customary rites.\n"
        )
        sections = split_sections(text)
        assert [s["section"] for s in sections] == ["5", "6", "7"]
        assert sections[1]["title"] == "Guardianship in marriage"
        assert sections[1]["text"].startswith("Omitted by")
        assert sections[2]["title"] == "Ceremonies"

    def test_bracket_substituted_section(self):
        # Wholly substituted sections print as "[16. Title.—..." after the
        # extractor strips the footnote superscript (HMA 16/19/22).
        text = (
            "BE it enacted by Parliament as follows:—\n"
            "15. Divorced persons.—When a marriage has been dissolved.\n"
            "[16. Legitimacy of children.—(1) Notwithstanding the decree.]\n"
            "17. Punishment.—Any marriage between two Hindus.\n"
        )
        sections = split_sections(text)
        assert [s["section"] for s in sections] == ["15", "16", "17"]
        assert sections[1]["title"] == "Legitimacy of children"

    def test_space_before_period_in_section_number(self):
        # Code on Wages 2019 s.39 is typeset " 39 .Time limit…" — a stray
        # space between the number and the period.
        text = (
            "BE it enacted by Parliament as follows:—\n"
            "38. Deduction of certain amounts from bonus payable.—Where in any "
            "accounting year an employee is found guilty.\n"
            " 39 .Time limit for payment of bonus.—(1) All amounts payable to "
            "an employee by way of bonus shall be paid within eight months.\n"
        )
        sections = split_sections(text)
        assert [s["section"] for s in sections] == ["38", "39"]
        assert sections[1]["title"] == "Time limit for payment of bonus"

    def test_duplicate_number_footnote_block_dropped(self):
        # Commencement-notification footnotes print as "1. 18 December,
        # 2020.—Sub-section (1)…" at a page bottom, mid-act — a phantom
        # duplicate of section 1. First occurrence wins; later dupes drop.
        text = (
            "BE it enacted by Parliament as follows:—\n"
            "1. Short title.—This Code may be called the Test Code, 2019.\n"
            "2. Definitions.—In this Code, wages means all remuneration.\n"
            "1. 18 December, 2020.—Sub-section (1), (2) and (3) of section 42 "
            "vide notification number S.O. 4604(E).\n"
            "3. Payment.—Wages shall be paid in current coin.\n"
        )
        sections = split_sections(text)
        assert [s["section"] for s in sections] == ["1", "2", "3"]
        assert sections[0]["title"] == "Short title"

    def test_toc_style_lines_do_not_create_sections(self):
        # A ToC line like "3. Offence of testing." has no em-dash separator —
        # feeding un-sliced text must not double-count sections.
        sections = split_sections(FIXTURE)
        assert [s["section"] for s in sections] == ["1", "2", "3", "3A", "4", "5", "6"]


CONSTITUTION_SNIPPET = """PART I
THE UNION AND ITS TERRITORY
1. Name and territory of the Union.—(1) India, that is Bharat, shall be a
Union of States.
(2) The States and the territories thereof shall be as specified in the First
Schedule.
2. Admission or establishment of new States.—Parliament may by law admit
into the Union, or establish, new States.
PART III
FUNDAMENTAL RIGHTS
21. Protection of life and personal liberty.—No person shall be deprived of
his life or personal liberty except according to procedure established by law.
21A. Right to education.—The State shall provide free and compulsory
education to all children of the age of six to fourteen years.
"""


class TestSplitArticles:
    def test_splits_articles_with_part_attribution(self):
        arts = split_articles(CONSTITUTION_SNIPPET)
        nums = [a["section"] for a in arts]
        assert nums == ["1", "2", "21", "21A"]
        assert arts[0]["title"] == "Name and territory of the Union"
        assert "Union of States" in arts[0]["text"]
        assert arts[0]["chapter"] == "PART I — The Union And Its Territory"
        assert arts[3]["chapter"] == "PART III — Fundamental Rights"

    def test_em_dash_separates_title_from_body(self):
        arts = split_articles(CONSTITUTION_SNIPPET)
        # title must not swallow the body after the em dash
        assert arts[2]["title"] == "Protection of life and personal liberty"
        assert arts[2]["text"].startswith("No person shall be deprived")


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
