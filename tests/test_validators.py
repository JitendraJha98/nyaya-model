"""Tests for the citation-extraction regex — the citation gate is the primary metric,
so both failure directions matter: a false positive wrongly rejects a good example,
a false negative lets a hallucinated citation through unverified.
"""

from nyaya.validators import extract_citations


class TestExtractsRealCitations:
    def test_standard_form(self):
        assert extract_citations("You may be charged under Section 318 of the BNS.") == ["Section 318"]

    def test_abbreviated_form(self):
        assert extract_citations("See Sec. 173 for FIR procedure.") == ["Sec. 173"]

    def test_section_symbol_without_space(self):
        assert extract_citations("Cheating is covered by §420 of the IPC.") == ["§420"]

    def test_section_symbol_with_space(self):
        assert extract_citations("Cheating is covered by § 420.") == ["§ 420"]

    def test_romanized_hindi(self):
        assert extract_citations("FIR ke liye dhara 154 dekhiye.") == ["dhara 154"]

    def test_devanagari_hindi(self):
        assert extract_citations("FIR के लिए धारा 154 देखिए।") == ["धारा 154"]

    def test_alphanumeric_section(self):
        assert extract_citations("Section 66A of the IT Act was struck down.") == ["Section 66A"]

    def test_subsection_parentheses(self):
        assert extract_citations("Section 318(4) prescribes the punishment.") == ["Section 318(4)"]

    def test_plural_sections(self):
        # Only the first number of an enumeration is captured; resolving
        # enumerations is verify_citations' job.
        assert extract_citations("charged under Sections 34 of the BNS") == ["Sections 34"]

    def test_lowercase_citation_in_prose(self):
        assert extract_citations("as per section 154 of BNSS") == ["section 154"]

    def test_multiple_citations(self):
        got = extract_citations("Section 318 of BNS replaced Section 420 of IPC.")
        assert got == ["Section 318", "Section 420"]


class TestIgnoresProse:
    def test_section_followed_by_word(self):
        assert extract_citations("This section is important for society.") == []

    def test_section_of_society(self):
        assert extract_citations("the section of society most affected") == []

    def test_capitalised_prose(self):
        assert extract_citations("In this Section we discuss remedies.") == []

    def test_marker_inside_word(self):
        assert extract_citations("the dissection 42 experiment") == []

    def test_number_glued_to_act_abbreviation(self):
        # "34IPC" must not be half-eaten into a plausible-but-wrong "34IP";
        # a clean miss is safer for the downstream statute-DB resolver.
        assert extract_citations("Section 34IPC ke tahat case darj hua.") == []
        assert extract_citations("dhara 302IPC ke tahat.") == []

    def test_no_citations_at_all(self):
        assert extract_citations("Consult a licensed advocate for your case.") == []
