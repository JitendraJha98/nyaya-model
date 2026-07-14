"""Tests for statute-DB-backed citation verification, near-dup and leakage."""

import pytest

from nyaya.validators import (
    detect_eval_leakage,
    is_near_duplicate,
    load_statute_db,
    resolve_citations,
    verify_citations,
)


@pytest.fixture(scope="module")
def db():
    # The real DB committed under data/canonical/ (BNS/BNSS/BSA/RTI).
    return load_statute_db()


class TestLoadStatuteDb:
    def test_families_present(self, db):
        assert {"bns", "bnss", "bsa", "rti"} <= set(db)

    def test_expected_sections(self, db):
        assert "318" in db["bns"] and "358" in db["bns"]
        assert "173" in db["bnss"] and "531" in db["bnss"]
        assert "63" in db["bsa"]
        assert "7" in db["rti"] and "31" in db["rti"]
        assert "999" not in db["bns"]


class TestVerifyCitations:
    def test_valid_citation_with_act(self, db):
        assert verify_citations("You are protected by Section 318 of the BNS.", db)

    def test_nonexistent_section_rejected(self, db):
        assert not verify_citations("See Section 999 of the BNS.", db)

    def test_abbreviation_and_full_name_both_resolve(self, db):
        assert verify_citations("Section 173 BNSS governs FIRs.", db)
        assert verify_citations(
            "Section 173 of the Bharatiya Nagarik Suraksha Sanhita governs FIRs.", db
        )

    def test_citation_without_act_context_rejected(self, db):
        assert not verify_citations("This is governed by Section 318.", db)

    def test_unknown_act_family_rejected(self, db):
        assert not verify_citations("See Section 154 of the CrPC.", db)

    def test_all_fourteen_acts_loaded(self, db):
        # dv/posh/hma/sma/wages/constitution were once silently dropped
        assert "37" in db["dv act"]
        assert "30" in db["posh"]
        assert "13" in db["hma"]   # divorce
        assert "4" in db["sma"]    # conditions for marriage
        assert "69" in db["wages code"]
        assert "21" in db["constitution"]

    def test_devanagari_act_name_resolves(self, db):
        assert verify_citations(
            "भारतीय न्याय संहिता की धारा 318 के तहत धोखाधड़ी दंडनीय है।", db
        )

    def test_article_citations_extract_and_resolve(self, db):
        assert verify_citations("Article 21 of the Constitution protects life.", db)
        assert not verify_citations("Article 999 of the Constitution.", db)
        assert verify_citations("संविधान का अनुच्छेद 21 जीवन की रक्षा करता है।", db)

    def test_dv_and_posh_citations_resolve(self, db):
        assert verify_citations(
            "Section 18 of the Protection of Women from Domestic Violence Act grants protection orders.",
            db,
        )
        assert verify_citations("Section 4 of the POSH Act mandates an Internal Committee.", db)

    def test_old_law_whitelist_from_mapping_table(self):
        # With include_old_law=True the mapping table whitelists old-law
        # sections for historical references ("IPC 420 was replaced by...").
        db = load_statute_db(include_old_law=True)
        assert verify_citations("Section 154 of the CrPC is now Section 173 BNSS.", db)
        assert verify_citations("IPC Section 420 became Section 318 of the BNS.", db)
        assert not verify_citations("Section 9999 of the IPC.", db)

    def test_subsection_resolves_to_base_section(self, db):
        assert verify_citations("Punishable under Section 318(4) of the BNS.", db)

    def test_nearest_act_wins_in_mixed_sentence(self, db):
        text = "IPC Section 420 was replaced by Section 318 of the BNS."
        results = {r["citation"]: r for r in resolve_citations(text, db)}
        bns_hit = [r for r in results.values() if r["section"] == "318"][0]
        assert bns_hit["act"] == "bns" and bns_hit["resolved"]

    def test_devanagari_citation(self, db):
        assert verify_citations("आप धारा 173 BNSS के तहत FIR कर सकते हैं।", db)

    def test_no_citations_is_vacuously_true(self, db):
        assert verify_citations("Consult a licensed advocate for your case.", db)

    def test_one_bad_citation_fails_all(self, db):
        text = "Section 318 of the BNS applies, read with Section 999 of the BNS."
        assert not verify_citations(text, db)


class TestNearDuplicate:
    def test_identical(self):
        assert is_near_duplicate("What is the punishment for theft?",
                                 "What is the punishment for theft?")

    def test_trivial_rewording_is_duplicate(self):
        assert is_near_duplicate(
            "What is the punishment for theft under the BNS?",
            "What is the punishment for theft under the BNS ?",
        )

    def test_different_questions_not_duplicate(self):
        assert not is_near_duplicate(
            "What is the punishment for theft?",
            "How do I file an RTI application online?",
        )


class TestEvalLeakage:
    def _eval_records(self):
        return [
            {
                "question": "What is the punishment for murder under current Indian law?",
                "expected_answer": "Section 103(1) BNS: death or imprisonment for life.",
            }
        ]

    def test_leaked_question_detected(self):
        example = {
            "messages": [
                {"role": "user", "content": "What is the punishment for murder under current Indian law?"},
                {"role": "assistant", "content": "Death or life imprisonment under Section 103 BNS."},
            ]
        }
        assert detect_eval_leakage(example, self._eval_records())

    def test_unrelated_example_passes(self):
        example = {
            "messages": [
                {"role": "user", "content": "How do I claim a refund for a defective phone?"},
                {"role": "assistant", "content": "File before the District Consumer Commission via e-daakhil."},
            ]
        }
        assert not detect_eval_leakage(example, self._eval_records())
