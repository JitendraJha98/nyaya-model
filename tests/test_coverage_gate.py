"""Guidance relevance floor and the coverage gate (Sept 2026).

Before these, every query received exactly KB_SLOTS guidance notes whether or
not they were about the question, and a question outside the acts in the
database still got eight confident statute sections."""
from pathlib import Path

import pytest

from nyaya.retrieval import StatuteIndex, load_statute_index

ROOT = Path(__file__).resolve().parents[1]

MURDER = {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023", "section": "103",
          "title": "Punishment for murder",
          "text": "Whoever commits murder shall be punished with death or imprisonment for life"}
DRUNK_DRIVING_NOTE = {"act_id": "procedures_kb", "act_name": "Official Procedural Guidance (India)",
                      "section": "drunk-driving", "title": "Penalties for drunk driving",
                      "text": "Driving under the influence of drink attracts a fine and licence suspension"}
FIR_SECTION = {"act_id": "bnss_2023", "act_name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
               "section": "173", "title": "Information in cognizable cases",
               "text": "Every information relating to the commission of a cognizable offence"}
FIR_NOTE = {"act_id": "procedures_kb", "act_name": "Official Procedural Guidance (India)",
            "section": "fir-refusal-remedy", "title": "What to do when police refuse to register an FIR",
            "text": "Send the information in writing to the Superintendent of Police"}


def test_guidance_is_not_appended_when_it_is_irrelevant():
    index = StatuteIndex([MURDER, DRUNK_DRIVING_NOTE], mappings=[])
    hits = index.retrieve("what is the punishment for murder", k=4)
    assert [h["section"] for h in hits] == ["103"]


def test_guidance_is_appended_when_it_matches():
    index = StatuteIndex([FIR_SECTION, FIR_NOTE], mappings=[])
    hits = index.retrieve("police refuse to register my FIR information cognizable", k=4)
    assert "fir-refusal-remedy" in [h["section"] for h in hits]


def test_purely_procedural_query_still_fills_from_guidance():
    index = StatuteIndex([MURDER, FIR_NOTE], mappings=[])
    hits = index.retrieve("police refuse to register FIR", k=4)
    assert [h["section"] for h in hits] == ["fir-refusal-remedy"]


@pytest.fixture(scope="module")
def real_index():
    return load_statute_index(ROOT / "data" / "canonical")


def test_coverage_true_for_an_indexed_topic(real_index):
    cov = real_index.coverage("what is the punishment for murder")
    assert cov["covered"] is True


def test_coverage_true_for_an_explicit_citation(real_index):
    assert real_index.coverage("Section 302 IPC")["covered"] is True


def test_coverage_false_when_no_act_in_the_database_matches(real_index):
    # A Devanagari tenancy question: no tenancy act is indexed and the index is
    # built over English text, so nothing should score.
    from nyaya.retrieval import COVERAGE_MIN_SCORE

    cov = real_index.coverage("मकान मालिक 2 महीने से सिक्योरिटी डिपॉजिट वापस नहीं कर रहा")
    assert cov["covered"] is False
    # Not exactly zero: the bare numeral "2" matches "articles 2 and 3" in the
    # Constitution, which is precisely the kind of false confidence the gate exists for.
    assert cov["top_statute_score"] < COVERAGE_MIN_SCORE
