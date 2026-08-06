"""Eval-v1 scorer: does it fix v0's false negatives WITHOUT going soft?

Two obligations, and the second matters more than the first:
  1. Legally correct paraphrases must score.
  2. Wrong answers must still fail. Partial credit is only worth having if it
     cannot be farmed by near-misses, wrong numbers, or wrong sections.
"""

import pytest

from nyaya.evaluation import fact_present
from nyaya.scoring import (
    PARTIAL_FLOOR,
    aggregate,
    lint_fact,
    normalize_tokens,
    score_fact,
    score_record,
)

# Paraphrases that are legally correct but fail Eval-v0's substring rule.
V0_FALSE_NEGATIVES = [
    ("death or imprisonment for life",
     "The punishment is imprisonment for life or death, along with a fine."),
    ("five or more persons", "A group of 5 or more people acting together."),
    ("up to 3 years", "imprisonment which may extend to three years"),
    ("ground of race, caste or community",
     "on grounds of race, caste, community, sex or language"),
    ("promise false from the beginning",
     "the promise was false from the very beginning"),
    ("value below Rs 5,000", "where the value is below five thousand rupees"),
]


@pytest.mark.parametrize("fact,response", V0_FALSE_NEGATIVES)
def test_v1_credits_paraphrases_that_v0_rejected(fact, response):
    assert not fact_present(fact, response), (
        "test case no longer reproduces the v0 failure — pick a new one"
    )
    assert score_fact(fact, response)["score"] == 1.0


# --- the leniency must not be farmable -------------------------------------

def test_wrong_number_scores_zero():
    """The numeric guard is what keeps partial credit honest."""
    assert score_fact("up to 10 years", "may extend to three years")["score"] == 0.0
    assert score_fact("minimum 7 years", "a minimum of three years")["score"] == 0.0


def test_missing_number_scores_zero_even_when_words_all_match():
    result = score_fact("within 7 years of marriage",
                        "occurring within some years of the marriage")
    assert result["score"] == 0.0
    assert "missing number" in result["reason"]


def test_wrong_section_scores_zero():
    """Citations get no partial credit — a near-miss section is just wrong."""
    assert score_fact("Section 103 BNS", "Section 302 IPC applies")["score"] == 0.0
    assert score_fact("Section 103 BNS", "Section 104 BNS applies")["score"] == 0.0


def test_right_number_wrong_act_scores_zero():
    assert score_fact("Section 103 BNS", "Section 103 of the IPC")["score"] == 0.0


def test_unrelated_answer_scores_zero():
    assert score_fact("five or more persons",
                      "You should file an RTI application online.")["score"] == 0.0


def test_partial_credit_respects_the_floor():
    result = score_fact("continuing unlawful activity by a syndicate",
                        "this concerns a syndicate")
    assert result["score"] < 1.0
    assert result["score"] == 0.0 or result["score"] >= PARTIAL_FLOOR


# --- record-level behaviour ------------------------------------------------

def _record(**kw):
    base = {"id": "t1", "language": "english", "legal_domain": "bns",
            "task_type": "legal_qa", "difficulty": "easy",
            "required_facts": [], "forbidden_facts": []}
    base.update(kw)
    return base


def test_forbidden_fact_zeroes_the_record():
    """Stale law is the one error class that cannot be partially forgiven."""
    record = _record(required_facts=["Section 103 BNS"],
                     forbidden_facts=["Section 302 IPC"])
    result = score_record(record, "Section 103 BNS, replacing Section 302 IPC.")
    assert result["forbidden_violated"]
    assert result["fact_recall"] == 0.0
    assert result["all_facts"] is False


def test_partial_record_lands_between_zero_and_one():
    record = _record(required_facts=["Section 103 BNS", "death or imprisonment for life"])
    result = score_record(record, "Section 103 BNS applies here.")
    assert 0.0 < result["fact_recall"] < 1.0
    assert result["all_facts"] is False


def test_full_record_scores_one():
    record = _record(required_facts=["Section 103 BNS", "death or imprisonment for life"])
    result = score_record(
        record, "Under Section 103 BNS the punishment is imprisonment for life or death.")
    assert result["fact_recall"] == 1.0
    assert result["all_facts"] is True


def test_citation_and_substance_tracked_separately():
    record = _record(required_facts=["Section 103 BNS", "five or more persons"])
    result = score_record(record, "A group of 5 or more people is covered.")
    assert result["citation_recall"] == 0.0
    assert result["substance_recall"] == 1.0


def test_scorer_discriminates_good_from_bad():
    """The whole point: v0 cannot separate these, v1 must."""
    record = _record(required_facts=["Section 103 BNS", "death or imprisonment for life"])
    good = score_record(
        record, "Under Section 103 BNS, punishable with imprisonment for life or death.")
    bad = score_record(record, "You should consult a lawyer about this matter.")
    assert good["fact_recall"] - bad["fact_recall"] > 0.9


# --- normalization safety --------------------------------------------------

def test_devanagari_survives_normalization():
    tokens = normalize_tokens("भारतीय न्याय संहिता की धारा 103")
    assert "भारतीय" in tokens and "103" in tokens


def test_stemming_leaves_short_and_double_s_words_alone():
    assert normalize_tokens("witness") == ["witness"]


def test_compound_numerals_fold():
    assert "25" in normalize_tokens("twenty five years")
    assert "200000" in normalize_tokens("2 lakh rupees")


# --- aggregation -----------------------------------------------------------

def test_aggregate_excludes_safety_rows_from_the_headline():
    scored = [
        score_record(_record(required_facts=["five or more persons"]),
                     "5 or more people"),
        score_record(_record(task_type="safety_abstention",
                             required_facts=["consult a lawyer"]), "no idea"),
    ]
    metrics = aggregate(scored)
    assert metrics["scored_total"] == 1
    assert metrics["safety_rows"] == 1
    assert metrics["fact_recall"] == 1.0


# --- fact linting ----------------------------------------------------------

def test_lint_flags_propositions_that_no_metric_can_grade():
    assert lint_fact("deferred, not in force")
    assert lint_fact("snatching is a distinct offence")


def test_lint_passes_quotable_content_and_citations():
    assert lint_fact("five or more persons") == []
    assert lint_fact("Section 103 BNS") == []
