"""Tests for dataset-baseline scoring (pure logic; the model is faked)."""

import pytest

from nyaya.evaluation import (
    language_matches,
    reference_similarity,
    run_dataset_eval,
    score_dataset_prediction,
)
from nyaya.validators import load_statute_db


@pytest.fixture(scope="module")
def db():
    return load_statute_db(include_old_law=True)


def make_record(language="english", task_type="grounded_qa",
                question="What is the punishment for cheating?",
                answer="Cheating is punishable under Section 318 of the Bharatiya Nyaya Sanhita."):
    return {
        "id": "t1",
        "messages": [
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {"language": language, "task_type": task_type,
                     "source_act": "bns_2023", "source_sections": ["bns_2023:318"]},
    }


class TestLanguageMatches:
    def test_hindi_needs_devanagari(self):
        assert language_matches("hindi", "धारा 318 के तहत सजा होगी।")
        assert not language_matches("hindi", "Section 318 punishes cheating.")

    def test_hinglish_is_latin(self):
        assert language_matches("hinglish", "Section 318 ke under saza hogi.")
        assert not language_matches("hinglish", "धारा 318 के तहत सजा होगी।")

    def test_english_is_latin(self):
        assert language_matches("english", "Section 318 punishes cheating.")


class TestReferenceSimilarity:
    def test_identical_is_one(self):
        assert reference_similarity("a b c", "a b c") == 1.0

    def test_disjoint_is_zero(self):
        assert reference_similarity("alpha beta", "gamma delta") == 0.0

    def test_partial_between(self):
        assert 0.0 < reference_similarity("section 318 cheating", "section 318 theft") < 1.0


class TestScoreDatasetPrediction:
    def test_valid_citation_scores_ok(self, db):
        s = score_dataset_prediction(
            make_record(), "You can be charged under Section 318 of the BNS.", db)
        assert s["citation_ok"] and s["has_citations"] and not s["old_law_cited"]
        assert s["language_ok"]

    def test_hallucinated_citation_flagged(self, db):
        s = score_dataset_prediction(
            make_record(), "You can be charged under Section 999 of the BNS.", db)
        assert not s["citation_ok"]

    def test_old_law_citation_counted(self, db):
        s = score_dataset_prediction(
            make_record(), "You can be charged under Section 420 of the IPC.", db)
        assert s["old_law_cited"]

    def test_wrong_language_flagged(self, db):
        s = score_dataset_prediction(
            make_record(language="hindi"), "Section 318 applies to your case.", db)
        assert not s["language_ok"]


class TestRunDatasetEval:
    def test_metrics_aggregate(self, db):
        records = [make_record(), make_record(language="hindi",
                   answer="धारा 318 के तहत धोखाधड़ी दंडनीय है।")]

        def fake_generate(questions):
            return ["Section 318 of the BNS applies."] * len(questions)

        predictions, metrics = run_dataset_eval(fake_generate, records, db, batch_size=2)
        assert metrics["total"] == 2
        assert metrics["citation_pass_rate"] == 1.0
        assert metrics["by_language"]["hindi"]["language_ok"] == 0  # english reply to hindi q
        assert 0 <= metrics["mean_reference_similarity"] <= 1
