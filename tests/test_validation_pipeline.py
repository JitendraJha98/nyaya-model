"""Tests for the per-example validation gate used by scripts/05."""

import pytest

from nyaya.validators import load_statute_db, validate_example

EVAL_RECORDS = [
    {"question": "What is the punishment for murder under current Indian law?",
     "expected_answer": "Section 103 BNS: death or imprisonment for life."},
]


def make_example(answer=None, question="What happens if someone cheats me?",
                 language="english", task_type="grounded_qa",
                 source_sections=None, words=120):
    if answer is None:
        base = "Under Section 318 of the Bharatiya Nyaya Sanhita cheating is punishable. "
        answer = (base * (words // len(base.split()) + 1))
        answer = " ".join(answer.split()[:words])
    return {
        "id": "gen_000001_01",
        "messages": [
            {"role": "system", "content": "You are Nyaya."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "language": language,
            "legal_domain": "bns_2023",
            "task_type": task_type,
            "source_act": "bns_2023",
            "source_sections": source_sections if source_sections is not None else ["bns_2023:318"],
            "generator": "test",
            "verified": False,
            "dataset_version": "v1",
        },
    }


@pytest.fixture(scope="module")
def db():
    return load_statute_db(include_old_law=True)


class TestValidateExample:
    def test_good_example_passes(self, db):
        ok, reasons = validate_example(make_example(), db, EVAL_RECORDS)
        assert ok, reasons

    def test_hallucinated_citation_rejected(self, db):
        words = ("Under Section 999 of the Bharatiya Nyaya Sanhita you will win. " * 20).split()
        ex = make_example(answer=" ".join(words[:120]))
        ok, reasons = validate_example(ex, db, EVAL_RECORDS)
        assert not ok and "citation" in " ".join(reasons)

    def test_grounded_task_without_source_sections_rejected(self, db):
        ex = make_example(source_sections=[])
        ok, reasons = validate_example(ex, db, EVAL_RECORDS)
        assert not ok and "source_sections" in " ".join(reasons)

    def test_too_short_answer_rejected(self, db):
        ex = make_example(words=30)
        ok, reasons = validate_example(ex, db, EVAL_RECORDS)
        assert not ok and "length" in " ".join(reasons)

    def test_mapping_answers_have_lower_floor(self, db):
        ex = make_example(
            answer="IPC Section 420 is now Section 318 of the Bharatiya Nyaya Sanhita; "
                   "for offences before 1 July 2024 the IPC still applies. "
                   "This is general information, not legal advice.",
            task_type="law_mapping")
        ok, reasons = validate_example(ex, db, EVAL_RECORDS)
        assert ok, reasons

    def test_mapping_answers_still_reject_one_liners(self, db):
        ex = make_example(answer="Section 318 of the BNS.", task_type="law_mapping")
        ok, reasons = validate_example(ex, db, EVAL_RECORDS)
        assert not ok and "length" in " ".join(reasons)

    def test_safety_answers_may_be_short(self, db):
        ex = make_example(
            answer="I cannot help with that. Filing a false case is a crime; please consult a licensed advocate.",
            task_type="safety_abstention", source_sections=[])
        ok, reasons = validate_example(ex, db, EVAL_RECORDS)
        assert ok, reasons

    def test_hindi_example_must_use_devanagari(self, db):
        ex = make_example(language="hindi")  # english-script answer
        ok, reasons = validate_example(ex, db, EVAL_RECORDS)
        assert not ok and "language" in " ".join(reasons)

    def test_hindi_devanagari_passes(self, db):
        words = ("भारतीय न्याय संहिता की धारा 318 के तहत धोखाधड़ी दंडनीय है और सजा होती है। " * 30).split()
        ex = make_example(answer=" ".join(words[:120]), language="hindi",
                          question="धोखाधड़ी पर क्या सजा है?")
        ok, reasons = validate_example(ex, db, EVAL_RECORDS)
        assert ok, reasons

    def test_eval_leakage_rejected(self, db):
        ex = make_example(question="What is the punishment for murder under current Indian law?")
        ok, reasons = validate_example(ex, db, EVAL_RECORDS)
        assert not ok and "leakage" in " ".join(reasons)

    def test_malformed_messages_rejected(self, db):
        ex = make_example()
        ex["messages"] = ex["messages"][:2]  # no assistant turn
        ok, reasons = validate_example(ex, db, EVAL_RECORDS)
        assert not ok and "schema" in " ".join(reasons)
