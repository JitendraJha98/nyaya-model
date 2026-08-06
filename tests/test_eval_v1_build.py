"""Nyaya-Eval-v1 builder: the curated set must be gradeable and non-leaky.

The defining property of v1 is that a perfect answer can score 1.0. Eval-v0
lacked it — scoring v0's own gold answers under its own metric gave ~10.7%,
so no model could ever exceed that. These tests assert the property holds and
that the public/private split cannot silently leak.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def builder():
    sys.argv = ["25_build_eval_v1.py"]
    spec = importlib.util.spec_from_file_location(
        "build_eval_v1", ROOT / "scripts" / "25_build_eval_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def curated(builder):
    records = builder._load(builder.EVAL_V0)
    v1, report = builder.curate(records)
    return v1, report


def test_gold_answers_score_perfectly(curated, builder):
    """THE point of v1: the reference answer must be able to score 1.0.

    Under Eval-v0 this was ~10.7%, which capped every model below it.
    """
    from nyaya.scoring import aggregate, score_record

    v1, _ = curated
    gradeable = [r for r in v1 if not r["needs_curation"]]
    metrics = aggregate([score_record(r, r["expected_answer"]) for r in gradeable])
    assert metrics["fact_recall"] == 1.0
    assert metrics["citation_accuracy"] == 1.0


def test_no_gold_answer_trips_its_own_forbidden_facts(curated, builder):
    from nyaya.scoring import forbidden_present

    v1, _ = curated
    for record in v1:
        for fact in record["forbidden_facts"]:
            assert not forbidden_present(fact, record["expected_answer"]), (
                f"{record['id']}: forbidden fact {fact!r} fires on its own gold answer"
            )


def test_nothing_is_silently_discarded(curated):
    """Quarantine, never delete — the rewrite work must stay visible."""
    v1, report = curated
    original = {r["id"]: r for r in
                [json.loads(l) for l in
                 (ROOT / "data" / "eval" / "nyaya_eval_v0.jsonl")
                 .read_text(encoding="utf-8").splitlines() if l.strip()]}
    assert len(v1) == len(original)
    for record in v1:
        before = original[record["id"]]
        after = record["required_facts"] + [q["fact"] for q in record["quarantined_facts"]]
        assert sorted(after) == sorted(before["required_facts"]), record["id"]
    assert report["facts_kept"] + report["facts_quarantined"] == sum(
        len(r["required_facts"]) for r in original.values())


def test_records_with_no_gradeable_fact_are_flagged_out(curated):
    v1, _ = curated
    for record in v1:
        assert record["needs_curation"] == (not record["required_facts"])


class TestSplit:
    def test_split_is_disjoint_and_complete(self, builder, curated):
        v1, _ = curated
        public, private = builder.split_public_private(v1)
        public_ids = {r["id"] for r in public}
        private_ids = {r["id"] for r in private}
        assert not (public_ids & private_ids), "a record is in BOTH halves"
        assert public_ids | private_ids == {r["id"] for r in v1}

    def test_split_is_deterministic_across_runs(self, builder, curated):
        """Stable ids, not RNG state — otherwise the private set leaks over time."""
        v1, _ = curated
        first, _ = builder.split_public_private(v1)
        second, _ = builder.split_public_private(v1)
        assert [r["id"] for r in first] == [r["id"] for r in second]

    def test_split_is_stratified_by_domain(self, builder, curated):
        """Both halves must stay representative, or they measure different things."""
        v1, _ = curated
        public, private = builder.split_public_private(v1)
        domains = {r["legal_domain"] for r in v1 if r["legal_domain"]}
        public_domains = {r["legal_domain"] for r in public}
        for domain in domains:
            count = sum(1 for r in v1 if r["legal_domain"] == domain)
            if count >= 5:
                assert domain in public_domains, f"{domain} missing from public half"

    def test_visibility_is_tagged_on_every_record(self, builder, curated):
        v1, _ = curated
        public, private = builder.split_public_private(v1)
        assert all(r["visibility"] == "public" for r in public)
        assert all(r["visibility"] == "private" for r in private)
