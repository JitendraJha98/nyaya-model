"""Tests for RAFT v3 generation: plan building, response parsing, context gate."""

import pytest

from nyaya.generation import build_raft_plan, parse_raft_response
from nyaya.retrieval import StatuteIndex, context_statute_db
from nyaya.validators import verify_citations

ROWS = [
    {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023", "section": "318",
     "title": "Cheating", "text": "Whoever, by deceiving any person, fraudulently or "
     "dishonestly induces the person so deceived to deliver any property commits cheating. "
     "Punishment extends to seven years imprisonment and fine.", "chapter": "XVII"},
    {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023", "section": "303",
     "title": "Theft", "text": "Whoever intends to take dishonestly any movable property "
     "out of the possession of any person without consent commits theft.", "chapter": "XVII"},
    {"act_id": "rti_2005", "act_name": "Right to Information Act, 2005", "section": "7",
     "title": "Disposal of request", "text": "Request shall be disposed of within thirty "
     "days of the receipt of the request by the public information officer.", "chapter": "II"},
]


def _record(rec_id="gen_000042_ab_01", sections=("bns_2023:318",), language="english"):
    return {
        "id": rec_id,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Someone tricked my mother into a fake investment."},
            {"role": "assistant", "content": "old answer"},
        ],
        "metadata": {"language": language, "task_type": "grounded_qa",
                     "source_act": "bns_2023", "source_sections": list(sections)},
    }


@pytest.fixture(scope="module")
def index():
    return StatuteIndex(ROWS, [])


class TestBuildRaftPlan:
    def test_task_per_record_with_context_and_question(self, index):
        plan = build_raft_plan([_record()], index, k=2, miss_fraction=0.0, seed=7)
        assert len(plan) == 1
        task = plan[0]
        assert "deceiving any person" in task["prompt"]  # gold in teacher prompt
        assert "tricked my mother" in task["prompt"]
        assert "deceiving any person" in task["user_prompt"]  # student-visible prompt
        assert task["source_sections"] == ["bns_2023:318"]
        assert task["is_miss"] is False
        assert "bns_2023:318" in task["context_keys"]

    def test_miss_tasks_exclude_gold(self, index):
        records = [_record(rec_id=f"gen_{i:06d}_xx_01") for i in range(40)]
        plan = build_raft_plan(records, index, k=2, miss_fraction=0.5, seed=7)
        misses = [t for t in plan if t["is_miss"]]
        assert 10 <= len(misses) <= 30  # ~50% of 40, deterministic split
        for t in misses:
            assert "bns_2023:318" not in t["context_keys"]
            assert t["source_sections"] == []

    def test_deterministic(self, index):
        records = [_record(rec_id=f"gen_{i:06d}_xx_01") for i in range(10)]
        a = build_raft_plan(records, index, k=2, miss_fraction=0.3, seed=7)
        b = build_raft_plan(records, index, k=2, miss_fraction=0.3, seed=7)
        assert a == b

    def test_hindi_directive_included(self, index):
        plan = build_raft_plan([_record(language="hindi")], index, k=2,
                               miss_fraction=0.0, seed=7)
        assert "Hindi" in plan[0]["prompt"]


class TestParseRaftResponse:
    def test_builds_training_record(self, index):
        task = build_raft_plan([_record()], index, k=2, miss_fraction=0.0, seed=7)[0]
        raw = ("This is cheating under Section 318 of the Bharatiya Nyaya Sanhita, 2023. "
               "Punishment extends to seven years imprisonment plus fine. "
               "Practical next steps: report at your police station with the payment trail.")
        recs = parse_raft_response(raw, task, "nyaya_instruct_v3")
        assert len(recs) == 1
        rec = recs[0]
        assert rec["messages"][1]["content"] == task["user_prompt"]
        assert rec["messages"][2]["content"] == raw
        assert rec["metadata"]["source_sections"] == ["bns_2023:318"]
        assert rec["metadata"]["rag"]["is_miss"] is False
        assert rec["metadata"]["dataset_version"] == "nyaya_instruct_v3"

    def test_rejects_trivial_output(self, index):
        task = build_raft_plan([_record()], index, k=2, miss_fraction=0.0, seed=7)[0]
        assert parse_raft_response("Yes.", task, "v3") == []
        assert parse_raft_response("", task, "v3") == []


class TestContextStatuteDb:
    def test_verify_against_context_only(self):
        db = context_statute_db(["bns_2023:318", "rti_2005:7"])
        ok = "You are protected by Section 318 of the Bharatiya Nyaya Sanhita."
        stray = "You are protected by Section 303 of the Bharatiya Nyaya Sanhita."
        assert verify_citations(ok, db)
        assert not verify_citations(stray, db)  # 303 exists in the DB, not in context
