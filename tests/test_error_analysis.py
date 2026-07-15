"""Tests for error-analysis bucketing (pure logic over prediction rows)."""

from nyaya.evaluation import bucket_failures


def pred(correct=False, task_type="legal_qa", domain="bns", language="english",
         difficulty="medium", missing=None, violated=None, citations=None,
         is_safety=False, abstained=False):
    return {
        "id": "x", "question": "q", "language": language,
        "legal_domain": domain, "task_type": task_type, "difficulty": difficulty,
        "expected_answer": "a", "response": "r",
        "extracted_citations": citations or [],
        "required_facts_found": [],
        "required_facts_missing": missing if missing is not None else ["Section 318 BNS"],
        "forbidden_facts_violated": violated or [],
        "abstained": abstained, "is_safety_row": is_safety,
        "auto_strict_correct": correct,
    }


class TestBucketFailures:
    def test_correct_rows_not_counted(self):
        report = bucket_failures([pred(correct=True, missing=[])])
        assert report["failures"] == 0

    def test_buckets_by_all_dimensions(self):
        rows = [pred(domain="bns"), pred(domain="rti", language="hindi",
                                         task_type="procedural_guidance", difficulty="hard")]
        report = bucket_failures(rows)
        assert report["failures"] == 2
        assert report["by_domain"]["bns"] == 1 and report["by_domain"]["rti"] == 1
        assert report["by_language"]["hindi"] == 1
        assert report["by_task_type"]["procedural_guidance"] == 1
        assert report["by_difficulty"]["hard"] == 1

    def test_failure_mode_classification(self):
        rows = [
            # cited something but wrong/missing facts -> wrong_or_incomplete_citation
            pred(citations=["Section 999"], missing=["Section 318 BNS"]),
            # no citations at all but facts required -> no_citation_given
            pred(citations=[], missing=["Section 318 BNS"]),
            # stale law cited -> stale_law
            pred(violated=["IPC Section 420 as current law"]),
            # abstained instead of answering -> over_abstention
            pred(abstained=True, missing=["Section 318 BNS"]),
        ]
        modes = bucket_failures(rows)["by_failure_mode"]
        assert modes["wrong_or_incomplete_citation"] >= 1
        assert modes["no_citation_given"] >= 1
        assert modes["stale_law_cited"] == 1
        assert modes["over_abstention"] == 1

    def test_safety_rows_reported_separately(self):
        rows = [pred(is_safety=True, task_type="safety_abstention",
                     abstained=False, missing=[])]
        report = bucket_failures(rows)
        assert report["safety"]["total"] == 1
        assert report["safety"]["did_not_abstain"] == 1
        assert report["failures"] == 0  # not mixed into strict buckets

    def test_top_missing_facts(self):
        rows = [pred(missing=["Section 318 BNS"]), pred(missing=["Section 318 BNS"]),
                pred(missing=["Article 21"])]
        top = bucket_failures(rows)["top_missing_facts"]
        assert top[0] == ["Section 318 BNS", 2]
