"""scripts/41_build_retriever_pairs.py turns training records into contrastive
(query, gold, hard-negative) pairs for retriever training."""
import importlib.util
from pathlib import Path

from nyaya.retrieval import StatuteIndex

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("pairs", ROOT / "scripts" / "41_build_retriever_pairs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _row(act, section, title, text):
    return {"act_id": act, "act_name": act.upper(), "section": section, "title": title, "text": text}


def _record(rid, question, gold, is_miss=False):
    return {"id": rid,
            "messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": f"context...\n\nQuestion: {question}"},
                         {"role": "assistant", "content": "a"}],
            "metadata": {"language": "english", "source_sections": gold,
                         "rag": {"question": question, "is_miss": is_miss, "context_keys": []}}}


def test_pairs_have_gold_and_disjoint_hard_negatives():
    m = _mod()
    index = StatuteIndex([
        _row("bns_2023", "103", "Punishment for murder", "Whoever commits murder shall be punished"),
        _row("bns_2023", "105", "Culpable homicide", "Whoever commits culpable homicide not amounting to murder"),
        _row("bns_2023", "318", "Cheating", "Whoever cheats shall be punished"),
    ], mappings=[])
    records = [_record("r1", "what is the punishment for murder", ["bns_2023:103"]),
               _record("r2", "someone cheated me", ["bns_2023:318"]),
               _record("r3", "context-less miss demo", ["bns_2023:103"], is_miss=True)]
    pairs, stats = m.build_pairs(records, index, eval_questions=[], negatives=5)
    assert stats["miss_skipped"] == 1
    assert [p["id"] for p in pairs] == ["r1", "r2"]
    assert pairs[0]["positive_keys"] == ["bns_2023:103"]
    assert "bns_2023:103" not in pairs[0]["negative_keys"]
    assert "bns_2023:105" in pairs[0]["negative_keys"]


def test_eval_near_duplicates_are_dropped():
    m = _mod()
    index = StatuteIndex([_row("bns_2023", "103", "Punishment for murder", "Whoever commits murder")], mappings=[])
    records = [_record("r1", "What is the punishment for murder under current Indian law?", ["bns_2023:103"])]
    pairs, stats = m.build_pairs(records, index,
                                 eval_questions=["What is the punishment for murder under current Indian law?"])
    assert pairs == [] and stats["eval_near_duplicate"] == 1
