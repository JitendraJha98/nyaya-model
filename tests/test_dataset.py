"""Tests for dataset loading and grouped splitting (pure logic)."""

from nyaya.dataset import grouped_split


def _record(i, sections, act="bns_2023"):
    return {
        "id": f"ex_{i:04d}",
        "messages": [{"role": "user", "content": f"q{i}"}],
        "metadata": {"source_sections": sections, "source_act": act},
    }


def _records():
    rows = []
    i = 0
    for section in range(50):
        for _ in range(4):  # 4 examples per source section
            rows.append(_record(i, [f"bns_2023:{section}"]))
            i += 1
    return rows


class TestGroupedSplit:
    def test_ratios_roughly_honoured(self):
        splits = grouped_split(_records(), val_fraction=0.1, test_fraction=0.1, seed=7)
        total = sum(len(v) for v in splits.values())
        assert total == 200
        assert 0.7 <= len(splits["train"]) / total <= 0.9

    def test_no_source_section_straddles_splits(self):
        splits = grouped_split(_records(), val_fraction=0.1, test_fraction=0.1, seed=7)
        seen = {}
        for split, rows in splits.items():
            for r in rows:
                for s in r["metadata"]["source_sections"]:
                    assert seen.setdefault(s, split) == split, f"{s} in two splits"

    def test_deterministic_for_same_seed(self):
        a = grouped_split(_records(), val_fraction=0.1, test_fraction=0.1, seed=7)
        b = grouped_split(_records(), val_fraction=0.1, test_fraction=0.1, seed=7)
        assert [r["id"] for r in a["train"]] == [r["id"] for r in b["train"]]

    def test_holdout_acts_fully_excluded_from_train(self):
        rows = _records() + [_record(1000 + i, [f"posh_2013:{i}"], act="posh_2013") for i in range(8)]
        splits = grouped_split(
            rows, val_fraction=0.1, test_fraction=0.1, seed=7, holdout_acts=["posh_2013"]
        )
        train_acts = {r["metadata"]["source_act"] for r in splits["train"]}
        val_acts = {r["metadata"]["source_act"] for r in splits["val"]}
        assert "posh_2013" not in train_acts
        assert "posh_2013" not in val_acts
        heldout_in_test = [r for r in splits["test"] if r["metadata"]["source_act"] == "posh_2013"]
        assert len(heldout_in_test) == 8
