"""Reranking contracts, verified with a fake scorer (no model download).

The properties that matter are structural, not model-quality: a reranker must
never displace an exact citation lookup, never let the KB appendix eat statute
slots, and degrade to a no-op rather than to noise when it is uninformative.
"""

import pytest

from nyaya.rerank import CachedReranker, passage_text
from nyaya.retrieval import KB_SLOTS, StatuteIndex


class KeywordReranker:
    """Deterministic stand-in: scores by keyword overlap, no model involved."""

    def __init__(self, depth=50):
        self.depth = depth
        self.calls = []

    def score(self, query, passages):
        terms = set(query.lower().split())
        return [len(terms & set(p.lower().split())) for p in passages]

    def rerank(self, query, rows, k):
        if not rows or k <= 0:
            return []
        self.calls.append((query, len(rows), k))
        candidates = rows[: self.depth]
        scored = sorted(
            zip(self.score(query, [passage_text(r) for r in candidates]),
                range(len(candidates))),
            key=lambda t: (-t[0], t[1]))
        return [candidates[i] for _s, i in scored[:k]]


class IndifferentReranker(KeywordReranker):
    """Scores everything identically — must behave as a no-op."""

    def score(self, query, passages):
        return [1.0] * len(passages)


def _row(act, section, title, text):
    return {"act_id": act, "section": section, "title": title, "text": text,
            "act_name": act.upper()}


@pytest.fixture
def index():
    rows = [
        _row("bns", "103", "Punishment for murder",
             "Whoever commits murder shall be punished with death or imprisonment for life"),
        _row("bns", "318", "Cheating", "Whoever cheats shall be punished"),
        _row("bnss", "173", "Information in cognizable cases",
             "Every information relating to the commission of a cognizable offence"),
        _row("bns", "63", "Rape", "A man is said to commit rape"),
        _row("procedures_kb", "FIR-1", "How to file an FIR",
             "Go to the police station and file an FIR; zero FIR is available"),
        _row("procedures_kb", "FIR-2", "FIR refusal remedy",
             "If police refuse, approach the Superintendent of Police"),
        _row("procedures_kb", "FIR-3", "Cybercrime reporting",
             "Report cyber fraud on the 1930 helpline"),
    ]
    return StatuteIndex(rows, mappings=[])


def test_reranker_defaults_to_none(index):
    assert index.reranker is None


def test_exact_citation_is_never_reranked_away(index):
    """A named section is a resolved fact; no model score may displace it."""
    index.set_reranker(KeywordReranker())
    hits = index.retrieve("What does Section 103 BNS say about murder?", k=4)
    assert hits[0]["section"] == "103", "exact reference must stay first"


def test_guidance_cannot_take_statute_slots(index):
    index.set_reranker(KeywordReranker())
    hits = index.retrieve("police station FIR cognizable offence information", k=4)
    guidance = [h for h in hits if h["act_id"] == "procedures_kb"]
    assert len(guidance) <= KB_SLOTS
    statutes = [h for h in hits if h["act_id"] != "procedures_kb"]
    assert statutes, "statutes must still be present"


def test_statutes_and_guidance_reranked_separately(index):
    """Two independent rerank calls — a shared ranking would let short,
    keyword-dense KB rows outrank statutes and crowd them out."""
    rr = KeywordReranker()
    index.set_reranker(rr)
    # A query that surfaces BOTH pools. (With a statute-only query the
    # guidance pool is empty and rerank short-circuits before recording.)
    index.retrieve("police station FIR cognizable offence information", k=4)
    pools = [n_rows for _q, n_rows, _k in rr.calls]
    assert len(rr.calls) >= 2, f"expected statute + guidance calls, got {rr.calls}"
    assert all(n > 0 for n in pools)


def test_indifferent_reranker_is_a_noop(index):
    """A useless reranker must degrade to first-stage order, not to noise."""
    before = index.retrieve("cheating punishment", k=4)
    index.set_reranker(IndifferentReranker())
    after = index.retrieve("cheating punishment", k=4)
    assert [r["section"] for r in before] == [r["section"] for r in after]


def test_respects_k(index):
    index.set_reranker(KeywordReranker())
    for k in (1, 2, 5):
        assert len(index.retrieve("murder", k=k)) <= k + KB_SLOTS


def test_depth_caps_candidates_considered():
    rr = KeywordReranker(depth=3)
    rows = [_row("bns", str(i), f"title {i}", "text") for i in range(10)]
    out = rr.rerank("title 9", rows, k=5)
    assert len(out) <= 3, "must not consider beyond depth"
    assert all(r["section"] in {"0", "1", "2"} for r in out)


class TestPassageText:
    def test_leads_with_act_section_and_title(self):
        text = passage_text(_row("bns", "103", "Punishment for murder", "body here"))
        assert text.startswith("BNS Section 103: Punishment for murder")

    def test_truncates_long_bodies(self):
        text = passage_text(_row("bns", "1", "t", "x" * 5000), max_chars=100)
        assert len(text) < 300

    def test_survives_missing_fields(self):
        assert passage_text({"act_id": "bns", "section": "1"})


class TestCachedReranker:
    def test_second_call_hits_cache(self, tmp_path):
        inner = KeywordReranker()
        calls = {"n": 0}
        original = inner.score

        def counting(query, passages):
            calls["n"] += 1
            return original(query, passages)

        inner.score = counting
        inner.model_name = "fake-model"
        cached = CachedReranker(inner, tmp_path / "c.json")
        rows = [_row("bns", "103", "murder", "body")]
        cached.rerank("murder", rows, 1)
        cached.rerank("murder", rows, 1)
        assert calls["n"] == 1, "identical pairs must not be re-scored"

    def test_cache_persists_across_instances(self, tmp_path):
        inner = KeywordReranker()
        inner.model_name = "fake-model"
        path = tmp_path / "c.json"
        rows = [_row("bns", "103", "murder", "body")]
        first = CachedReranker(inner, path)
        first.rerank("murder", rows, 1)
        first.flush()
        assert path.exists()
        assert CachedReranker(inner, path)._cache
