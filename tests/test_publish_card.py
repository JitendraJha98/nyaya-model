"""Guards on the published model card (scripts/31_publish_hf.py).

The card is maintained by hand in docs/cards/nyaya-3b-<version>.md -- it is
prose about results, not a template -- and the publish script uploads that
file. These tests pin the parts that must never regress on the Hub:

  1. `license: other` / `qwen-research` -- Qwen2.5-3B-Instruct is NOT Apache-2.0
     and a merged LoRA inherits the restriction (the first v3 card got this wrong).
  2. The card says the weights are statistically tied with base and that no
     human evaluation has been passed.
  3. It discloses that nyaya-eval-v0 is public and therefore contaminated.
  4. It keeps the "not legal advice" disclaimer.
  5. The eval upload can never include the private Eval-v1 split.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _publish_module():
    spec = importlib.util.spec_from_file_location(
        "publish_hf", ROOT / "scripts" / "31_publish_hf.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pub():
    return _publish_module()


@pytest.fixture(scope="module")
def cards(pub):
    return {v: pub.load_model_card(v) for v in pub.VERSIONS}


def test_every_version_card_loads(cards, pub):
    assert set(cards) == set(pub.VERSIONS)
    assert "v3" in cards


def test_card_never_claims_apache_for_the_weights(cards):
    for version, card in cards.items():
        frontmatter = card.split("---")[1]
        assert "license: other" in frontmatter, version
        assert "license_name: qwen-research" in frontmatter, version
        assert "apache" not in frontmatter.lower(), version
        assert "non-commercial" in card.lower(), version


def test_card_states_the_tie_and_the_missing_human_eval(cards):
    for version, card in cards.items():
        assert "statistically tied" in card, version
        assert "No human evaluation" in card, version


def test_card_discloses_eval_contamination(cards):
    for version, card in cards.items():
        assert "contaminated" in card, version


def test_card_keeps_the_not_legal_advice_disclaimer(cards):
    for version, card in cards.items():
        assert "Not legal advice" in card, version
        assert "Advocates Act, 1961" in card, version


def test_missing_card_is_fatal(pub):
    with pytest.raises(pub.CardDataError):
        pub.load_model_card("v99")


def test_card_without_frontmatter_licence_is_fatal(pub, tmp_path, monkeypatch):
    bad = tmp_path / "nyaya-3b-v3.md"
    bad.write_text("---\nlicense: apache-2.0\n---\n# card\n", encoding="utf-8")
    monkeypatch.setattr(pub, "CARDS", tmp_path)
    with pytest.raises(pub.CardDataError, match="license"):
        pub.load_model_card("v3")


def test_eval_upload_allowlist_can_never_match_the_private_split(pub):
    assert all("private" not in pattern for pattern in pub.EVAL_ALLOW_PATTERNS)
    assert "nyaya_eval_v1.jsonl" not in pub.EVAL_ALLOW_PATTERNS
    assert "nyaya_eval_v0.jsonl" in pub.EVAL_ALLOW_PATTERNS
