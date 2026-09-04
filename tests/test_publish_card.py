"""Guards on the published model card (scripts/31_publish_hf.py).

The first v3 release shipped three defects that all reached the public Hub:
  1. `license: apache-2.0` — but Qwen2.5-3B-Instruct is qwen-research
     (non-commercial), and a merged LoRA inherits that.
  2. A dataset citation pass-rate read from checkpoint_evals.json, which holds
     *v1* numbers, presented as v3's.
  3. A silent 'n/a' in the eval table after the re-baseline renamed run keys —
     the lookup used .get() and _pct() swallowed the None.

These tests exist so none of the three can recur.
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
    return {v: pub.build_model_card(v) for v in pub.VERSIONS}


def test_every_version_card_builds(cards, pub):
    assert set(cards) == set(pub.VERSIONS)
    for version, card in cards.items():
        assert f"Nyaya-3B-{version}" in card


def test_card_never_claims_apache_for_the_weights(cards):
    """Qwen2.5-3B is qwen-research; the merged derivative cannot be Apache-2.0."""
    for version, card in cards.items():
        frontmatter = card.split("---")[1]
        assert "license: other" in frontmatter, version
        assert "license_name: qwen-research" in frontmatter, version
        assert "apache" not in frontmatter.lower(), version
        assert "non-commercial" in card.lower(), version


def test_card_has_no_blank_metrics(cards):
    """A missing metric must abort the build, never render as 'n/a'."""
    for version, card in cards.items():
        assert "n/a" not in card, f"{version} card rendered a blank metric"


def test_missing_run_key_is_fatal(pub):
    """Reproduce the exact regression: a renamed run key must raise, not blank."""
    original = pub.VERSIONS["v3"]["eval_key"]
    pub.VERSIONS["v3"]["eval_key"] = "rag_dense_k8_legal-3b-v3-checkpoint-300"
    try:
        with pytest.raises(pub.CardDataError):
            pub.build_model_card("v3")
    finally:
        pub.VERSIONS["v3"]["eval_key"] = original


def test_card_does_not_quote_v1_dataset_eval(cards):
    """checkpoint_evals.json holds v1 numbers — it must not back a v3/v4 claim."""
    for version, card in cards.items():
        assert "90.2%" not in card, version
        assert "checkpoint_evals" not in card, version


def test_card_carries_the_strict_metric_caveat(cards):
    """The strict number is verbatim-phrase agreement, not accuracy — say so."""
    for version, card in cards.items():
        assert "not legal correctness" in card, version
        assert "human-eval ship gate has NOT been passed" in card, version


def test_card_discloses_eval_contamination(cards):
    """nyaya-eval-v0 is public, so it is no longer a held-out benchmark."""
    for version, card in cards.items():
        assert "contaminated" in card, version


def test_card_keeps_the_not_legal_advice_disclaimer(cards):
    for version, card in cards.items():
        assert "Not legal advice" in card, version
        assert "Advocates Act, 1961" in card, version
