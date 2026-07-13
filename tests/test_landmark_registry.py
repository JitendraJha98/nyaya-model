"""Schema tests for configs/landmark_cases.yaml (asset 4 scaffold)."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "landmark_cases.yaml"

REQUIRED = {"case_id", "name", "citation", "year", "court", "topics", "why_landmark"}


def test_registry_schema_and_uniqueness():
    cases = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 50  # grows toward 200 over time
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case_id"
    for c in cases:
        assert REQUIRED <= set(c), f"{c.get('case_id')} missing {REQUIRED - set(c)}"
        assert isinstance(c["year"], int) and 1950 <= c["year"] <= 2026
        assert isinstance(c["topics"], list) and c["topics"]
