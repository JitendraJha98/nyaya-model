"""The procedures KB: schema, ingestion, and honest prompt rendering."""

import json
from pathlib import Path

import pytest

from nyaya.retrieval import format_context, load_statute_index
from nyaya.validators import load_statute_db

KB_PATH = Path(__file__).resolve().parents[1] / "data" / "canonical" / "procedures_kb.jsonl"
CANONICAL_DIR = KB_PATH.parent


@pytest.fixture(scope="module")
def kb_rows():
    assert KB_PATH.exists(), "data/canonical/procedures_kb.jsonl not built yet"
    with KB_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestKbSchema:
    def test_minimum_coverage(self, kb_rows):
        assert len(kb_rows) >= 60

    def test_constant_act_identity(self, kb_rows):
        assert {r["act_id"] for r in kb_rows} == {"procedures_kb"}
        assert {r["act_name"] for r in kb_rows} == {"Official Procedural Guidance (India)"}

    def test_slugs_unique_and_kebab(self, kb_rows):
        slugs = [r["section"] for r in kb_rows]
        assert len(slugs) == len(set(slugs))
        assert all(s == s.lower() and " " not in s for s in slugs)

    def test_every_row_has_content_and_source(self, kb_rows):
        for r in kb_rows:
            assert r["title"].strip() and len(r["text"].split()) >= 30, r["section"]
            assert r["source_url"].startswith("http"), r["section"]
            assert r.get("tags"), r["section"]  # lay/Hindi terms power retrieval


class TestKbIngestion:
    @pytest.fixture(scope="class")
    def index(self):
        return load_statute_index(CANONICAL_DIR)

    def test_cyber_fraud_query_surfaces_1930_guidance(self, index):
        hits = index.retrieve("I got scammed online, someone took money from my "
                              "account. Where do I report?", k=8)
        blob = " ".join(h["text"] for h in hits)
        assert "1930" in blob and "cybercrime.gov.in" in blob

    def test_kb_does_not_crowd_out_statute_lookup(self, index):
        hits = index.retrieve("What is the punishment under Section 103 of the BNS?", k=4)
        assert hits[0]["act_id"] == "bns_2023" and hits[0]["section"] == "103"

    def test_kb_rows_excluded_from_citation_whitelist(self):
        db = load_statute_db(CANONICAL_DIR)
        assert "procedures_kb" not in db  # guidance is not citable statute


class TestKbRendering:
    def test_guidance_renders_without_section_framing(self):
        row = {"act_id": "procedures_kb",
               "act_name": "Official Procedural Guidance (India)",
               "section": "cyber-fraud-reporting", "title": "Reporting cyber fraud",
               "text": "Call 1930 immediately."}
        out = format_context([row])
        assert out == "Reporting cyber fraud — official guidance\nCall 1930 immediately."
        assert "Section cyber-fraud-reporting" not in out

    def test_statute_rendering_unchanged(self):
        row = {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023",
               "section": "318", "title": "Cheating", "text": "Whoever deceives."}
        assert format_context([row]) == (
            "Section 318 of the Bharatiya Nyaya Sanhita, 2023 — Cheating\nWhoever deceives.")
