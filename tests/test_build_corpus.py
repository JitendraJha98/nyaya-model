"""Tests for scripts/03_build_corpus.py — the statute-DB build ORCHESTRATOR.

The heavy lifting (PDF text extraction, section splitting, validation, NCRB
mapping parsing) lives in src/nyaya/corpus.py and is tested in test_corpus.py +
test_mappings.py. This script is a thin I/O layer over that library, so these
tests pin down the orchestration behaviour: download retry/skip bookkeeping,
that build_act maps library output into full StatuteSection-shaped rows (the
chapter/effective_date/source_url fields the canonical DB depends on), that
build_mappings writes law_mappings.jsonl, and that spot-check sampling is
deterministic. No real network or PDF files are involved.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "build_corpus", ROOT / "scripts" / "03_build_corpus.py"
)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


# ---------------------------------------------------------------------------
# download_pdf — retry/skip bookkeeping (network itself is faked)
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, content=b"%PDF-1.4 fake", status=200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def _session(get):
    """Build a fake requests.Session whose .get is `get(self, url, headers, timeout)`."""
    return type("S", (), {"get": get})()


class TestDownloadPdf:
    def test_skips_existing_nonempty_file(self, tmp_path):
        dest = tmp_path / "act.pdf"
        dest.write_bytes(b"already here")

        def boom(self, *a, **k):
            raise AssertionError("must not re-download an existing file")

        assert bc.download_pdf("http://example.test/a.pdf", dest, _session(boom)) is True
        assert dest.read_bytes() == b"already here"

    def test_succeeds_on_first_try(self, tmp_path):
        dest = tmp_path / "act.pdf"
        session = _session(lambda self, url, headers, timeout: FakeResponse())
        assert bc.download_pdf("http://example.test/a.pdf", dest, session) is True
        assert dest.exists()

    def test_sends_referer_when_given(self, tmp_path):
        dest = tmp_path / "act.pdf"
        seen = {}

        def capture(self, url, headers, timeout):
            seen.update(headers)
            return FakeResponse()

        bc.download_pdf("http://x/a.pdf", dest, _session(capture), referer="http://handle/")
        assert seen.get("Referer") == "http://handle/"
        assert "Mozilla" in seen.get("User-Agent", "")

    def test_retries_then_succeeds(self, tmp_path, monkeypatch):
        dest = tmp_path / "act.pdf"
        calls = {"n": 0}

        def flaky_get(self, url, headers, timeout):
            calls["n"] += 1
            if calls["n"] < 3:
                raise requests.ConnectionError("transient network error")
            return FakeResponse()

        monkeypatch.setattr(bc.time, "sleep", lambda _s: None)
        assert bc.download_pdf("http://example.test/a.pdf", dest, _session(flaky_get)) is True
        assert calls["n"] == 3

    def test_gives_up_after_exhausting_retries(self, tmp_path, monkeypatch):
        dest = tmp_path / "act.pdf"
        session = _session(lambda self, url, headers, timeout: FakeResponse(status=403))
        monkeypatch.setattr(bc.time, "sleep", lambda _s: None)
        assert bc.download_pdf("http://example.test/a.pdf", dest, session) is False
        assert not dest.exists()
        assert not dest.with_suffix(dest.suffix + ".tmp").exists()

    def test_skip_download_of_missing_file_returns_false(self, tmp_path):
        dest = tmp_path / "act.pdf"

        def boom(self, *a, **k):
            raise AssertionError("--skip-download must not hit the network")

        assert bc.download_pdf("http://x/a.pdf", dest, _session(boom), skip_download=True) is False


# ---------------------------------------------------------------------------
# build_act — library output -> full StatuteSection-shaped rows
# ---------------------------------------------------------------------------

_ACT = {
    "act_id": "bns_2023",
    "act_name": "Bharatiya Nyaya Sanhita, 2023",
    "url": "http://indiacode/bns.pdf",
    "handle_url": "http://indiacode/handle/bns",
    "expected_sections": 2,
    "effective_date": "2024-07-01",
}


def _patch_extraction(monkeypatch, sections):
    monkeypatch.setattr(bc, "download_pdf", lambda *a, **k: True)
    monkeypatch.setattr(bc, "extract_pdf_text", lambda pdf: "raw text")
    monkeypatch.setattr(bc, "slice_act_body", lambda text: text)
    monkeypatch.setattr(bc, "split_sections", lambda text: sections)
    monkeypatch.setattr(
        bc, "validate_sections",
        lambda secs, expected_count=None: {
            "extracted": len(secs), "expected": expected_count, "monotonic": True,
            "numbering_gaps": [], "empty_or_short": [], "clean_fraction": 1.0,
        },
    )


class TestBuildAct:
    def test_rows_carry_chapter_effective_date_and_source_url(self, monkeypatch):
        _patch_extraction(monkeypatch, [
            {"section": "103", "title": "Punishment for murder", "text": "Whoever...", "chapter": "CHAPTER VI — Offences Affecting The Human Body"},
        ])
        rows, report = bc.build_act(_ACT, session=object())
        row = rows[0]
        # the regression the fix repairs: chapter must survive, not be dropped
        assert row["chapter"].startswith("CHAPTER VI")
        assert row["effective_date"] == "2024-07-01"
        assert row["source_url"] == "http://indiacode/bns.pdf"
        assert row["act_id"] == "bns_2023"
        assert row["section"] == "103"
        # StatuteSection-shaped: every dataclass field present
        for key in ("subsection", "replaces", "punishment_summary", "tags"):
            assert key in row
        assert report["clean_fraction"] == 1.0

    def test_missing_pdf_raises(self, monkeypatch):
        monkeypatch.setattr(bc, "download_pdf", lambda *a, **k: False)
        with pytest.raises(FileNotFoundError):
            bc.build_act(_ACT, session=object())


# ---------------------------------------------------------------------------
# build_mappings — NCRB pairs -> law_mappings.jsonl
# ---------------------------------------------------------------------------

class TestBuildMappings:
    def test_writes_pairs_and_counts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bc, "OUT_DIR", tmp_path)
        monkeypatch.setattr(bc, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(bc, "download_pdf", lambda *a, **k: True)
        monkeypatch.setattr(bc, "parse_ncrb_mapping", lambda pdf: [("103", "302"), ("115", "323")])

        counts = bc.build_mappings(
            [{"mapping_id": "bns_ipc", "new_act": "BNS", "old_act": "IPC", "url": "http://ncrb/x.pdf"}],
            session=object(),
        )
        assert counts == {"bns_ipc": 2}
        rows = [json.loads(l) for l in (tmp_path / "law_mappings.jsonl").read_text(encoding="utf-8").splitlines()]
        assert {"old_act": "IPC", "old_section": "302", "new_act": "BNS", "new_section": "103", "note": None} in rows

    def test_unavailable_pdf_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bc, "OUT_DIR", tmp_path)
        monkeypatch.setattr(bc, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(bc, "download_pdf", lambda *a, **k: False)
        counts = bc.build_mappings(
            [{"mapping_id": "bns_ipc", "new_act": "BNS", "old_act": "IPC", "url": "http://ncrb/x.pdf"}],
            session=object(),
        )
        assert counts == {}
        assert not (tmp_path / "law_mappings.jsonl").exists()


# ---------------------------------------------------------------------------
# spot_check_sample — deterministic 5% sampling
# ---------------------------------------------------------------------------

class TestSpotCheckSample:
    def test_samples_five_percent_with_minimum_three(self):
        rows = [{"section": str(i), "title": "T", "text": "body", "chapter": None} for i in range(100)]
        sample = bc.spot_check_sample(rows, fraction=0.05)
        assert len(sample) == 5
        # only the reviewer-facing fields are kept
        assert set(sample[0]) == {"section", "title", "text"}

    def test_minimum_three_when_fraction_would_round_lower(self):
        rows = [{"section": str(i), "title": "T", "text": "body"} for i in range(10)]
        assert len(bc.spot_check_sample(rows, fraction=0.05)) == 3

    def test_deterministic_across_calls(self):
        rows = [{"section": str(i), "title": "T", "text": "body"} for i in range(50)]
        assert bc.spot_check_sample(rows) == bc.spot_check_sample(rows)

    def test_empty_rows_returns_empty(self):
        assert bc.spot_check_sample([]) == []
