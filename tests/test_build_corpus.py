"""Tests for scripts/03_build_corpus.py — the statute-DB extraction pipeline.

The section splitter is inherently heuristic (bare acts are not perfectly
uniform), which is exactly why docs/ROADMAP.md mandates a 5% manual
spot-check and a >=98% clean gate before this output feeds synthetic
generation. These tests pin down the behavior the splitter *must* get right
(TOC-entry rejection, header/footer stripping, retry/skip bookkeeping) using
synthetic bare-act-style fixtures — no real network or PDF files involved.
"""

import importlib.util
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
# clean_pages — running header/footer + page-number stripping
# ---------------------------------------------------------------------------

class TestCleanPages:
    def test_strips_repeated_running_header(self):
        pages = [
            "THE CONSUMER PROTECTION ACT, 2019\n1. Short title.\nBody one.\n1",
            "THE CONSUMER PROTECTION ACT, 2019\n2. Definitions.\nBody two.\n2",
            "THE CONSUMER PROTECTION ACT, 2019\n3. Application.\nBody three.\n3",
            "THE CONSUMER PROTECTION ACT, 2019\n4. Objects.\nBody four.\n4",
        ]
        text = bc.clean_pages(pages)
        assert "THE CONSUMER PROTECTION ACT, 2019" not in text
        assert "Short title" in text and "Definitions" in text

    def test_strips_bare_page_numbers(self):
        pages = ["Header\nSome section text.\n42", "Header\nMore text.\n43"] * 3
        text = bc.clean_pages(pages)
        assert " 42 " not in f" {text} "
        assert " 43 " not in f" {text} "

    def test_collapses_whitespace_into_single_spaces(self):
        text = bc.clean_pages(["Line one.\n\n  Line   two.  "])
        assert "  " not in text
        assert "\n" not in text

    def test_short_document_keeps_content_lines(self):
        # Fewer than 4 pages: nothing should be treated as a "repeated" header.
        pages = ["THE ACT\n1. Short title.\nBody.", "THE ACT\n2. Extent.\nBody two."]
        text = bc.clean_pages(pages)
        assert "Short title" in text
        assert "Extent" in text


# ---------------------------------------------------------------------------
# split_sections — the core regex splitter
# ---------------------------------------------------------------------------

class TestSplitSectionsNumeric:
    def test_basic_two_sections(self):
        text = (
            "1. Short title, extent and commencement.—This Act may be called "
            "the Test Act, 2024. "
            "2. Definitions.—In this Act, unless the context otherwise requires, "
            "(a) 'court' means a court of competent jurisdiction."
        )
        result = bc.split_sections(text)
        assert [r["section"] for r in result] == ["1", "2"]
        assert result[0]["title"] == "Short title, extent and commencement"
        assert "Test Act, 2024" in result[0]["text"]
        assert result[1]["title"] == "Definitions"

    def test_toc_entry_is_dropped_in_favor_of_full_section(self):
        # "ARRANGEMENT OF SECTIONS" table-of-contents entries match the same
        # header pattern but have little/no body — the real section (with a
        # substantial body) must win.
        text = (
            "ARRANGEMENT OF SECTIONS "
            "1. Short title, extent and commencement. "
            "2. Definitions. "
            "CHAPTER I PRELIMINARY "
            "1. Short title, extent and commencement.—This Act extends to the "
            "whole of India and shall come into force on such date as the "
            "Central Government may notify. "
            "2. Definitions.—In this Act, unless the context otherwise "
            "requires, (a) 'appropriate Government' means the Central "
            "Government or a State Government, as the case may be."
        )
        result = bc.split_sections(text)
        by_num = {r["section"]: r for r in result}
        assert len(by_num) == 2
        assert "whole of India" in by_num["1"]["text"]
        assert "appropriate Government" in by_num["2"]["text"]

    def test_alphanumeric_section_number(self):
        text = (
            "66. Computer related offences.—If any person dishonestly does any "
            "act referred to in section 43. "
            "66A. Punishment for offensive messages.—Any person who sends, by "
            "means of a computer resource, any information that is grossly "
            "offensive shall be punishable."
        )
        result = bc.split_sections(text)
        nums = [r["section"] for r in result]
        assert "66A" in nums

    def test_no_markers_returns_empty_list(self):
        assert bc.split_sections("This document has no numbered sections at all.") == []

    def test_section_without_title_dash_falls_back_to_full_text(self):
        text = "5. This section has no em dash separator at all so title stays empty."
        result = bc.split_sections(text)
        assert len(result) == 1
        assert result[0]["section"] == "5"
        assert result[0]["text"]

    def test_preserves_document_order(self):
        text = (
            "3. Third section.—Body three. "
            "1. First section.—Body one. "  # out of numeric order in text stream is unusual but code order must follow text order
        )
        result = bc.split_sections(text)
        assert [r["section"] for r in result] == ["3", "1"]


class TestSplitSectionsArticles:
    def test_constitution_style_articles(self):
        text = (
            "Article 14. Equality before law.—The State shall not deny to any "
            "person equality before the law or the equal protection of the laws "
            "within the territory of India. "
            "Article 21. Protection of life and personal liberty.—No person "
            "shall be deprived of his life or personal liberty except according "
            "to procedure established by law."
        )
        result = bc.split_sections(text, use_articles=True)
        nums = [r["section"] for r in result]
        assert nums == ["14", "21"]
        assert "equal protection" in result[0]["text"]


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


class TestDownloadPdf:
    def test_skips_existing_nonempty_file(self, tmp_path, monkeypatch):
        dest = tmp_path / "act.pdf"
        dest.write_bytes(b"already here")

        def boom(*a, **k):
            raise AssertionError("must not re-download an existing file")

        session = type("S", (), {"get": boom})()
        assert bc.download_pdf("http://example.test/a.pdf", dest, "http://example.test/", session) is True
        assert dest.read_bytes() == b"already here"

    def test_succeeds_on_first_try(self, tmp_path, monkeypatch):
        dest = tmp_path / "act.pdf"
        session = type("S", (), {"get": lambda self, url, headers, timeout: FakeResponse()})()
        assert bc.download_pdf("http://example.test/a.pdf", dest, "http://example.test/", session) is True
        assert dest.exists()

    def test_retries_then_succeeds(self, tmp_path, monkeypatch):
        dest = tmp_path / "act.pdf"
        calls = {"n": 0}

        def flaky_get(self, url, headers, timeout):
            calls["n"] += 1
            if calls["n"] < 3:
                raise requests.ConnectionError("transient network error")
            return FakeResponse()

        session = type("S", (), {"get": flaky_get})()
        monkeypatch.setattr(bc.time, "sleep", lambda _s: None)
        assert bc.download_pdf("http://example.test/a.pdf", dest, "http://example.test/", session) is True
        assert calls["n"] == 3

    def test_gives_up_after_exhausting_retries(self, tmp_path, monkeypatch):
        dest = tmp_path / "act.pdf"

        def always_403(self, url, headers, timeout):
            return FakeResponse(status=403)

        session = type("S", (), {"get": always_403})()
        monkeypatch.setattr(bc.time, "sleep", lambda _s: None)
        assert bc.download_pdf("http://example.test/a.pdf", dest, "http://example.test/", session) is False
        assert not dest.exists()
        assert not dest.with_suffix(dest.suffix + ".tmp").exists()


# ---------------------------------------------------------------------------
# write_jsonl / write_spot_check — output bookkeeping
# ---------------------------------------------------------------------------

class TestOutputWriting:
    def test_write_jsonl_roundtrip(self, tmp_path):
        from nyaya.schemas import StatuteSection

        records = [
            StatuteSection(act_id="bns", act_name="BNS", section="318", title="Cheating", text="Whoever cheats..."),
        ]
        out = tmp_path / "canonical" / "bns.jsonl"
        bc.write_jsonl(records, out)
        assert out.exists()
        import json
        row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert row["section"] == "318"
        assert row["act_id"] == "bns"

    def test_spot_check_samples_five_percent_with_minimum_one(self, tmp_path):
        from nyaya.schemas import StatuteSection

        records = [
            StatuteSection(act_id="bns", act_name="BNS", section=str(i), title="T", text="body")
            for i in range(40)
        ]
        out = tmp_path / "spotcheck.jsonl"
        n = bc.write_spot_check(records, out, fraction=0.05)
        assert n == 2  # round(40 * 0.05)
        assert len(out.read_text(encoding="utf-8").splitlines()) == 2

    def test_spot_check_of_empty_records_writes_nothing(self, tmp_path):
        out = tmp_path / "spotcheck.jsonl"
        n = bc.write_spot_check([], out)
        assert n == 0
        assert not out.exists()
