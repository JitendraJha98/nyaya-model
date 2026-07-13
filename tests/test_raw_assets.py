"""Tests for scripts/13_download_raw_assets.py (pure logic, no network)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

download_raw_assets = __import__("13_download_raw_assets")


class FakeResponse:
    def __init__(self, content=b"%PDF-1.7 fake", status=200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise download_raw_assets.requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses = responses  # url -> FakeResponse
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(url)
        return self.responses.get(url, FakeResponse(status=404))


def test_fetch_group_writes_files_and_manifest(tmp_path):
    assets = [
        {"asset_id": "coi_hindi", "group": "hindi_statutes",
         "url": "https://example.gov.in/coi_hi.pdf", "filetype": "pdf"},
    ]
    session = FakeSession({"https://example.gov.in/coi_hi.pdf": FakeResponse()})
    result = download_raw_assets.fetch_assets(assets, session, out_dir=tmp_path, delay=0)
    saved = tmp_path / "hindi_statutes" / "coi_hindi.pdf"
    assert saved.exists() and saved.read_bytes().startswith(b"%PDF")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["coi_hindi"]["bytes"] == saved.stat().st_size
    assert result.failed == []


def test_failed_fetch_goes_to_manual_list_not_crash(tmp_path):
    assets = [{"asset_id": "gone", "group": "g",
               "url": "https://example.gov.in/404.pdf", "filetype": "pdf"}]
    result = download_raw_assets.fetch_assets(assets, FakeSession({}), out_dir=tmp_path, delay=0)
    assert result.failed == [{"asset_id": "gone",
                              "url": "https://example.gov.in/404.pdf",
                              "reason": "download failed after retries"}]


def test_cached_asset_is_not_refetched(tmp_path):
    assets = [{"asset_id": "a", "group": "g",
               "url": "https://example.gov.in/a.pdf", "filetype": "pdf"}]
    session = FakeSession({"https://example.gov.in/a.pdf": FakeResponse()})
    download_raw_assets.fetch_assets(assets, session, out_dir=tmp_path, delay=0)
    download_raw_assets.fetch_assets(assets, session, out_dir=tmp_path, delay=0)
    assert len(session.calls) == 1
