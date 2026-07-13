"""Download raw official assets listed in configs/raw_assets.yaml.

Generic fetcher for every 'just get these official files' need: Hindi statute
texts, Law Commission reports, AIBE papers, procedure-KB sources. Writes
data/raw/assets/<group>/<asset_id>.<ext>, a manifest with checksums, and a
manual-download task list for anything it could not fetch (never crashes on a
bad source).

Usage:
    python scripts/13_download_raw_assets.py                 # all groups
    python scripts/13_download_raw_assets.py --group hindi_statutes
"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "raw_assets.yaml"
OUT_DIR = ROOT / "data" / "raw" / "assets"
REPORTS = ROOT / "reports"
RETRIES = 3

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
}


@dataclass
class FetchResult:
    fetched: list = field(default_factory=list)
    cached: list = field(default_factory=list)
    failed: list = field(default_factory=list)


def _load_manifest(out_dir: Path) -> dict:
    p = out_dir / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def fetch_assets(assets: list[dict], session, out_dir: Path = OUT_DIR,
                 delay: float = 1.0) -> FetchResult:
    """Fetch each asset once; polite delay between requests; manifest-tracked."""
    result = FetchResult()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(out_dir)
    for asset in assets:
        dest = out_dir / asset["group"] / f"{asset['asset_id']}.{asset['filetype']}"
        if dest.exists() and dest.stat().st_size > 0:
            result.cached.append(asset["asset_id"])
            continue
        headers = dict(BROWSER_HEADERS)
        if asset.get("referer"):
            headers["Referer"] = asset["referer"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = None
        for attempt in range(RETRIES):
            try:
                resp = session.get(asset["url"], headers=headers, timeout=180)
                resp.raise_for_status()
                content = resp.content
                break
            except requests.RequestException:
                if attempt < RETRIES - 1:
                    time.sleep(delay * (attempt + 1))
        if content is None:
            result.failed.append({"asset_id": asset["asset_id"], "url": asset["url"],
                                  "reason": "download failed after retries"})
            continue
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(dest)
        manifest[asset["asset_id"]] = {
            "url": asset["url"],
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        result.fetched.append(asset["asset_id"])
        time.sleep(delay)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", help="fetch a single group from the config")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assets = [a for a in config["assets"]
              if not args.group or a["group"] == args.group]
    result = fetch_assets(assets, requests.Session())
    print(f"fetched={len(result.fetched)} cached={len(result.cached)} "
          f"failed={len(result.failed)}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "manual_downloads.json").write_text(
        json.dumps(result.failed, indent=2), encoding="utf-8")
    if result.failed:
        print(f"[manual] {len(result.failed)} assets need manual download — "
              f"see reports/manual_downloads.json")


if __name__ == "__main__":
    main()
