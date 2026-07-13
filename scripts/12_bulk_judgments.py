"""Bulk judgment downloader — AWS Open Data archives, anonymous, resumable.

Syncs configured prefixes of the public SC/HC judgment buckets into
data/raw/judgments/<source_id>/, enforcing a total-size cap across all
sources. Anonymous S3 (UNSIGNED) — zero cost, no AWS account. Safe to
interrupt: a manifest records completed keys; re-running resumes.

Buckets (registry.opendata.aws, both CC-BY-4.0, ap-south-1, sourced from
the official eCourts site):
    indian-supreme-court-judgments   SC 1950-2025, English + regional zips
    indian-high-court-judgments      25 High Courts, metadata + pdf tars

Usage:
    python scripts/12_bulk_judgments.py                # all enabled sources
    python scripts/12_bulk_judgments.py --dry-run      # plan only, no download
    python scripts/12_bulk_judgments.py --source sc_judgments
"""

import argparse
import json
from pathlib import Path

import boto3
import yaml
from botocore import UNSIGNED
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bulk_sources.yaml"
OUT_DIR = ROOT / "data" / "raw" / "judgments"
MANIFEST = OUT_DIR / "manifest.json"


def select_keys(keys: list[dict], prefixes: list[str], max_bytes: int,
                already: set[str], contains: list[str] | None = None) -> list[dict]:
    """Pick keys matching any prefix (and any `contains` substring, if given),
    stopping at the byte cap.

    `contains` filters within a prefix where S3 prefixes can't express the
    slice (e.g. only "/english/" tars across every year= partition). Bytes of
    already-downloaded keys count toward the cap so a re-run never exceeds it;
    the keys themselves are not re-downloaded.
    """
    picked, used = [], 0
    for k in keys:
        if prefixes and not any(k["Key"].startswith(p) for p in prefixes):
            continue
        if contains and not any(s in k["Key"] for s in contains):
            continue
        if used + k["Size"] > max_bytes:
            break
        used += k["Size"]
        if k["Key"] in already:
            continue
        picked.append(k)
    return picked


def list_bucket(client, bucket: str, prefixes: list[str]) -> list[dict]:
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for prefix in (prefixes or [""]):
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys.extend({"Key": o["Key"], "Size": o["Size"]}
                        for o in page.get("Contents", []))
    return keys


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def sync_source(source: dict, max_bytes_left: int, dry_run: bool) -> int:
    """Sync one source; returns bytes downloaded (or planned, if dry_run)."""
    client = boto3.client("s3", region_name=source["region"],
                          config=Config(signature_version=UNSIGNED))
    manifest = load_manifest()
    done = set(manifest.get(source["source_id"], []))
    keys = list_bucket(client, source["bucket"], source["prefixes"])
    plan = select_keys(keys, source["prefixes"], max_bytes_left, done,
                       contains=source.get("contains"))
    total = sum(k["Size"] for k in plan)
    print(f"[{source['source_id']}] {len(plan)} objects, "
          f"{total / 2**30:.2f} GiB planned ({len(done)} already done)")
    if dry_run:
        return total
    downloaded = 0
    for i, k in enumerate(plan):
        dest = OUT_DIR / source["source_id"] / k["Key"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(source["bucket"], k["Key"], str(dest))
        downloaded += k["Size"]
        done.add(k["Key"])
        if i % 50 == 0 or i == len(plan) - 1:  # checkpoint the manifest
            manifest[source["source_id"]] = sorted(done)
            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")
            print(f"  {i + 1}/{len(plan)} ({downloaded / 2**30:.2f} GiB)", flush=True)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="sync a single source_id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cap = int(config["max_total_gb"] * 2**30)
    already_bytes = sum(
        f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()
    ) if OUT_DIR.exists() else 0
    left = cap - already_bytes
    print(f"cap={config['max_total_gb']} GB, used={already_bytes / 2**30:.2f} GiB")
    for source in config["sources"]:
        if not source.get("enabled") or (args.source and source["source_id"] != args.source):
            continue
        if left <= 0:
            print(f"[{source['source_id']}] SKIPPED — cap reached")
            continue
        left -= sync_source(source, left, args.dry_run)


if __name__ == "__main__":
    main()
