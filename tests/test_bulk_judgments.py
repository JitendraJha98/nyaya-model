"""Tests for scripts/12_bulk_judgments.py selection logic (no network)."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("boto3")  # scripts/12 pulls S3 exports; boto3 is not a dev dependency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

bulk = __import__("12_bulk_judgments")

KEYS = [
    {"Key": "metadata/a.parquet", "Size": 100},
    {"Key": "metadata/b.parquet", "Size": 200},
    {"Key": "pdf/2024/x.pdf", "Size": 1000},
    {"Key": "pdf/2010/y.pdf", "Size": 1000},
]


def test_prefix_filter():
    picked = bulk.select_keys(KEYS, prefixes=["metadata/"], max_bytes=10**9, already=set())
    assert [k["Key"] for k in picked] == ["metadata/a.parquet", "metadata/b.parquet"]


def test_empty_prefixes_means_everything():
    picked = bulk.select_keys(KEYS, prefixes=[], max_bytes=10**9, already=set())
    assert len(picked) == 4


def test_byte_cap_stops_selection():
    picked = bulk.select_keys(KEYS, prefixes=[], max_bytes=350, already=set())
    assert [k["Key"] for k in picked] == ["metadata/a.parquet", "metadata/b.parquet"]


def test_contains_filter_selects_substring_matches():
    keys = [
        {"Key": "data/tar/year=2024/english/english.tar", "Size": 10},
        {"Key": "data/tar/year=2024/regional/regional.tar", "Size": 10},
        {"Key": "data/tar/year=2023/english/english.tar", "Size": 10},
    ]
    picked = bulk.select_keys(keys, prefixes=["data/tar/"], max_bytes=10**9,
                              already=set(), contains=["/english/"])
    assert [k["Key"] for k in picked] == [
        "data/tar/year=2024/english/english.tar",
        "data/tar/year=2023/english/english.tar",
    ]


def test_already_downloaded_skipped_and_counted_against_cap():
    picked = bulk.select_keys(KEYS, prefixes=["metadata/"], max_bytes=350,
                              already={"metadata/a.parquet"})
    # a.parquet's 100 bytes count toward the 350 cap (100+200 <= 350), but the
    # file itself is not re-fetched — only b is downloaded
    assert [k["Key"] for k in picked] == ["metadata/b.parquet"]
