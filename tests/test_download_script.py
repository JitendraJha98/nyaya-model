"""Tests for scripts/00_download_hf_datasets.py download bookkeeping.

The network fetch (datasets.load_dataset) is faked; everything else is real.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "download_hf_datasets", ROOT / "scripts" / "00_download_hf_datasets.py"
)
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


class FakeSplit:
    def __len__(self):
        return 3


class FakeDataset(dict):
    """Mimics DatasetDict: save_to_disk + .items() of split -> rows."""

    def __init__(self, fail_after_partial_write=False):
        super().__init__(train=FakeSplit())
        self.fail_after_partial_write = fail_after_partial_write

    def save_to_disk(self, path):
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "partial.arrow").write_text("partial")
        if self.fail_after_partial_write:
            raise OSError("disk full halfway through save")
        (target / "dataset_dict.json").write_text("{}")


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "OUT_DIR", tmp_path)
    return tmp_path


def test_safe_name_replaces_slashes():
    assert dl.safe_name("org/name") == "org__name"


def test_successful_download_persists_and_reports_success(out_dir, monkeypatch):
    monkeypatch.setattr(dl, "load_dataset", lambda _id: FakeDataset())
    assert dl.download("org/name") is True
    assert (out_dir / "org__name" / "dataset_dict.json").exists()


def test_completed_download_is_skipped_on_rerun(out_dir, monkeypatch):
    monkeypatch.setattr(dl, "load_dataset", lambda _id: FakeDataset())
    dl.download("org/name")

    def boom(_id):
        raise AssertionError("must not re-download a completed dataset")

    monkeypatch.setattr(dl, "load_dataset", boom)
    assert dl.download("org/name") is True


def test_failure_reports_false(out_dir, monkeypatch):
    def boom(_id):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(dl, "load_dataset", boom)
    assert dl.download("org/name") is False


def test_transient_rename_failure_is_retried(out_dir, monkeypatch):
    # Windows AV/indexers briefly lock freshly-written directories; a completed
    # save must survive a transient rename failure, not be deleted as a failure.
    monkeypatch.setattr(dl, "load_dataset", lambda _id: FakeDataset())
    real_rename = Path.rename
    calls = {"n": 0}

    def flaky_rename(self, target):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "Access is denied")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    monkeypatch.setattr(dl.time, "sleep", lambda _s: None)
    assert dl.download("org/name") is True
    assert (out_dir / "org__name" / "dataset_dict.json").exists()


def test_exhausted_rename_retries_keep_the_completed_save(out_dir, monkeypatch):
    # If the lock outlasts the retry budget, the completed save must be kept
    # (as .tmp) for the next run — not deleted as if the download had failed.
    monkeypatch.setattr(dl, "load_dataset", lambda _id: FakeDataset())

    def always_locked(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "rename", always_locked)
    monkeypatch.setattr(dl.time, "sleep", lambda _s: None)
    assert dl.download("org/name") is False
    assert (out_dir / "org__name.tmp" / "dataset_dict.json").exists()


def test_interrupted_save_is_not_mistaken_for_complete(out_dir, monkeypatch):
    # A save that dies halfway must not leave a directory that future runs skip.
    monkeypatch.setattr(dl, "load_dataset", lambda _id: FakeDataset(fail_after_partial_write=True))
    assert dl.download("org/name") is False

    # Retry with a healthy fetch: must actually download, not "[skip]".
    monkeypatch.setattr(dl, "load_dataset", lambda _id: FakeDataset())
    assert dl.download("org/name") is True
    assert (out_dir / "org__name" / "dataset_dict.json").exists()
