"""Step 0 — Download Hugging Face datasets listed in configs/hf_datasets.yaml.

Each enabled dataset id is downloaded into data/hf/<id-with-slashes-replaced>/.
Add or enable datasets by editing the config, then rerun. An already-populated
target directory is skipped outright; an interrupted download is cleaned up and
retried on the next run (Hugging Face's own cache makes the retry cheap).

NOTE: some listed datasets (e.g. the aalap instruction set, IL-TUR) are gated
on the Hub — run `huggingface-cli login` and accept each dataset's terms on
huggingface.co before enabling them.

Usage:
    python scripts/00_download_hf_datasets.py            # all enabled in config
    python scripts/00_download_hf_datasets.py --id org/name   # one-off, any id
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hf_datasets.yaml"
OUT_DIR = ROOT / "data" / "hf"


def safe_name(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def _rename_with_retries(src: Path, dst: Path, attempts: int = 5) -> None:
    # Windows AV/indexers briefly lock freshly-written directories; retry
    # instead of discarding a completed download over a transient lock.
    for attempt in range(attempts):
        try:
            src.rename(dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.2 * (attempt + 1))


def download(
    dataset_id: str,
    name: str | None = None,
    split: str | None = None,
    allow_patterns: list[str] | None = None,
) -> bool:
    """Download one dataset. True if it is (or already was) fully downloaded.

    `name`/`split` select a single HF config/split (for corpora where the
    full download is not needed). `allow_patterns` switches to a raw-file
    snapshot download for repos that are not load_dataset-compatible (e.g.
    NyayaAnumana ships plain zips) — only matching repo files are fetched.
    """
    target = OUT_DIR / safe_name(dataset_id)
    if target.exists() and any(target.iterdir()):
        print(f"[skip] {dataset_id} already in {target}")
        return True
    print(f"[download] {dataset_id} -> {target}")
    # Save to a temp sibling and rename only on success, so an interrupted
    # save is never mistaken for a completed download by the skip check above.
    tmp = target.with_name(target.name + ".tmp")
    ds = None
    try:
        if tmp.exists():
            shutil.rmtree(tmp)
        if allow_patterns:
            snapshot_download(
                dataset_id,
                repo_type="dataset",
                allow_patterns=allow_patterns,
                local_dir=str(tmp),
            )
        else:
            args = (name,) if name else ()
            kwargs = {"split": split} if split else {}
            try:
                ds = load_dataset(dataset_id, *args, **kwargs)
                ds.save_to_disk(str(tmp))
            except ValueError as e:
                # Multi-config datasets (IL-TUR) refuse a bare load — fetch
                # every config into its own subdirectory instead.
                if name or "Config name is missing" not in str(e):
                    raise
                for cfg in get_dataset_config_names(dataset_id):
                    d = load_dataset(dataset_id, cfg, **kwargs)
                    d.save_to_disk(str(tmp / cfg))
                    counts = (
                        f"{split}: {len(d)} rows" if split
                        else ", ".join(f"{s}: {len(x)}" for s, x in d.items())
                    )
                    print(f"    [{cfg}] {counts} rows")
    except Exception as e:  # keep going; main() prints a failure summary at the end
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[FAILED] {dataset_id}: {e}")
        return False
    try:
        if target.exists():
            target.rmdir()  # only ever an empty leftover; skip check caught the rest
        _rename_with_retries(tmp, target)
    except OSError as e:
        # The save itself completed — keep tmp so the work is not lost; the
        # next run cleans it up and retries cheaply via the HF cache.
        print(f"[FAILED] {dataset_id}: saved to {tmp} but could not move into place: {e}")
        return False
    if ds is not None:
        if split:
            print(f"    {split}: {len(ds)} rows")
        else:
            for split_name, d in ds.items():
                print(f"    {split_name}: {len(d)} rows")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", help="download a single dataset id, ignoring the config")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.id:
        entries = [{"id": args.id}]
    else:
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        entries = []
        for entry in config["datasets"]:
            if entry.get("enabled"):
                entries.append(entry)
            else:
                print(f"[disabled] {entry['id']}  ({entry.get('purpose', '')})")

    failed = [
        e["id"]
        for e in entries
        if not download(
            e["id"],
            name=e.get("name"),
            split=e.get("split"),
            allow_patterns=e.get("allow_patterns"),
        )
    ]
    if failed:
        print(f"\n{len(failed)}/{len(entries)} downloads FAILED: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
