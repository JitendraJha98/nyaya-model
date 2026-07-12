"""Step 0 — Download Hugging Face datasets listed in configs/hf_datasets.yaml.

Each enabled dataset id is downloaded into data/hf/<id-with-slashes-replaced>/.
Add or enable datasets by editing the config, then rerun; already-downloaded
datasets are reused from the HF cache.

Usage:
    python scripts/00_download_hf_datasets.py            # all enabled in config
    python scripts/00_download_hf_datasets.py --id org/name   # one-off, any id
"""

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hf_datasets.yaml"
OUT_DIR = ROOT / "data" / "hf"


def safe_name(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def download(dataset_id: str) -> None:
    target = OUT_DIR / safe_name(dataset_id)
    if target.exists() and any(target.iterdir()):
        print(f"[skip] {dataset_id} already in {target}")
        return
    print(f"[download] {dataset_id} -> {target}")
    try:
        ds = load_dataset(dataset_id)
        target.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(target))
        for split, d in ds.items():
            print(f"    {split}: {len(d)} rows")
    except Exception as e:  # keep going; report at the end
        print(f"[FAILED] {dataset_id}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", help="download a single dataset id, ignoring the config")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.id:
        download(args.id)
        return

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for entry in config["datasets"]:
        if entry.get("enabled"):
            download(entry["id"])
        else:
            print(f"[disabled] {entry['id']}  ({entry.get('purpose', '')})")


if __name__ == "__main__":
    main()
