"""Step 37 — Publish the retrieval demo Space (NyayaLabs98/nyaya-demo).

The Space is self-contained: this script assembles a staging directory with
space/ (app, README, requirements), the `nyaya` package from src/, and the
statute DB from data/canonical, then uploads it. One source of truth stays in
the repository; nothing is duplicated in git.

Auth: HF_TOKEN with write access to the NyayaLabs98 org.

Usage:
    python scripts/37_publish_space.py --dry-run     # assemble and list, upload nothing
    python scripts/37_publish_space.py
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SPACE_REPO = "NyayaLabs98/nyaya-demo"


def assemble(staging: Path) -> None:
    shutil.copytree(ROOT / "space", staging, dirs_exist_ok=True)
    shutil.copytree(ROOT / "src" / "nyaya", staging / "nyaya",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (staging / "data" / "canonical").mkdir(parents=True)
    for path in sorted((ROOT / "data" / "canonical").glob("*.jsonl")):
        shutil.copy(path, staging / "data" / "canonical" / path.name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=SPACE_REPO)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "space"
        assemble(staging)
        files = sorted(str(f.relative_to(staging)) for f in staging.rglob("*") if f.is_file())
        size_mb = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file()) / 1e6
        print(f"[space] staged {len(files)} files, {size_mb:.1f} MB")
        if args.dry_run:
            for f in files:
                print("  ", f)
            return
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=str(staging),
                          commit_message="Publish retrieval demo")
        print(f"[space] pushed -> https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
