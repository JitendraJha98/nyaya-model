"""Step 31 — Publish a Nyaya-3B release to the Hugging Face Hub.

Pushes the MERGED full model (a merged model directory), the maintained model
card from docs/cards/, and the training/eval artifacts to
`NyayaLabs98/nyaya-3b-<version>`.

The card is prose about results and is maintained BY HAND in
docs/cards/nyaya-3b-<version>.md. This script only checks the parts that must
never regress (licence, frontmatter) before uploading it; every number in the
card must be traceable to a file in reports/ (docs/RELEASE_PLAN.md).

Auth: create a WRITE token for the NyayaLabs98 org, then either
    huggingface-cli login          (interactive, once)
    export HF_TOKEN=hf_...          (headless)

Usage:
    # dry run — validate the card and paths, upload nothing
    python scripts/31_publish_hf.py --version v3 --card-only --dry-run

    # push ONLY the card to the existing repo (the common case)
    python scripts/31_publish_hf.py --version v3 --card-only

    # publish the model + training artifacts
    python scripts/31_publish_hf.py --model-dir outputs/nyaya-3b-v3-merged

    # also publish the statute DB / training datasets (public)
    python scripts/31_publish_hf.py --model-dir outputs/nyaya-3b-v3-merged --publish-datasets

    # ALSO publish the frozen Eval-v0 set — IRREVERSIBLE, burns the benchmark.
    # Only EVAL_ALLOW_PATTERNS are uploaded; the private Eval-v1 split can
    # never leave data/eval/ through this script.
    python scripts/31_publish_hf.py --model-dir outputs/nyaya-3b-v3-merged \
        --publish-datasets --publish-eval
"""

import argparse
import datetime
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "docs" / "cards"

ORG = "NyayaLabs98"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Qwen2.5-3B-Instruct is NOT Apache-2.0. Unlike the 1.5B/7B/14B/32B siblings it
# ships under the Qwen Research License — non-commercial, research use only.
# A merged LoRA is a derivative of those weights, so the release inherits that
# restriction. Publishing it as Apache-2.0 (as the first v3 card did) is a
# licence misstatement. Verify at https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
BASE_LICENSE = "other"
BASE_LICENSE_NAME = "qwen-research"

# Per-version release wiring. v4/v5/v6 were never published: v4 tied with base
# on a metric later found broken, v5 and v6 regressed (docs/RESULTS.md §1).
VERSIONS = {
    "v3": {
        "repo": f"{ORG}/nyaya-3b-v3",
        "checkpoint": "checkpoint-300",
        "train_config": "train_v3.yaml",
        "history": "training_history_v3.json",
        "comparison": "eval_v1_comparison_nyaya-3b-v3.json",
    },
}

# Datasets to publish (repo suffix -> local dir). Eval is handled separately.
DATASET_DIRS = {
    "nyaya-statute-db": ROOT / "data" / "canonical",
    "nyaya-train-v3": ROOT / "data" / "splits_rag_v3",
}
EVAL_REPO = f"{ORG}/nyaya-eval-v0"
EVAL_DIR = ROOT / "data" / "eval"
# The ONLY files --publish-eval may upload. data/eval/ also holds the Eval-v1
# private split after scripts/25 has run locally; .gitignore protects git, this
# list protects the Hub.
EVAL_ALLOW_PATTERNS = ["nyaya_eval_v0.jsonl", "README.md"]


class CardDataError(RuntimeError):
    """The card is missing or fails a must-never-regress check — refuse to publish."""


def load_model_card(version: str) -> str:
    """Read docs/cards/nyaya-3b-<version>.md and check what must never regress.

    The first v3 card shipped with `license: apache-2.0`; the base model is
    qwen-research and a merged LoRA inherits it. That check lives here so the
    upload path cannot skip it.
    """
    path = CARDS / f"nyaya-3b-{version}.md"
    if not path.exists():
        raise CardDataError(f"no card at {path}")
    card = path.read_text(encoding="utf-8")
    parts = card.split("---")
    if len(parts) < 3 or not parts[0].strip() == "":
        raise CardDataError(f"{path.name}: missing YAML frontmatter")
    frontmatter = parts[1]
    for must in (f"license: {BASE_LICENSE}", f"license_name: {BASE_LICENSE_NAME}"):
        if must not in frontmatter:
            raise CardDataError(f"{path.name}: frontmatter lacks '{must}'")
    if "apache" in frontmatter.lower():
        raise CardDataError(f"{path.name}: frontmatter must not mention Apache for the weights")
    return card


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", choices=sorted(VERSIONS), default="v3",
                   help="Which release to publish (selects repo and artifacts)")
    p.add_argument("--model-dir",
                   help="Path to the MERGED full model (config.json + *.safetensors). "
                        "Not needed with --card-only.")
    p.add_argument("--repo", help="Override the target repo (default: per --version)")
    p.add_argument("--card-only", action="store_true",
                   help="Push ONLY the model card to an existing repo — no weights.")
    p.add_argument("--private", action="store_true", help="Create the model repo private")
    p.add_argument("--create-pr", action="store_true",
                   help="Open a Pull Request instead of committing to main. Needed when the "
                        "token can read the org but lacks org write (HF returns 403 with a "
                        "'pass create_pr=1' hint).")
    p.add_argument("--publish-datasets", action="store_true",
                   help="Also create + push the training/statute dataset repos (public)")
    p.add_argument("--publish-eval", action="store_true",
                   help="IRREVERSIBLE: also publish the frozen Nyaya-Eval-v0 set (contaminates the benchmark)")
    p.add_argument("--dry-run", action="store_true", help="Validate the card + paths, upload nothing")
    args = p.parse_args()

    spec = VERSIONS[args.version]
    repo = args.repo or spec["repo"]

    if not args.card_only and not args.model_dir:
        sys.exit("[publish] --model-dir is required unless --card-only is set")

    model_dir = Path(args.model_dir) if args.model_dir else None
    if model_dir is not None:
        weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
        if not (model_dir / "config.json").exists() or not weights:
            sys.exit(f"[publish] {model_dir} has no config.json / weight shards — is this the merged model dir?")

    try:
        card = load_model_card(args.version)
    except CardDataError as exc:
        sys.exit(f"[publish] refusing to publish: {exc}")

    if model_dir is not None:
        card_path = model_dir / "README.md"
        card_path.write_text(card, encoding="utf-8")
        print(f"[publish] wrote model card -> {card_path}  ({len(card)} chars)")

    if args.dry_run:
        print(f"[publish] --dry-run ({args.version} -> {repo}): card valid ({len(card)} chars), nothing uploaded.")
        return

    from huggingface_hub import HfApi, upload_folder
    api = HfApi()
    who = api.whoami()
    print(f"[publish] authenticated as: {who.get('name')}")

    # 1) model repo
    api.create_repo(repo, repo_type="model", private=args.private, exist_ok=True)

    if args.card_only:
        res = api.upload_file(
            path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
            repo_id=repo, repo_type="model", create_pr=args.create_pr or None,
            commit_message=f"Update model card ({datetime.date.today().isoformat()})",
        )
        url = getattr(res, "pr_url", None)
        if url:
            print(f"[publish] opened PR (token lacks org write) -> {url}")
            print("[publish] review and merge it from the Hub UI to make the change live.")
        else:
            print(f"[publish] card-only update pushed -> https://huggingface.co/{repo}")
        return

    print(f"[publish] uploading merged model -> {repo} (this is the slow part)")
    upload_folder(repo_id=repo, folder_path=str(model_dir), repo_type="model",
                  commit_message=f"Nyaya-3B-{args.version}: merged weights + model card")

    # 2) training / eval artifacts alongside the model, under training/
    # NB: checkpoint_evals.json is deliberately NOT shipped — it holds v1
    # checkpoint numbers, and the v3 card once quoted its 90.2% dataset
    # citation pass-rate as if it were v3's.
    artifacts = [
        ROOT / "configs" / spec["train_config"],
        ROOT / "reports" / spec["history"],
        ROOT / "reports" / "eval_v1_results.json",
        ROOT / "reports" / spec["comparison"],
        ROOT / "reports" / "retrieval_recall_rerank.json",
        ROOT / "reports" / "bhashabench_scores.json",
    ]
    for a in artifacts:
        if a.exists():
            api.upload_file(path_or_fileobj=str(a), path_in_repo=f"training/{a.name}",
                            repo_id=repo, repo_type="model",
                            commit_message=f"training artifact: {a.name}")
            print(f"[publish]   + training/{a.name}")
        else:
            print(f"[publish]   ! missing artifact: {a}")

    # training curves / ablation graphs (PNG + source CSV) -> training/figures/
    figures = ROOT / "reports" / "figures"
    if figures.exists():
        upload_folder(repo_id=repo, folder_path=str(figures), repo_type="model",
                      path_in_repo="training/figures",
                      allow_patterns=["*.png", "*.csv"],
                      commit_message="training curves + ablation figures")
        n = len(list(figures.glob("*.png")))
        print(f"[publish]   + training/figures/ ({n} graphs)")

    # 3) datasets (opt-in)
    if args.publish_datasets:
        for suffix, d in DATASET_DIRS.items():
            if not d.exists():
                print(f"[publish]   ! skip dataset {suffix}: {d} not found")
                continue
            ds_repo = f"{ORG}/{suffix}"
            api.create_repo(ds_repo, repo_type="dataset", exist_ok=True)
            upload_folder(repo_id=ds_repo, folder_path=str(d), repo_type="dataset",
                          commit_message=f"{suffix}: upload")
            print(f"[publish]   + dataset {ds_repo}")

    # 4) frozen eval — irreversible, explicit opt-in, allow-listed files only
    if args.publish_eval:
        if not EVAL_DIR.exists():
            print(f"[publish]   ! --publish-eval set but {EVAL_DIR} not found")
        else:
            print("[publish]   !! publishing the FROZEN eval set — this permanently contaminates the benchmark")
            api.create_repo(EVAL_REPO, repo_type="dataset", exist_ok=True)
            upload_folder(repo_id=EVAL_REPO, folder_path=str(EVAL_DIR), repo_type="dataset",
                          allow_patterns=EVAL_ALLOW_PATTERNS,
                          commit_message="Nyaya-Eval-v0: frozen eval set")
            print(f"[publish]   + dataset {EVAL_REPO} ({', '.join(EVAL_ALLOW_PATTERNS)})")

    print(f"\n[publish] done -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
