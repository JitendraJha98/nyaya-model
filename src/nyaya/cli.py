"""`nyaya` command line: show the sections of current Indian law a question retrieves.

Retrieval only, standard library only. The statute DB is read from
data/canonical when run inside the repository; otherwise it is downloaded once
from the Hub dataset NyayaLabs98/nyaya-statute-db (about 5 MB).
"""
import argparse
from pathlib import Path

from .retrieval import format_context, load_statute_index

HUB_DATASET = "NyayaLabs98/nyaya-statute-db"
DISCLAIMER = ("⚖️ Legal information, not legal advice. Consult a licensed advocate; "
              "free legal aid: NALSA / DLSA.")


def _canonical_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    local = Path(__file__).resolve().parents[2] / "data" / "canonical"
    if local.exists():
        return local
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(HUB_DATASET, repo_type="dataset", allow_patterns=["*.jsonl"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nyaya", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="show the statute sections a question retrieves")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=5, help="statute sections to show (default 5)")
    ask.add_argument("--canonical-dir", default=None,
                     help="override the statute DB directory (default: repo data/canonical, else the Hub copy)")
    args = p.parse_args(argv)

    index = load_statute_index(_canonical_dir(args.canonical_dir))
    rows = index.retrieve(args.question, k=args.k)
    print(format_context(rows) if rows else "No matching sections in the acts this database holds.")
    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
