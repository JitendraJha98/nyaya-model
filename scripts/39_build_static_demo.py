"""Step 39 — Build (and verify) the static, zero-backend retrieval demo.

Hugging Face charges for Gradio Spaces; static Spaces are free. So the demo is
a JavaScript port of the retriever (space-static/app.js) running in the
browser over an exported copy of the statute DB. This script:

  1. exports data.json from data/canonical plus the exact constants and
     tables the Python retriever uses (synonyms, act aliases, BM25 params),
     so the port and the source cannot drift on data;
  2. with --check, runs the JS retriever under Node on N questions and
     compares its top-k keys and coverage verdicts with nyaya.retrieval;
  3. with --publish, uploads space-static/ to the Hub as a static Space.

Usage:
    python scripts/39_build_static_demo.py                 # export only
    python scripts/39_build_static_demo.py --check 120     # export + parity check
    python scripts/39_build_static_demo.py --publish       # export + upload
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya import retrieval as R  # noqa: E402
from nyaya.validators import ACT_ALIASES  # noqa: E402

STATIC = ROOT / "space-static"
DATA_OUT = STATIC / "data.json"
SPACE_REPO = "NyayaLabs98/nyaya-demo"


def export() -> dict:
    index = R.load_statute_index(ROOT / "data" / "canonical")
    mappings = [json.loads(l) for l in (ROOT / "data" / "canonical" / "law_mappings.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
    data = {
        "rows": [{"act_id": r["act_id"], "act_name": r["act_name"], "section": r["section"],
                  "title": r.get("title") or "", "text": r.get("text") or "",
                  "tags": r.get("tags") or [], "punishment_summary": r.get("punishment_summary")}
                 for r in index.rows],
        "mappings": [{"old_act": m["old_act"], "old_section": m["old_section"],
                      "new_act": m["new_act"], "new_section": m["new_section"]} for m in mappings],
        "synonyms": R.LEGAL_SYNONYMS,
        "act_aliases": ACT_ALIASES,
        "family_to_act_id": R._FAMILY_TO_ACT_ID,
        "old_to_new_act": R._OLD_TO_NEW_ACT,
        "params": {"k1": 1.5, "b": 0.75, "title_bonus": R.TITLE_BONUS, "kb_slots": R.KB_SLOTS,
                   "guidance_floor_ratio": R.GUIDANCE_FLOOR_RATIO, "coverage_min_score": R.COVERAGE_MIN_SCORE},
    }
    STATIC.mkdir(exist_ok=True)
    DATA_OUT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[static] wrote {DATA_OUT} ({DATA_OUT.stat().st_size / 1e6:.1f} MB, {len(data['rows'])} rows)")
    return data


def check(n: int) -> bool:
    """Compare JS and Python retrieval on the first n public Eval-v1 questions
    plus every citizen question, on top-8 keys and the coverage verdict."""
    index = R.load_statute_index(ROOT / "data" / "canonical")
    questions = [json.loads(l)["question"] for l in (ROOT / "data" / "eval" / "nyaya_eval_v1_public.jsonl")
                 .read_text(encoding="utf-8").splitlines() if l.strip()][:n]
    questions += [l.strip() for l in (ROOT / "data" / "raw" / "citizen_questions.txt")
                  .read_text(encoding="utf-8").splitlines() if l.strip()]
    py = []
    for q in questions:
        keys = [f"{r['act_id']}:{r['section'].upper()}" for r in index.retrieve(q, k=8)]
        py.append({"q": q, "keys": keys, "covered": index.coverage(q)["covered"]})
    probe = STATIC / "_parity_probe.mjs"
    probe.write_text(f"""
import {{ readFileSync }} from "node:fs";
import {{ buildIndex }} from "./app.js";
const data = JSON.parse(readFileSync(new URL("./data.json", import.meta.url), "utf8"));
const idx = buildIndex(data);
const qs = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = qs.map(q => ({{ q, keys: idx.retrieve(q, 8).map(r => r.act_id + ":" + r.section.toUpperCase()),
                           covered: idx.coverage(q).covered }}));
process.stdout.write(JSON.stringify(out));
""", encoding="utf-8")
    qfile = STATIC / "_parity_questions.json"
    qfile.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    try:
        out = subprocess.run(["node", str(probe), str(qfile)], capture_output=True, text=True,
                             encoding="utf-8", check=True).stdout
    finally:
        probe.unlink(missing_ok=True)
        qfile.unlink(missing_ok=True)
    js = json.loads(out)
    key_mismatch = [(a["q"], a["keys"], b["keys"]) for a, b in zip(py, js) if a["keys"] != b["keys"]]
    cov_mismatch = [(a["q"], a["covered"], b["covered"]) for a, b in zip(py, js) if a["covered"] != b["covered"]]
    print(f"[parity] {len(questions)} questions: top-8 identical on {len(questions) - len(key_mismatch)}, "
          f"coverage identical on {len(questions) - len(cov_mismatch)}")
    for q, a, b in key_mismatch[:5]:
        print(f"  MISMATCH {q[:60]!r}\n    py {a}\n    js {b}")
    return not key_mismatch and not cov_mismatch


def publish(repo: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(STATIC),
                      ignore_patterns=["_parity_*"], commit_message="Publish static retrieval demo")
    print(f"[static] pushed -> https://huggingface.co/spaces/{repo}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", type=int, metavar="N", help="parity-check the JS port on N eval questions + all citizen questions")
    p.add_argument("--publish", action="store_true")
    p.add_argument("--repo", default=SPACE_REPO)
    args = p.parse_args()
    export()
    if args.check is not None and not check(args.check):
        sys.exit("[parity] JS and Python retrievers disagree — fix app.js before publishing")
    if args.publish:
        publish(args.repo)


if __name__ == "__main__":
    main()
