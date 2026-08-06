"""Step 24 — Publish Nyaya-3B-v3 to the Hugging Face Hub.

Pushes the MERGED full model (from the cluster PVC), an honest auto-generated
model card, and the training/eval artifacts to `NyayaLabs98/nyaya-3b-v3`.

Run this FROM THE CLUSTER (inside the pod / a K8s job) where the merged model
lives on the PVC — do not copy 6 GB down to a laptop and back up.

Eval numbers in the card are read from reports/ so the card can never drift
from the actual results. The card carries the mandatory "legal information,
not legal advice" disclaimer (docs/ROADMAP.md, Positioning & safety).

Auth: create a WRITE token for the NyayaLabs98 org, then either
    huggingface-cli login          (interactive, once)
    export HF_TOKEN=hf_...          (headless / K8s secret)

Usage:
    # dry run — build the card, upload nothing
    python scripts/24_publish_hf.py --model-dir /pvc/outputs/legal-3b-v3-merged --dry-run

    # publish the model + training artifacts
    python scripts/24_publish_hf.py --model-dir /pvc/outputs/legal-3b-v3-merged

    # also publish the training/RAG datasets (public)
    python scripts/24_publish_hf.py --model-dir /pvc/outputs/legal-3b-v3-merged --publish-datasets

    # ALSO publish the frozen eval set — IRREVERSIBLE, burns the benchmark
    python scripts/24_publish_hf.py --model-dir /pvc/outputs/legal-3b-v3-merged \
        --publish-datasets --publish-eval
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

ORG = "NyayaLabs98"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Qwen2.5-3B-Instruct is NOT Apache-2.0. Unlike the 1.5B/7B/14B/32B siblings it
# ships under the Qwen Research License — non-commercial, research use only.
# A merged LoRA is a derivative of those weights, so the release inherits that
# restriction. Publishing it as Apache-2.0 (as the first v3 card did) is a
# licence misstatement. Verify at https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
BASE_LICENSE = "other"
BASE_LICENSE_NAME = "qwen-research"
BASE_LICENSE_LINK = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE"

# Per-version release wiring. `eval_key` / `base_key` index reports/rag_eval_rescored.json;
# they MUST exist there or card generation aborts (see _need).
VERSIONS = {
    "v3": {
        "repo": f"{ORG}/nyaya-3b-v3",
        "eval_key": "rebase-r3_v3-ckpt300",
        "checkpoint": "checkpoint-300",
        "train_config": "train_v3.yaml",
        "history": "training_history_v3.json",
    },
    "v4": {
        "repo": f"{ORG}/nyaya-3b-v4",
        "eval_key": "rag_dense_k8_legal-3b-v4-checkpoint-354",
        "checkpoint": "checkpoint-354",
        "train_config": "train_v4.yaml",
        "history": "training_history_v4.json",
    },
}
BASE_EVAL_KEY = "rebase-r3_base"

# Datasets to publish (repo suffix -> local dir). Eval is handled separately.
DATASET_DIRS = {
    "nyaya-statute-db": ROOT / "data" / "canonical",
    "nyaya-train-v3": ROOT / "data" / "splits_rag_v3",
}
EVAL_REPO = f"{ORG}/nyaya-eval-v0"
EVAL_DIR = ROOT / "data" / "eval"


class CardDataError(RuntimeError):
    """A metric the card claims is missing from reports/ — refuse to publish."""


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "n/a"


def _need(runs: dict, key: str, field: str):
    """Fetch a metric or abort.

    The v3 card silently rendered 'n/a' after the re-baseline renamed the run
    keys, because the old lookup used .get() and _pct() swallowed the None.
    A card that quietly drops its own numbers is worse than one that fails to
    build, so a missing key is now fatal.
    """
    if key not in runs:
        raise CardDataError(
            f"run '{key}' not in reports/rag_eval_rescored.json "
            f"(have: {sorted(runs)}). Re-run scripts/17_rescore.py or fix VERSIONS."
        )
    if runs[key].get(field) is None:
        raise CardDataError(f"run '{key}' has no '{field}'")
    return runs[key][field]


def build_model_card(version: str) -> str:
    """Assemble the model card from the actual reports/ numbers.

    Raises CardDataError if any claimed metric is absent, so the card can
    never ship with a silently-blank or stale figure.
    """
    spec = VERSIONS[version]
    rescored = _load(ROOT / "reports" / "rag_eval_rescored.json") or {}
    runs = rescored.get("runs", {})

    key, base_key = spec["eval_key"], BASE_EVAL_KEY
    base_s = _need(runs, base_key, "strict_accuracy")
    base_l = _need(runs, base_key, "lenient_accuracy")
    m_s = _need(runs, key, "strict_accuracy")
    m_l = _need(runs, key, "lenient_accuracy")
    gic = runs[key].get("by_retrieval", {}).get("gold_in_context", {})
    base_gic = runs[base_key].get("by_retrieval", {}).get("gold_in_context", {})

    # Behavioural profile (abstention / grounding) comes from the raw eval report.
    raw = (_load(ROOT / "reports" / "rag_eval.json") or {}).get("runs", {})
    abst = raw.get(key, {}).get("abstention_rate")
    base_abst = raw.get(base_key, {}).get("abstention_rate")

    n_scored = runs[key].get("scored_total", "n/a")
    MODEL_REPO = spec["repo"]

    # YAML frontmatter — drives the Hub UI (base_model link, license, filters).
    frontmatter = f"""---
license: {BASE_LICENSE}
license_name: {BASE_LICENSE_NAME}
license_link: {BASE_LICENSE_LINK}
base_model: {BASE_MODEL}
base_model_relation: finetune
language:
- en
- hi
library_name: transformers
pipeline_tag: text-generation
tags:
- legal
- india
- indian-law
- bns
- retrieval-augmented-generation
- qwen2.5
- non-commercial
---"""

    body = f"""
# Nyaya-3B-{version} — Indian Legal Information Model

> **⚖️ Not legal advice.** Nyaya is a legal *information / guidance* model, **not a
> legal advisor**. The practice of law in India is reserved to enrolled advocates
> under the Advocates Act, 1961. Always consult a licensed advocate for anything
> consequential. Free legal aid is available via NALSA / DLSA (Legal Services
> Authorities Act, 1987).

> **📋 Non-commercial licence.** The base model `{BASE_MODEL}` is released under the
> **Qwen Research License** (`{BASE_LICENSE_NAME}`), *not* Apache-2.0 — the 3B is one of the
> Qwen2.5 sizes that carries the restricted licence. These merged weights are a
> derivative and inherit that restriction: **research / non-commercial use only**.
> See [the base model licence]({BASE_LICENSE_LINK}).

Nyaya answers everyday legal questions from Indian citizens in **English, Hindi,
and Hinglish**, cites specific sections of **current law** (BNS / BNSS / BSA,
post-July-2024, with IPC↔BNS bridging), and signals when a real lawyer is needed.

- **Base model:** [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) ({BASE_LICENSE_NAME}, non-commercial)
- **Method:** LoRA SFT (bf16, no quantization) using **RAFT** — teacher answers
  regenerated under the inference-time RAG prompt, including deliberate
  retrieval-miss demonstrations. Merged to full weights for release.
- **Designed to run with retrieval (RAG).** Statute passages must be supplied in
  context by a retriever; used bare, it behaves close to the base model. On the
  frozen benchmark it is **statistically tied with the base model** — the
  measurable differences are behavioural (abstention, citing, staying grounded),
  not accuracy gains. See the caveat under Evaluation.

## Intended use & limitations

- **Use for:** plain-language legal information, "how do I…" procedural guidance,
  old→new law (IPC↔BNS) mapping, terminology.
- **Do not use for:** actual legal advice, court filings, or any decision without
  a licensed advocate. Coverage is limited to the acts in the statute DB; it can
  be wrong or incomplete outside them.

## Evaluation (Nyaya-Eval-v0, 500 frozen questions)

All rows below come from the **same matched-retrieval run** (merged retriever,
dense fusion, k=8) so base and adapter are directly comparable. n={n_scored}
scored questions (safety rows graded separately).

| Setting | Strict | Lenient | Gold-in-context (strict) | Abstention |
|---|---|---|---|---|
| Base + dense RAG | {_pct(base_s)} | {_pct(base_l)} | {_pct(base_gic.get("strict"))} | {_pct(base_abst)} |
| **Nyaya-3B-{version} + dense RAG** | **{_pct(m_s)}** | **{_pct(m_l)}** | **{_pct(gic.get("strict"))}** | **{_pct(abst)}** |

### ⚠️ Read the strict metric carefully

`strict` requires **every** required fact of a question to appear, and ~85% of those
facts are free-text phrases matched as normalised substrings. Legally correct
paraphrases therefore fail: *"imprisonment for life or death"* does not match the
required fact `death or imprisonment for life`; *"5 or more people"* does not match
`five or more persons`. **These numbers measure verbatim phrasing agreement with the
eval author's wording, not legal correctness**, and base/v3/v4 land within a
2-answer spread of each other — i.e. inside noise.

**Do not read the strict column as an accuracy claim, and do not use it to rank
this model against others.** A partial-credit rebuild (Nyaya-Eval-v1) is in progress.

### Honest status

- The project's **human-eval ship gate has NOT been passed.**
- There is **no comparison against other legal or general models yet** — external
  benchmarks (BhashaBench-Legal, IL-TUR) have not been run.
- The retriever, not the model, is the current bottleneck: full-hit@8 is ~65%,
  and BSA-domain recall is ~36%.
- `nyaya-eval-v0` has been published publicly, so it is **contaminated** as a
  held-out benchmark from that point on.

This is an **early research release**, not a validated "best" legal model.

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{MODEL_REPO}")
model = AutoModelForCausalLM.from_pretrained("{MODEL_REPO}", device_map="auto")

system = (
    "You are Nyaya, an Indian legal information model. You provide accurate, "
    "plain-language legal guidance for Indian citizens, cite specific sections of "
    "current law (BNS/BNSS/BSA and other acts in force), clearly state uncertainty, "
    "and recommend consulting a licensed advocate for anything consequential. "
    "You provide legal information, not legal advice."
)
# For best results, prepend retrieved statute passages to the user turn (RAG).
messages = [
    {{"role": "system", "content": system}},
    {{"role": "user", "content": "Police FIR nahi likh rahi, kya karu?"}},
]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=512)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

## Training curves & ablations

![Program progression](training/figures/program_progression.png)
![RAG ablation](training/figures/rag_ablation.png)
![Checkpoint frozen-eval](training/figures/checkpoint_frozen_eval.png)
![Before/after behaviour](training/figures/behaviour_before_after.png)

## Training details

Released checkpoint: **{spec["checkpoint"]}**. See `training/` in this repo for the
exact config, metric history, and all figures (PNG + source CSV):
`{spec["train_config"]}`, `{spec["history"]}`, `rag_eval_rescored.json`, `figures/`.

## License & attribution

**Research / non-commercial use only.** These weights are a merged LoRA derivative
of [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}), which is licensed under the
**Qwen Research License** (`{BASE_LICENSE_NAME}`) — *not* Apache-2.0. The derivative
inherits that restriction; commercial use is **not** permitted.

The *training/eval code* in the source repository is Apache-2.0, but that licence
does not extend to these weights.

Statutory text is Government of India material (India Code / legislative.gov.in),
public domain under Section 52(1)(q) of the Copyright Act, 1957. Some aggregated
research datasets referenced by the project carry CC-BY-NC terms. See the
repository `NOTICE` for full data provenance.
"""
    return frontmatter + "\n" + body


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", choices=sorted(VERSIONS), default="v4",
                   help="Which release to publish (selects repo, eval run key, artifacts)")
    p.add_argument("--model-dir",
                   help="Path to the MERGED full model (config.json + *.safetensors). "
                        "Not needed with --card-only.")
    p.add_argument("--repo", help="Override the target repo (default: per --version)")
    p.add_argument("--card-only", action="store_true",
                   help="Push ONLY the model card to an existing repo — no weights. "
                        "Use this to correct a published card in place.")
    p.add_argument("--private", action="store_true", help="Create the model repo private")
    p.add_argument("--create-pr", action="store_true",
                   help="Open a Pull Request instead of committing to main. Needed when the "
                        "token can read the org but lacks org write (HF returns 403 with a "
                        "'pass create_pr=1' hint).")
    p.add_argument("--publish-datasets", action="store_true",
                   help="Also create + push the training/statute dataset repos (public)")
    p.add_argument("--publish-eval", action="store_true",
                   help="IRREVERSIBLE: also publish the frozen Nyaya-Eval-v0 set (contaminates the benchmark)")
    p.add_argument("--dry-run", action="store_true", help="Build the card + validate paths, upload nothing")
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

    # Fatal on any missing metric — never ship a card with blanks (see _need).
    try:
        card = build_model_card(args.version)
    except CardDataError as exc:
        sys.exit(f"[publish] refusing to build card: {exc}")

    if model_dir is not None:
        card_path = model_dir / "README.md"
        card_path.write_text(card, encoding="utf-8")
        print(f"[publish] wrote model card -> {card_path}  ({len(card)} chars)")

    if args.dry_run:
        print(f"[publish] --dry-run ({args.version} -> {repo}): nothing uploaded.\n")
        print(card)
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
            commit_message=f"Correct model card: licence ({BASE_LICENSE_NAME}), "
                           "matched-retrieval metrics, strict-metric caveat",
            commit_description=(
                "Corrects three defects in the published card:\n"
                f"1. Licence was apache-2.0; the base {BASE_MODEL} is "
                f"{BASE_LICENSE_NAME} (non-commercial) and a merged LoRA inherits it.\n"
                "2. The 90.2% dataset citation pass-rate was a v1 checkpoint-50 "
                "number presented as v3's.\n"
                "3. Eval table now uses the matched-retrieval re-baseline, states that "
                "base/v3/v4 are statistically tied, and explains that `strict` measures "
                "verbatim phrasing agreement rather than legal correctness."
            ),
        )
        url = getattr(res, "pr_url", None)
        if url:
            print(f"[publish] opened PR (token lacks org write) -> {url}")
            print("[publish] review and merge it from the Hub UI to make the fix live.")
        else:
            print(f"[publish] card-only update pushed -> https://huggingface.co/{repo}")
        return

    print(f"[publish] uploading merged model -> {repo} (this is the slow part)")
    upload_folder(repo_id=repo, folder_path=str(model_dir), repo_type="model",
                  commit_message=f"Nyaya-3B-{args.version}: merged weights + model card")

    # 2) training / eval artifacts alongside the model, under training/
    # NB: checkpoint_evals.json is deliberately NOT shipped — it holds v1
    # checkpoint numbers, and the v3 card previously quoted its 90.2% dataset
    # citation pass-rate as if it were v3's.
    artifacts = [
        ROOT / "configs" / spec["train_config"],
        ROOT / "reports" / spec["history"],
        ROOT / "reports" / "rag_eval_rescored.json",
        ROOT / "reports" / "retrieval_recall_dense.json",
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
                          commit_message=f"{suffix}: initial upload")
            print(f"[publish]   + dataset {ds_repo}")

    # 4) frozen eval — irreversible, explicit opt-in only
    if args.publish_eval:
        if not EVAL_DIR.exists():
            print(f"[publish]   ! --publish-eval set but {EVAL_DIR} not found")
        else:
            print("[publish]   !! publishing the FROZEN eval set — this permanently contaminates the benchmark")
            api.create_repo(EVAL_REPO, repo_type="dataset", exist_ok=True)
            upload_folder(repo_id=EVAL_REPO, folder_path=str(EVAL_DIR), repo_type="dataset",
                          commit_message="Nyaya-Eval-v0: frozen eval set")
            print(f"[publish]   + dataset {EVAL_REPO}")

    print(f"\n[publish] done -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
