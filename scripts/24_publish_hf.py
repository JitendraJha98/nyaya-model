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
MODEL_REPO = f"{ORG}/nyaya-3b-v3"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Datasets to publish (repo suffix -> local dir). Eval is handled separately.
DATASET_DIRS = {
    "nyaya-statute-db": ROOT / "data" / "canonical",
    "nyaya-train-v3": ROOT / "data" / "splits_rag_v3",
}
EVAL_REPO = f"{ORG}/nyaya-eval-v0"
EVAL_DIR = ROOT / "data" / "eval"


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "n/a"


def build_model_card() -> str:
    """Assemble the model card from the actual reports/ numbers."""
    rescored = _load(ROOT / "reports" / "rag_eval_rescored.json") or {}
    ckpt = _load(ROOT / "reports" / "checkpoint_evals.json") or {}
    runs = rescored.get("runs", {})

    def acc(run):
        r = runs.get(run, {})
        return r.get("strict_accuracy"), r.get("lenient_accuracy")

    base_s, _ = acc("rag_dense_k8_base")
    v3_s, v3_l = acc("rag_dense_k8_legal-3b-v3-checkpoint-300")
    ds = (ckpt.get("best_dataset_eval") or {})
    cite = ds.get("citation_pass_rate")

    # YAML frontmatter — drives the Hub UI (base_model link, license, filters).
    frontmatter = f"""---
license: apache-2.0
base_model: {BASE_MODEL}
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
---"""

    body = f"""
# Nyaya-3B-v3 — Indian Legal Information Model

> **⚖️ Not legal advice.** Nyaya is a legal *information / guidance* model, **not a
> legal advisor**. The practice of law in India is reserved to enrolled advocates
> under the Advocates Act, 1961. Always consult a licensed advocate for anything
> consequential. Free legal aid is available via NALSA / DLSA (Legal Services
> Authorities Act, 1987).

Nyaya answers everyday legal questions from Indian citizens in **English, Hindi,
and Hinglish**, cites specific sections of **current law** (BNS / BNSS / BSA,
post-July-2024, with IPC↔BNS bridging), and signals when a real lawyer is needed.

- **Base model:** [`{BASE_MODEL}`]( https://huggingface.co/{BASE_MODEL}) (Apache-2.0)
- **Method:** LoRA SFT (bf16, no quantization) using **RAFT** — teacher answers
  regenerated under the inference-time RAG prompt, including deliberate
  retrieval-miss demonstrations. Merged to full weights for release.
- **Designed to run with retrieval (RAG).** The gains below come *with* a dense
  retriever supplying statute passages in context; used bare, it behaves close
  to the base model.

## Intended use & limitations

- **Use for:** plain-language legal information, "how do I…" procedural guidance,
  old→new law (IPC↔BNS) mapping, terminology.
- **Do not use for:** actual legal advice, court filings, or any decision without
  a licensed advocate. Coverage is limited to the acts in the statute DB; it can
  be wrong or incomplete outside them.

## Evaluation (Nyaya-Eval-v0, 500 frozen questions)

Primary metric is **citation accuracy verified against the statute DB**, not loss.
Numbers below are strict exact-match on the frozen eval, with dense RAG (k=8):

| Setting | Strict acc. | Lenient acc. |
|---|---|---|
| Base + dense RAG | {_pct(base_s)} | — |
| **Nyaya-3B-v3 + dense RAG** | **{_pct(v3_s)}** | **{_pct(v3_l)}** |

On the held-out *dataset* eval, citation pass-rate is **{_pct(cite)}** across
{ds.get("total", "n/a")} examples.

**Honest status:** strict exact-match on the frozen benchmark is still low and the
project's human-eval ship gate has **not** been passed. This is an early research
release, **not** a validated "best" legal model. See the repo's `docs/ROADMAP.md`
go/no-go gates.

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

See `training/` in this repo for the exact config, metric history, and all
figures (PNG + source CSV): `train_v3.yaml`, `training_history_v3.json`,
`rag_eval_rescored.json`, `checkpoint_evals.json`, `figures/`.

## License & attribution

Model weights released under **Apache-2.0**, inheriting the base model's license.
Statutory text is Government of India material (India Code / legislative.gov.in).
See the repository `NOTICE` for data provenance.
"""
    return frontmatter + "\n" + body


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True,
                   help="Path to the MERGED full model on the PVC (config.json + *.safetensors)")
    p.add_argument("--repo", default=MODEL_REPO)
    p.add_argument("--private", action="store_true", help="Create the model repo private")
    p.add_argument("--publish-datasets", action="store_true",
                   help="Also create + push the training/statute dataset repos (public)")
    p.add_argument("--publish-eval", action="store_true",
                   help="IRREVERSIBLE: also publish the frozen Nyaya-Eval-v0 set (contaminates the benchmark)")
    p.add_argument("--dry-run", action="store_true", help="Build the card + validate paths, upload nothing")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not (model_dir / "config.json").exists() or not weights:
        sys.exit(f"[publish] {model_dir} has no config.json / weight shards — is this the merged model dir?")

    card = build_model_card()
    card_path = model_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"[publish] wrote model card -> {card_path}  ({len(card)} chars)")

    if args.dry_run:
        print("[publish] --dry-run: nothing uploaded. Review the card above, then re-run without --dry-run.")
        print(card[:1200])
        return

    from huggingface_hub import HfApi, upload_folder
    api = HfApi()
    who = api.whoami()
    print(f"[publish] authenticated as: {who.get('name')}")

    # 1) model repo
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"[publish] uploading merged model -> {args.repo} (this is the slow part)")
    upload_folder(repo_id=args.repo, folder_path=str(model_dir), repo_type="model",
                  commit_message="Nyaya-3B-v3: merged weights + model card")

    # 2) training / eval artifacts alongside the model, under training/
    artifacts = [
        ROOT / "configs" / "train_v3.yaml",
        ROOT / "reports" / "training_history_v3.json",
        ROOT / "reports" / "rag_eval_rescored.json",
        ROOT / "reports" / "checkpoint_evals.json",
        ROOT / "reports" / "retrieval_recall_dense.json",
        ROOT / "reports" / "dpo_train_report.json",
    ]
    for a in artifacts:
        if a.exists():
            api.upload_file(path_or_fileobj=str(a), path_in_repo=f"training/{a.name}",
                            repo_id=args.repo, repo_type="model",
                            commit_message=f"training artifact: {a.name}")
            print(f"[publish]   + training/{a.name}")

    # training curves / ablation graphs (PNG + source CSV) -> training/figures/
    figures = ROOT / "reports" / "figures"
    if figures.exists():
        upload_folder(repo_id=args.repo, folder_path=str(figures), repo_type="model",
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
            repo = f"{ORG}/{suffix}"
            api.create_repo(repo, repo_type="dataset", exist_ok=True)
            upload_folder(repo_id=repo, folder_path=str(d), repo_type="dataset",
                          commit_message=f"{suffix}: initial upload")
            print(f"[publish]   + dataset {repo}")

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

    print(f"\n[publish] done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
