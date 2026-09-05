# Reproducing the full 38 GB data directory on a new machine

Nothing large is stored in git. Every asset either re-downloads from its
official source via a script in this repo, or comes from one small (~135 MB)
private HF bundle of files that had to be downloaded by hand.

Total download: ~38 GB. Python 3.10+ required; `pip install -r requirements.txt`.

## 1. Clone + install

```bash
git clone https://github.com/JitendraJha98/nyaya-model.git
cd nyaya-model
pip install -r requirements.txt
```

## 2. Hugging Face login (one-time)

Several datasets are gated. Create a free HF account, then:

```bash
huggingface-cli login
```

Accept the terms on each gated dataset page (open each URL once, click agree):
- https://huggingface.co/datasets/opennyaiorg/aalap_instruction_dataset
- https://huggingface.co/datasets/Exploration-Lab/IL-TUR
- https://huggingface.co/datasets/bharatgenai/BhashaBench-Legal
- https://huggingface.co/datasets/L-NLProc/NyayaAnumana-Classification-Data

## 3. Scripted downloads (everything except the manual bundle)

```bash
# 14 statute PDFs (English) from India Code, downloaded + extracted into the
# canonical corpus                                 -> data/raw/acts, data/canonical
python scripts/03_build_corpus.py                  # see configs/acts.yaml
# 14 more acts section-by-section from the India Code DSpace API (indiacode.gov.in)
python scripts/42_fetch_act_sections.py            # api_acts in configs/acts.yaml

# 12 Hindi statute PDFs + 4 procedure pages        -> data/raw/assets     (~160 MB)
python scripts/13_download_raw_assets.py

# HF datasets (5 enabled)                          -> data/hf             (~8 GB)
python scripts/00_download_hf_datasets.py

# SC judgments 1950-2025 + SC/HC metadata parquet  -> data/raw/judgments  (~29 GB)
# AWS Open Data, anonymous access, resumable — rerun if interrupted.
python scripts/12_bulk_judgments.py
```

All scripts are idempotent: already-downloaded items are skipped, so rerunning
after an interruption is always safe.

Note (Windows only): if console prints crash on Devanagari text, set
`PYTHONIOENCODING=utf-8`. Downloads run fine on Linux without this.

Troubleshooting — `CERTIFICATE_VERIFY_FAILED` on huggingface.co: machines
behind a TLS-inspecting corporate proxy have the proxy's root CA in the OS
trust store but not in Python's bundled `certifi` list. Fix without disabling
verification: `pip install pip-system-certs` (patches Python's SSL to use the
OS store). Side effect: urllib3 then prints spurious `InsecureRequestWarning`
lines on some code paths (e.g. boto3 downloads) — verification is still
active; we confirmed bad certificates are rejected.

## 4. Manual bundle (~135 MB, private HF dataset)

34 Law Commission report PDFs and the Parivahan FAQ were downloaded by hand
(JS-only portals) and live in the private dataset repo
`JitendraJha98/nyaya-manual-assets`. Ask Jitendra for access
(Settings → Collaborators on that repo), then:

```bash
huggingface-cli download JitendraJha98/nyaya-manual-assets \
  --repo-type dataset --local-dir data/raw/assets
```

This drops `law_commission_reports/*.pdf` and
`procedure_sources/parivahan_faq.txt` into place alongside the scripted assets.

## 5. Verify

`data/raw/assets/manifest.json` (included in the manual bundle) records the
SHA-256 of every asset. Check integrity with:

```bash
python - <<'EOF'
import hashlib, json, pathlib
root = pathlib.Path("data/raw/assets")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
missing = bad = ok = 0
for asset_id, meta in manifest.items():
    matches = list(root.rglob(f"{asset_id}.*"))
    if not matches:
        print("MISSING", asset_id); missing += 1; continue
    h = hashlib.sha256(matches[0].read_bytes()).hexdigest()
    if h != meta["sha256"]:
        print("CHECKSUM MISMATCH", asset_id); bad += 1
    else:
        ok += 1
print(f"{ok} ok, {missing} missing, {bad} mismatched")
EOF
```

Expected final layout:

```
data/
  raw/
    acts/                 14 statute PDFs + extracted text
    assets/               hindi_statutes, law_commission_reports,
                          procedure_sources, manifest.json
    judgments/            sc_judgments/, hc_judgments/, manifest.json
    citizen_questions.txt (versioned in git)
  hf/                     5 dataset dirs
```
