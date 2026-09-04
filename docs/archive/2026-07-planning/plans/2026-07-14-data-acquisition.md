# Data Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Acquire every data asset the Nyaya roadmap needs — 13 statute extractions + Constitution, Hindi raw texts, HF datasets, procedure/report sources, and a capped bulk judgment archive — at zero cost.

**Architecture:** Extend the existing `configs/acts.yaml` → `scripts/03_build_corpus.py` → `data/canonical/` pipeline for the remaining acts and the Constitution (new article splitter in `src/nyaya/corpus.py`). Add two new acquisition scripts: a generic manifest-tracked raw-asset downloader (Hindi texts, Law Commission reports, AIBE papers, procedure sources) and a resumable, size-capped bulk judgment downloader over the AWS Open Data S3 archives (anonymous access).

**Tech Stack:** Python 3.14 (`C:\Python314\python.exe` — plain `python` is NOT on Bash PATH), requests, PyMuPDF, PyYAML, boto3 (anonymous/UNSIGNED), pytest.

## Global Constraints

- **Zero cost:** no paid APIs, no API keys, no AWS account. S3 access is anonymous (`signature_version=UNSIGNED`). HF gated datasets use only a free `huggingface-cli login`.
- **Polite scraping:** ≤1 request/sec to government portals; browser User-Agent + Referer for indiacode.nic.in (existing `BROWSER_HEADERS` pattern in `scripts/03_build_corpus.py:80`).
- **Bulk cap:** total bytes under `data/raw/judgments/` must stay ≤ 100 GB (config-enforced, default cap 80 GB to leave headroom).
- **No ToS violations:** never scrape indiankanoon.org. Official sources only.
- **Failure degrades to a task list:** a source that can't be fetched by script is reported in a `manual_downloads.json` list, never a crash.
- **Run tests with:** `/c/Python314/python.exe -m pytest tests -q --ignore=tests/test_download_script.py` (that module needs `datasets`; include it once Task 6 installs deps).
- All downloads land under `data/raw/` (gitignored); only configs, code, reports, and canonical JSONL are committed.

---

### Task 1: Verify and enable Consumer Protection Act 2019 (template act)

No new code — this task establishes the verification procedure reused in Task 2.

**Files:**
- Modify: `configs/acts.yaml` (the `cpa_2019` entry, lines 62–68)

**Interfaces:**
- Consumes: `scripts/03_build_corpus.py --act <act_id>` (exits non-zero if <98% clean or download fails; writes `data/canonical/<act_id>.jsonl`, `reports/corpus_extraction_report.json`, `reports/corpus_spotcheck_sample.json`)
- Produces: `data/canonical/cpa_2019.jsonl` and the documented per-act verification procedure below.

- [ ] **Step 1: Run the single-act build**

```bash
cd "C:\Users\mjwea\Desktop\nyaya-model" && /c/Python314/python.exe scripts/03_build_corpus.py --act cpa_2019
```

Expected: `[cpa_2019] Consumer Protection Act, 2019` … `sections=107 expected=107 clean=100.00%` (or close). If the download 403s/404s, open the `handle_url` from `configs/acts.yaml` in a browser, re-scrape the current PDF link, update `url`, retry once; if it still fails, record the act in the manual list and move on.

- [ ] **Step 2: Spot-check the sample**

Open `reports/corpus_spotcheck_sample.json`, key `cpa_2019` (~5% of sections). For each sampled row check: section number matches title, body text is complete prose (no mid-sentence truncation, no page-number noise, no footnote fragments). Record pass/fail count in the commit message.

- [ ] **Step 3: Flip `enabled: true`**

In `configs/acts.yaml` set `cpa_2019.enabled: true` and set `expected_sections` to the verified count if it was null.

- [ ] **Step 4: Re-run the full gate**

```bash
/c/Python314/python.exe scripts/03_build_corpus.py --skip-download
```

Expected: `[gate] all acts >= 98% clean`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add configs/acts.yaml data/canonical/cpa_2019.jsonl reports/
git commit -m "Enable CPA 2019 in statute DB: <N> sections, <X>% clean, spot-check <k>/<k> pass"
```

---

### Task 2: Verify and enable the remaining 8 acts

**Files:**
- Modify: `configs/acts.yaml` (entries: `dv_act_2005`, `posh_2013`, `hma_1955`, `sma_1954`, `ni_act_1881`, `mv_act_1988`, `it_act_2000`, `wages_code_2019`)

**Interfaces:**
- Consumes: the Task 1 procedure, verbatim, per act.
- Produces: `data/canonical/<act_id>.jsonl` for all 8; all 12 non-Constitution acts `enabled: true`.

- [ ] **Step 1: Small acts first** — repeat Task 1 Steps 1–3 for `dv_act_2005` (expect 37), `posh_2013` (expect 30), `wages_code_2019` (expect 69), `hma_1955`, `sma_1954`.
- [ ] **Step 2: Hard acts** — repeat for `ni_act_1881` (letter-suffixed sections like 138, 143A), `it_act_2000` (heavy amendment footnotes — check the footnote filter output carefully in the spot-check), `mv_act_1988` (largest; expect >200 sections).
  - For the three acts with `expected_sections: null`: after extraction, cross-check the count against the act's last section number in the PDF's final pages; write the verified count into `expected_sections`.
  - If any act extracts <98% clean, do NOT enable it; instead file its failure mode in `reports/corpus_extraction_report.json` notes and leave a `# BLOCKED: <reason>` comment on its yaml entry. The gate must stay green.
- [ ] **Step 3: Full-gate run** — `/c/Python314/python.exe scripts/03_build_corpus.py --skip-download` → exit 0, all enabled acts ≥98%.
- [ ] **Step 4: Full test suite** — `/c/Python314/python.exe -m pytest tests -q --ignore=tests/test_download_script.py` → all pass.
- [ ] **Step 5: Commit** — one commit per act or one batch commit listing per-act counts/clean%:

```bash
git add configs/acts.yaml data/canonical/ reports/
git commit -m "Enable 8 remaining acts in statute DB (DV, POSH, Wages, HMA, SMA, NI, IT, MV)"
```

---

### Task 3: Constitution article splitter

The Constitution uses Articles/Parts, not Sections/Chapters — `split_sections` cannot split it. Add `split_articles` + a `splitter: articles` config switch.

**Files:**
- Modify: `src/nyaya/corpus.py` (add `_PART`, `_ARTICLE_START`, `split_articles`)
- Modify: `scripts/03_build_corpus.py:135-168` (`build_act`: choose splitter from `act.get("splitter")`)
- Test: `tests/test_corpus.py` (append new test class)

**Interfaces:**
- Consumes: `slice_act_body(text) -> str` (existing; the enacting-formula regex `_ENACTING` must also match the Constitution's "WE, THE PEOPLE OF INDIA" preamble — check `_ENACTING` in `src/nyaya/corpus.py` and extend its alternation if needed).
- Produces: `split_articles(text: str) -> list[dict]` with keys `section` (article number as str, e.g. `"21A"`), `title`, `text`, `chapter` (the PART label, e.g. `"PART III — Fundamental Rights"`). Same row shape as `split_sections` so `build_act` needs no other change.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_corpus.py`:

```python
from nyaya.corpus import split_articles


CONSTITUTION_SNIPPET = """PART I
THE UNION AND ITS TERRITORY
1. Name and territory of the Union.—(1) India, that is Bharat, shall be a
Union of States.
(2) The States and the territories thereof shall be as specified in the First
Schedule.
2. Admission or establishment of new States.—Parliament may by law admit
into the Union, or establish, new States.
PART III
FUNDAMENTAL RIGHTS
21. Protection of life and personal liberty.—No person shall be deprived of
his life or personal liberty except according to procedure established by law.
21A. Right to education.—The State shall provide free and compulsory
education to all children of the age of six to fourteen years.
"""


class TestSplitArticles:
    def test_splits_articles_with_part_attribution(self):
        arts = split_articles(CONSTITUTION_SNIPPET)
        nums = [a["section"] for a in arts]
        assert nums == ["1", "2", "21", "21A"]
        assert arts[0]["title"] == "Name and territory of the Union"
        assert "Union of States" in arts[0]["text"]
        assert arts[0]["chapter"] == "PART I — The Union And Its Territory"
        assert arts[3]["chapter"] == "PART III — Fundamental Rights"

    def test_em_dash_separates_title_from_body(self):
        arts = split_articles(CONSTITUTION_SNIPPET)
        # title must not swallow the body after the em dash
        assert arts[2]["title"] == "Protection of life and personal liberty"
        assert arts[2]["text"].startswith("No person shall be deprived")
```

- [ ] **Step 2: Run to verify failure**

`/c/Python314/python.exe -m pytest tests/test_corpus.py -q -k SplitArticles`
Expected: FAIL — `ImportError: cannot import name 'split_articles'`.

- [ ] **Step 3: Implement**

Add to `src/nyaya/corpus.py` (after `split_sections`):

```python
_PART = re.compile(r"^[ \t]*(PART\s+[IVXLC]+[A-Z]?)\s*$", re.MULTILINE)
# "21A. Right to education.—The State…": number, title up to the em dash,
# then the body. Titles without an em dash (rare) end at the line break.
_ARTICLE_START = re.compile(
    r"^[ \t]*(\d{1,3}[A-Z]{0,2})\.\s+(.+?)\.?\s*(?:—|—)", re.MULTILINE
)


def split_articles(text: str) -> list[dict]:
    """Split Constitution text into Articles with PART attribution.

    Same row shape as split_sections ('section' holds the article number)
    so downstream code treats the Constitution like any other act.
    """
    parts = []  # (position, "PART III — Fundamental Rights")
    for m in _PART.finditer(text):
        label = m.group(1).strip()
        rest = text[m.end():].lstrip("\n")
        heading = rest.split("\n", 1)[0].strip()
        if heading and heading == heading.upper() and not _ARTICLE_START.match(rest):
            label = f"{label} — {heading.title()}"
        parts.append((m.start(), label))

    def part_at(pos):
        current = None
        for ppos, label in parts:
            if ppos <= pos:
                current = label
            else:
                break
        return current

    articles = []
    matches = list(_ARTICLE_START.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end]
        body = _PART.split(body)[0]
        body = re.sub(r"\s+", " ", body).strip()
        articles.append(
            {
                "section": m.group(1),
                "title": re.sub(r"\s+", " ", m.group(2)).strip(),
                "text": body,
                "chapter": part_at(m.start()),
            }
        )
    return articles
```

In `scripts/03_build_corpus.py`, import `split_articles` and change one line in `build_act`:

```python
splitter = split_articles if act.get("splitter") == "articles" else split_sections
sections = splitter(slice_act_body(text))
```

- [ ] **Step 4: Run tests** — `/c/Python314/python.exe -m pytest tests/test_corpus.py tests/test_build_corpus.py -q` → all PASS.
- [ ] **Step 5: Commit**

```bash
git add src/nyaya/corpus.py scripts/03_build_corpus.py tests/test_corpus.py
git commit -m "Add Constitution article splitter (split_articles + splitter: articles config)"
```

---

### Task 4: Extract and enable the Constitution

**Files:**
- Modify: `configs/acts.yaml` (`constitution_1950` entry: add `splitter: articles`, flip `enabled`)

**Interfaces:**
- Consumes: Task 3's `splitter: articles` switch; Task 1's verification procedure.
- Produces: `data/canonical/constitution_1950.jsonl` (~395+ article rows).

- [ ] **Step 1:** Add `splitter: articles` to the `constitution_1950` yaml entry.
- [ ] **Step 2:** Run `/c/Python314/python.exe scripts/03_build_corpus.py --act constitution_1950`. The real PDF will surface splitter edge cases (repealed articles "[Repealed.]", articles like 243ZH, footnotes) — iterate on `_ARTICLE_START`/`slice_act_body` with new regression tests in `tests/test_corpus.py` for each real failure found, until the clean fraction is ≥98%.
- [ ] **Step 3:** Spot-check per Task 1 Step 2 (pay attention to Part IVA, Articles 31A–31C, 239AA — heavy amendment history).
- [ ] **Step 4:** Set `expected_sections` to the extracted verified count, flip `enabled: true`, run the full gate + full test suite (commands as in Task 2 Steps 3–4).
- [ ] **Step 5:** Commit: `git add -A && git commit -m "Extract Constitution of India: <N> articles, <X>% clean"`.

---

### Task 5: Generic raw-asset downloader (Hindi texts, Law Commission reports, AIBE papers, procedure sources)

One config + one script for every "just fetch these official files" need (design Stages A-Hindi and C).

**Files:**
- Create: `configs/raw_assets.yaml`
- Create: `scripts/13_download_raw_assets.py`
- Test: `tests/test_raw_assets.py`

**Interfaces:**
- Consumes: nothing from other tasks (URL verification happens inside this task).
- Produces: `data/raw/assets/<group>/<asset_id>.<ext>` files, `data/raw/assets/manifest.json` (`{asset_id: {url, sha256, bytes, fetched_at}}`), `reports/manual_downloads.json` (list of `{asset_id, url, reason}` for anything the script couldn't fetch).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_raw_assets.py
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

download_raw_assets = __import__("13_download_raw_assets")


class FakeResponse:
    def __init__(self, content=b"%PDF-1.7 fake", status=200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise download_raw_assets.requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses = responses  # url -> FakeResponse
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(url)
        return self.responses.get(url, FakeResponse(status=404))


def test_fetch_group_writes_files_and_manifest(tmp_path):
    assets = [
        {"asset_id": "coi_hindi", "group": "hindi_statutes",
         "url": "https://example.gov.in/coi_hi.pdf", "filetype": "pdf"},
    ]
    session = FakeSession({"https://example.gov.in/coi_hi.pdf": FakeResponse()})
    result = download_raw_assets.fetch_assets(assets, session, out_dir=tmp_path, delay=0)
    saved = tmp_path / "hindi_statutes" / "coi_hindi.pdf"
    assert saved.exists() and saved.read_bytes().startswith(b"%PDF")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["coi_hindi"]["bytes"] == saved.stat().st_size
    assert result.failed == []


def test_failed_fetch_goes_to_manual_list_not_crash(tmp_path):
    assets = [{"asset_id": "gone", "group": "g",
               "url": "https://example.gov.in/404.pdf", "filetype": "pdf"}]
    result = download_raw_assets.fetch_assets(assets, FakeSession({}), out_dir=tmp_path, delay=0)
    assert result.failed == [{"asset_id": "gone",
                              "url": "https://example.gov.in/404.pdf",
                              "reason": "download failed after retries"}]


def test_cached_asset_is_not_refetched(tmp_path):
    assets = [{"asset_id": "a", "group": "g",
               "url": "https://example.gov.in/a.pdf", "filetype": "pdf"}]
    session = FakeSession({"https://example.gov.in/a.pdf": FakeResponse()})
    download_raw_assets.fetch_assets(assets, session, out_dir=tmp_path, delay=0)
    download_raw_assets.fetch_assets(assets, session, out_dir=tmp_path, delay=0)
    assert len(session.calls) == 1
```

- [ ] **Step 2: Run to verify failure** — `/c/Python314/python.exe -m pytest tests/test_raw_assets.py -q` → FAIL (module not found).

- [ ] **Step 3: Implement `scripts/13_download_raw_assets.py`**

```python
"""Download raw official assets listed in configs/raw_assets.yaml.

Generic fetcher for every 'just get these official files' need: Hindi statute
texts, Law Commission reports, AIBE papers, procedure-KB sources. Writes
data/raw/assets/<group>/<asset_id>.<ext>, a manifest with checksums, and a
manual-download task list for anything it could not fetch (never crashes on a
bad source).

Usage:
    python scripts/13_download_raw_assets.py                 # all groups
    python scripts/13_download_raw_assets.py --group hindi_statutes
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "raw_assets.yaml"
OUT_DIR = ROOT / "data" / "raw" / "assets"
REPORTS = ROOT / "reports"
RETRIES = 3

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
}


@dataclass
class FetchResult:
    fetched: list = field(default_factory=list)
    cached: list = field(default_factory=list)
    failed: list = field(default_factory=list)


def _load_manifest(out_dir: Path) -> dict:
    p = out_dir / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def fetch_assets(assets: list[dict], session, out_dir: Path = OUT_DIR,
                 delay: float = 1.0) -> FetchResult:
    """Fetch each asset once; polite delay between requests; manifest-tracked."""
    result = FetchResult()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(out_dir)
    for asset in assets:
        dest = out_dir / asset["group"] / f"{asset['asset_id']}.{asset['filetype']}"
        if dest.exists() and dest.stat().st_size > 0:
            result.cached.append(asset["asset_id"])
            continue
        headers = dict(BROWSER_HEADERS)
        if asset.get("referer"):
            headers["Referer"] = asset["referer"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = None
        for attempt in range(RETRIES):
            try:
                resp = session.get(asset["url"], headers=headers, timeout=180)
                resp.raise_for_status()
                content = resp.content
                break
            except requests.RequestException:
                if attempt < RETRIES - 1:
                    time.sleep(delay * (attempt + 1))
        if content is None:
            result.failed.append({"asset_id": asset["asset_id"], "url": asset["url"],
                                  "reason": "download failed after retries"})
            continue
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(dest)
        manifest[asset["asset_id"]] = {
            "url": asset["url"],
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        result.fetched.append(asset["asset_id"])
        time.sleep(delay)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", help="fetch a single group from the config")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assets = [a for a in config["assets"]
              if not args.group or a["group"] == args.group]
    result = fetch_assets(assets, requests.Session())
    print(f"fetched={len(result.fetched)} cached={len(result.cached)} "
          f"failed={len(result.failed)}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "manual_downloads.json").write_text(
        json.dumps(result.failed, indent=2), encoding="utf-8")
    if result.failed:
        print(f"[manual] {len(result.failed)} assets need manual download — "
              f"see reports/manual_downloads.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests** — `/c/Python314/python.exe -m pytest tests/test_raw_assets.py -q` → 3 PASS.

- [ ] **Step 5: Build `configs/raw_assets.yaml` with VERIFIED URLs**

Verify every URL live before writing it into the config (use WebSearch/WebFetch or `curl -sI <url>` and require HTTP 200 + correct content type). Groups and targets to source — official portals only:

- `hindi_statutes`: Constitution of India (Hindi) from legislative.gov.in; Hindi BNS/BNSS/BSA from India Code handle pages (each act's handle page links a Hindi PDF — follow `handle_url` in `configs/acts.yaml`).
- `law_commission_reports`: from lawcommissionofindia.nic.in / DoJ — start with reports 262 (death penalty), 268 (bail), 277 (wrongful prosecution), and any report cited by the 13 priority acts' subject areas. 10–15 reports is enough to start.
- `aibe_papers`: AIBE previous-year question papers + answer keys from allindiabarexamination.com (official Bar Council site) — for the eval harness, not training.
- `procedure_sources`: NALSA publications page PDFs (legal-aid procedures), rtionline.gov.in FAQ page, consumerhelpline.gov.in complaint-process pages, cybercrime.gov.in citizen-manual PDF, parivahan.gov.in challan/licence FAQ pages. For HTML pages set `filetype: html` (the fetcher already saves any bytes).

Any target whose URL cannot be verified goes into the config commented out with `# UNVERIFIED — manual`, and is listed for the user. Do not guess URLs into the live config.

- [ ] **Step 6: Run the fetcher** — `/c/Python314/python.exe scripts/13_download_raw_assets.py` → check `fetched=N … failed=M` output and `reports/manual_downloads.json`.
- [ ] **Step 7: Commit** (config + script + tests + report; raw files stay gitignored):

```bash
git add configs/raw_assets.yaml scripts/13_download_raw_assets.py tests/test_raw_assets.py reports/manual_downloads.json
git commit -m "Add manifest-tracked raw-asset downloader + verified official source registry"
```

---

### Task 6: Verify HF dataset IDs, add slice support, download

**Files:**
- Modify: `configs/hf_datasets.yaml`
- Modify: `scripts/00_download_hf_datasets.py` (pass optional `name`/`split` to `load_dataset`)
- Test: `tests/test_download_script.py` (extend existing tests)

**Interfaces:**
- Consumes: free HF web API (`https://huggingface.co/api/datasets?search=<q>` — no key needed).
- Produces: populated `data/hf/<safe-name>/` dirs; config entries gain optional `name:` (HF config/subset) and `split:` keys honored by `download()`.

- [ ] **Step 1: Install deps** — `/c/Python314/python.exe -m pip install -r requirements.txt` (gets `datasets`). Then run the previously-skipped module: `/c/Python314/python.exe -m pytest tests/test_download_script.py -q` → PASS.
- [ ] **Step 2: Verify hub IDs** for the four unverified datasets by querying the free API:

```bash
curl -s "https://huggingface.co/api/datasets?search=BhashaBench" | /c/Python314/python.exe -c "import json,sys; print([d['id'] for d in json.load(sys.stdin)])"
curl -s "https://huggingface.co/api/datasets?search=NyayaAnumana" | /c/Python314/python.exe -c "import json,sys; print([d['id'] for d in json.load(sys.stdin)])"
curl -s "https://huggingface.co/api/datasets?search=MILDSum" | /c/Python314/python.exe -c "import json,sys; print([d['id'] for d in json.load(sys.stdin)])"
curl -s "https://huggingface.co/api/datasets?search=ILDC" | /c/Python314/python.exe -c "import json,sys; print([d['id'] for d in json.load(sys.stdin)])"
```

Update each entry's `id:` in `configs/hf_datasets.yaml` to the canonical hit and flip `enabled: true`. For each verified dataset also fetch its dataset card (`https://huggingface.co/api/datasets/<id>`) and record license + available configs in the yaml `purpose:` line. If a dataset requires a signed agreement (ILDC historically does), leave it `enabled: false` with a note — do not chase it.

- [ ] **Step 3: Failing test for slice support** — add to `tests/test_download_script.py` a test that a config entry `{id: x, name: single, split: train}` results in `load_dataset("x", "single", split="train")` (mock `load_dataset` as the existing tests in that file do — follow their fixture pattern).
- [ ] **Step 4: Implement** — in `scripts/00_download_hf_datasets.py::download`, read the entry dict instead of bare id where needed and call `load_dataset(dataset_id, entry.get("name"), split=entry.get("split"))`, filtering out `None` args. For NyayaAnumana set the smallest judgment-prediction config (check its card for config names, e.g. a "single"/SC-only subset) — the full ~700K-case corpus is NOT needed.
- [ ] **Step 5: Run tests** — `/c/Python314/python.exe -m pytest tests/test_download_script.py -q` → PASS. Commit code + config:

```bash
git add configs/hf_datasets.yaml scripts/00_download_hf_datasets.py tests/test_download_script.py
git commit -m "Verify HF dataset ids; support name/split slices in downloader"
```

- [ ] **Step 6: Download** — `huggingface-cli login` if gated (user does this once), then `/c/Python314/python.exe scripts/00_download_hf_datasets.py` (long-running; use run_in_background). Verify each `data/hf/*/` is non-empty and row counts roughly match the dataset cards; append actual counts to the yaml `purpose:` lines and commit the config touch-up.

---

### Task 7: Bulk judgment downloader (Stage D, capped)

**Files:**
- Create: `configs/bulk_sources.yaml`
- Create: `scripts/12_bulk_judgments.py`
- Test: `tests/test_bulk_judgments.py`
- Modify: `requirements.txt` (add `boto3`)

**Interfaces:**
- Consumes: AWS Open Data Registry public S3 buckets, anonymous access.
- Produces: `data/raw/judgments/<source>/<key>` files + `data/raw/judgments/manifest.json`; pure functions `select_keys(keys: list[dict], prefixes: list[str], max_bytes: int, already: set[str]) -> list[dict]` (each key dict: `{"Key": str, "Size": int}`).

- [ ] **Step 1: Confirm bucket names from the Open Data Registry** (data-driven — do not trust memory):

```bash
curl -s https://raw.githubusercontent.com/awslabs/open-data-registry/main/datasets/indian-high-court-judgments.yaml
curl -s https://raw.githubusercontent.com/awslabs/open-data-registry/main/datasets/indian-supreme-court-judgments.yaml
```

Read `Resources[].ARN` for the bucket names/regions. If either file 404s, list the registry dir (`https://github.com/awslabs/open-data-registry/tree/main/datasets`) and search for "india" datasets; also check `registry.opendata.aws`. Record the confirmed bucket names + prefix layout (court/year structure) in `configs/bulk_sources.yaml` comments.

- [ ] **Step 2: Write `configs/bulk_sources.yaml`**

```yaml
# Bulk judgment archives — AWS Open Data, anonymous access (no account/key).
# Cap is enforced across ALL sources combined; metadata/parquet prefixes first,
# raw PDFs only for the SC archive (HC PDFs would blow the cap).
max_total_gb: 80

sources:
  - source_id: sc_judgments
    bucket: "<CONFIRMED-IN-STEP-1>"
    region: "<CONFIRMED-IN-STEP-1>"
    prefixes: []            # empty = whole bucket (SC archive is small)
    enabled: true

  - source_id: hc_judgments
    bucket: "<CONFIRMED-IN-STEP-1>"
    region: "<CONFIRMED-IN-STEP-1>"
    # Filtered slices only: metadata + selected benches/years. Fill after
    # inspecting the bucket's prefix layout in Step 5, e.g.:
    #   - "metadata/parquet/"
    #   - "data/pdf/year=2024/court=xx~delhi/"
    prefixes: ["metadata/"]
    enabled: true
```

(The two `<CONFIRMED-IN-STEP-1>` values are filled in Step 1 of THIS task before the file is committed — never committed as placeholders.)

- [ ] **Step 3: Failing tests for the pure selection logic**

```python
# tests/test_bulk_judgments.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

bulk = __import__("12_bulk_judgments")

KEYS = [
    {"Key": "metadata/a.parquet", "Size": 100},
    {"Key": "metadata/b.parquet", "Size": 200},
    {"Key": "pdf/2024/x.pdf", "Size": 1000},
    {"Key": "pdf/2010/y.pdf", "Size": 1000},
]


def test_prefix_filter():
    picked = bulk.select_keys(KEYS, prefixes=["metadata/"], max_bytes=10**9, already=set())
    assert [k["Key"] for k in picked] == ["metadata/a.parquet", "metadata/b.parquet"]


def test_empty_prefixes_means_everything():
    picked = bulk.select_keys(KEYS, prefixes=[], max_bytes=10**9, already=set())
    assert len(picked) == 4


def test_byte_cap_stops_selection():
    picked = bulk.select_keys(KEYS, prefixes=[], max_bytes=350, already=set())
    assert [k["Key"] for k in picked] == ["metadata/a.parquet", "metadata/b.parquet"]


def test_already_downloaded_skipped_and_counted_against_cap():
    picked = bulk.select_keys(KEYS, prefixes=["metadata/"], max_bytes=350,
                              already={"metadata/a.parquet"})
    # a.parquet's 100 bytes count toward the 350 cap (100+200 <= 350), but the
    # file itself is not re-fetched — only b is downloaded
    assert [k["Key"] for k in picked] == ["metadata/b.parquet"]
```

- [ ] **Step 4: Run to verify failure**, then implement `scripts/12_bulk_judgments.py`:

```python
"""Bulk judgment downloader — AWS Open Data archives, anonymous, resumable.

Syncs configured prefixes of the public SC/HC judgment buckets into
data/raw/judgments/<source_id>/, enforcing a total-size cap across all
sources. Anonymous S3 (UNSIGNED) — zero cost, no AWS account. Safe to
interrupt: a manifest records completed keys; re-running resumes.

Usage:
    python scripts/12_bulk_judgments.py                # all enabled sources
    python scripts/12_bulk_judgments.py --dry-run      # plan only, no download
    python scripts/12_bulk_judgments.py --source sc_judgments
"""

import argparse
import json
import sys
import time
from pathlib import Path

import boto3
import yaml
from botocore import UNSIGNED
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bulk_sources.yaml"
OUT_DIR = ROOT / "data" / "raw" / "judgments"
MANIFEST = OUT_DIR / "manifest.json"


def select_keys(keys: list[dict], prefixes: list[str], max_bytes: int,
                already: set[str]) -> list[dict]:
    """Pick keys matching any prefix, stopping at the byte cap.

    Bytes of already-downloaded keys count toward the cap so a re-run never
    exceeds it; the keys themselves are not re-downloaded.
    """
    picked, used = [], 0
    for k in keys:
        if prefixes and not any(k["Key"].startswith(p) for p in prefixes):
            continue
        if used + k["Size"] > max_bytes:
            break
        used += k["Size"]
        if k["Key"] in already:
            continue
        picked.append(k)
    return picked


def list_bucket(client, bucket: str, prefixes: list[str]) -> list[dict]:
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for prefix in (prefixes or [""]):
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys.extend({"Key": o["Key"], "Size": o["Size"]}
                        for o in page.get("Contents", []))
    return keys


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def sync_source(source: dict, max_bytes_left: int, dry_run: bool) -> int:
    """Sync one source; returns bytes downloaded (or planned, if dry_run)."""
    client = boto3.client("s3", region_name=source["region"],
                          config=Config(signature_version=UNSIGNED))
    manifest = load_manifest()
    done = set(manifest.get(source["source_id"], []))
    keys = list_bucket(client, source["bucket"], source["prefixes"])
    plan = select_keys(keys, source["prefixes"], max_bytes_left, done)
    total = sum(k["Size"] for k in plan)
    print(f"[{source['source_id']}] {len(plan)} objects, "
          f"{total / 2**30:.2f} GiB planned ({len(done)} already done)")
    if dry_run:
        return total
    downloaded = 0
    for i, k in enumerate(plan):
        dest = OUT_DIR / source["source_id"] / k["Key"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(source["bucket"], k["Key"], str(dest))
        downloaded += k["Size"]
        done.add(k["Key"])
        if i % 50 == 0 or i == len(plan) - 1:  # checkpoint the manifest
            manifest[source["source_id"]] = sorted(done)
            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")
            print(f"  {i + 1}/{len(plan)} ({downloaded / 2**30:.2f} GiB)", flush=True)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="sync a single source_id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cap = int(config["max_total_gb"] * 2**30)
    already_bytes = sum(
        f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()
    ) if OUT_DIR.exists() else 0
    left = cap - already_bytes
    print(f"cap={config['max_total_gb']} GB, used={already_bytes / 2**30:.2f} GiB")
    for source in config["sources"]:
        if not source.get("enabled") or (args.source and source["source_id"] != args.source):
            continue
        if left <= 0:
            print(f"[{source['source_id']}] SKIPPED — cap reached")
            continue
        left -= sync_source(source, left, args.dry_run)


if __name__ == "__main__":
    main()
```

Add `boto3` to `requirements.txt` and `pip install boto3`.

- [ ] **Step 5: Dry-run against the real buckets** — `/c/Python314/python.exe scripts/12_bulk_judgments.py --dry-run`. Inspect the printed plan; explore the HC bucket's actual prefix layout (`aws-style key names in the listing`) and refine `prefixes:` in the config so the planned total stays well under the cap (metadata first; add specific bench/year PDF prefixes only if room remains).
- [ ] **Step 6: Run tests** — `/c/Python314/python.exe -m pytest tests/test_bulk_judgments.py -q` → 4 PASS.
- [ ] **Step 7: Commit code+config, then start the real sync in the background** (it runs for hours/days and is resumable — safe to interrupt):

```bash
git add configs/bulk_sources.yaml scripts/12_bulk_judgments.py tests/test_bulk_judgments.py requirements.txt
git commit -m "Add capped, resumable bulk judgment downloader (AWS Open Data, anonymous)"
/c/Python314/python.exe scripts/12_bulk_judgments.py   # background, long-running
```

---

### Task 8: Landmark-cases registry (asset 4 scaffold)

**Files:**
- Create: `configs/landmark_cases.yaml`
- Test: `tests/test_landmark_registry.py`

**Interfaces:**
- Consumes: nothing (pure data authoring + schema test).
- Produces: a curated registry consumed later (v2 plan) by a landmark-extraction script matching cases inside the downloaded HF corpora. Schema per entry: `case_id` (kebab-case), `name`, `citation`, `year` (int), `court` ("SC"), `topics` (list), `why_landmark` (one sentence).

- [ ] **Step 1: Failing schema test**

```python
# tests/test_landmark_registry.py
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "landmark_cases.yaml"

REQUIRED = {"case_id", "name", "citation", "year", "court", "topics", "why_landmark"}


def test_registry_schema_and_uniqueness():
    cases = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 50  # grows toward 200 over time
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case_id"
    for c in cases:
        assert REQUIRED <= set(c), f"{c.get('case_id')} missing {REQUIRED - set(c)}"
        assert isinstance(c["year"], int) and 1950 <= c["year"] <= 2026
```

- [ ] **Step 2: Author the registry** — write `configs/landmark_cases.yaml` with ≥50 verified landmark SC cases spanning the 13 priority acts' domains (constitutional rights, criminal procedure, consumer, IT/privacy, family, workplace). Seed set to include (verify citations before writing): Kesavananda Bharati (1973, basic structure), Maneka Gandhi (1978, Art 21), Vishaka (1997, workplace harassment), D.K. Basu (1997, arrest safeguards), Lalita Kumari (2014, mandatory FIR), Arnesh Kumar (2014, arrest under 498A), Shreya Singhal (2015, S.66A struck down), Puttaswamy (2017, privacy), Navtej Singh Johar (2018, S.377), Joseph Shine (2018, adultery), Common Cause (2018, passive euthanasia), Indra Sawhney (1992, reservations), S.R. Bommai (1994, federalism), Olga Tellis (1985, livelihood), M.C. Mehta line (environment), Hussainara Khatoon (1979, speedy trial), Nilabati Behera (1993, custodial death), Rupan Deol Bajaj (1995), Sarla Mudgal (1995), Githa Hariharan (1999), Selvi (2010, narcoanalysis), Aruna Shanbaug (2011), NALSA (2014, transgender rights), Shayara Bano (2017, triple talaq), Anuradha Bhasin (2020, internet shutdowns). Every entry's `citation` must be verified against a free, publicly accessible source (SC website / judgments portal / HF corpus metadata) — no memory-only citations.
- [ ] **Step 3: Run test** — `/c/Python314/python.exe -m pytest tests/test_landmark_registry.py -q` → PASS.
- [ ] **Step 4: Commit** — `git add configs/landmark_cases.yaml tests/test_landmark_registry.py && git commit -m "Add curated landmark SC case registry (asset 4 scaffold)"`.

---

## Human contribution checklist (give to the user after Task 2, 4, 5)

1. Spot-check samples in `reports/corpus_spotcheck_sample.json` for each newly enabled act (~5 min per act).
2. Manually fetch anything in `reports/manual_downloads.json` (drop files into `data/raw/assets/<group>/<asset_id>.<ext>`; re-run the fetcher to manifest them).
3. Write 100–200 real Hindi/Hinglish citizen legal questions (plain text file, one per line) — used later for training + eval authenticity.
4. `huggingface-cli login` + accept gated dataset terms (Task 6) — one-time, free.

## Execution order & independence

Tasks 1→2 sequential (same procedure), 3→4 sequential. Tasks 5, 6, 7, 8 are mutually independent and independent of 1–4 (parallelizable). Long-running downloads (Task 6 Step 6, Task 7 Step 7) run in the background while other tasks proceed.
