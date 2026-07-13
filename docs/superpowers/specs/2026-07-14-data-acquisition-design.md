# Nyaya Data Acquisition Plan — Design

**Date:** 2026-07-14
**Status:** Approved (staged approach, ~50–100 GB bulk cap)

## Goal

Acquire all legal data Nyaya needs — statutes, judgments, procedures, eval material — at
**zero cost** (no paid APIs, no API keys), from sources that are **legal to use**, staged so
v1 training is never blocked by bulk collection.

## Principles

1. **Zero cost, zero API keys.** Every source is a free government portal, an open S3
   archive (`--no-sign-request`), or a free Hugging Face dataset (free account login at most).
2. **Legal to use.** Indian acts and court judgments are exempt from copyright
   (s.52(1)(q), Copyright Act 1957). Commercial law books are copyrighted and are skipped —
   their substance exists in free form in judgments and Law Commission reports.
3. **"Expert lawyer knowledge" = judgments + Law Commission reports.** Supreme Court
   judgments contain the best legal reasoning in India, including summaries of senior
   advocates' arguments. This replaces books we cannot copy.
4. **Staged.** Quality data first (unblocks v1 training), bulk collection runs slowly in
   the background for v2. Bulk case-law is a v2 feature per the roadmap.
5. **No ToS violations.** We do not scrape Indian Kanoon (ToS forbids it; they block
   scrapers; their content comes from the official sources we use directly).

## Stage A — Finish the statute layer (v1, <1 GB)

4 of 13 priority acts are done (BNS, BNSS, BSA, RTI). Acquire the rest from official portals:

| Source | What |
|---|---|
| legislative.gov.in | Constitution of India — latest official edition, English + Hindi |
| indiacode.nic.in | Consumer Protection Act 2019, DV Act 2005, HMA 1955, SMA 1954, NI Act 1881, MV Act 1988, IT Act 2000, POSH 2013, Code on Wages 2019 / Payment of Wages Act |

- Prefer HTML for Hindi text (avoids Devanagari mojibake per roadmap).
- All acts flow through the existing `03_build_corpus.py` pipeline and the ≥98%-clean
  spot-check gate. Sources are declared in `configs/statute_sources.yaml`.
- If a portal blocks scripted download (captcha/broken link), the script emits a
  "manual download needed" list for the user instead of failing.

## Stage B — Free Hugging Face datasets (v1 + eval, ~10–30 GB)

`scripts/00_download_hf_datasets.py` + `configs/hf_datasets.yaml` already exist.
Work: **verify exact hub IDs** (config flags several as unverified), enable, download.

- Already enabled: `opennyaiorg/aalap_instruction_dataset`, `Exploration-Lab/IL-TUR`,
  `viber1/indian-law-dataset`.
- Verify then enable: NyayaAnumana (download **filtered Supreme Court slice only**, not
  all ~700K cases), MILDSum, ILDC (may need signed agreement — skip if so), BhashaBench-Legal.
- Extract the roadmap's **~200 landmark SC judgments** (asset 4) from these corpora —
  no scraping.
- Gated datasets need only a free `huggingface-cli login` + accepting terms on the site.

## Stage C — Procedure & citizen-knowledge sources (v1, ongoing)

Free official sources that ground the 60–80 hand-written procedure docs (asset 3):

- NALSA / DLSA publications — legal aid procedures
- rtionline.gov.in FAQs — RTI filing
- consumerhelpline.gov.in — consumer complaint procedure
- cybercrime.gov.in — cybercrime reporting flow
- parivahan.gov.in — challans, licenses, vehicle matters
- State police portals — FIR / e-FIR guides
- Law Commission of India reports — free PDFs, high-quality legal reasoning prose
- AIBE past papers — free, feed the eval harness

Claude drafts each procedure doc grounded in the collected sources + statute DB;
**the user verifies** against act text (human gate, matches roadmap).

## Stage D — Background bulk judgment collection (v2, 50–100 GB cap)

- **AWS Open Data Registry archives** (free, no account, `aws s3 sync --no-sign-request`):
  - Indian **Supreme Court** judgments archive — complete (small enough to take whole).
  - Indian **High Court** judgments archive (~16M docs) — **filtered slices only**:
    selected courts + recent years, metadata-first, staying under the 100 GB cap.
- Fallback: judgments.ecourts.gov.in (official portal).
- New `configs/bulk_sources.yaml` + resumable downloader `scripts/12_bulk_judgments.py`:
  rate-limited, manifest + checksums, safe to interrupt and resume, runs overnight.
- Explicitly **not** done: Indian Kanoon scraping, copyrighted book piracy, mass
  district-court scraping.

## Human contributions (the only things scripts can't do)

1. **Verify procedure docs** (Stage C) — read each drafted doc against the act text.
2. **5% spot-checks** of newly extracted statutes (existing gate).
3. **100–200 real Hindi/Hinglish citizen questions** written by the user — authentic
   phrasing that synthetic generation can't fake; used for training + eval.
4. **Manual downloads** only when a portal blocks scripts (script emits the list).
5. **Blind human eval** later (2 raters, per roadmap ship gate).

## Deliverables

- v1 complete data layer: 13 acts + Constitution (EN+HI), mappings, procedure sources,
  ~200 landmark judgments, eval benchmarks — ₹0 spent.
- v2 archive: Supreme Court + filtered High Court case law, ≤100 GB, resumable.

## Error handling

- Every downloader: retry with backoff, resume from manifest, per-file checksums,
  polite rate limits (1–2 req/s max on government portals).
- Any source that fails scripted acquisition degrades to a manual-download task list,
  never a pipeline crash.

## Testing

- Statute extractions: existing spot-check + citation-verification gates.
- HF downloads: row counts + schema checks against dataset cards.
- Bulk sync: manifest completeness check; re-run is idempotent.
