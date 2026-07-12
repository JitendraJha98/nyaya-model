"""Step 3 — Build the raw legal corpus (the single source of truth).

Assets, in order (see docs/ROADMAP.md, "Core data assets"):
  1. Statute DB — 13 priority acts from indiacode.nic.in / legislative.gov.in,
     one JSONL row per section (schema: src/nyaya/schemas.py::StatuteSection).
     Pipeline: PDF -> pymupdf text extraction -> regex section splitter ->
     manual spot-check 5% -> JSONL. Watch for Devanagari mojibake; prefer HTML for Hindi.
  2. IPC<->BNS / CrPC<->BNSS mapping tables — from official MHA comparison tables (~1,100 rows).
  3. Procedure KB — 60–80 hand-written "how do I..." docs verified against act text.
  4. Landmark judgments (~200 SC cases) — from HF datasets, not scraping.

Outputs -> data/canonical/*.jsonl

Go/no-go gate: statute extraction spot-check must be >=98% clean before any
synthetic generation.

TODO: implement extractors per act.
"""

raise NotImplementedError("See docs/ROADMAP.md 'Core data assets' — build Statute DB first.")
