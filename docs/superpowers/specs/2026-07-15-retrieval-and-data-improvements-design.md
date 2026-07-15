# Retrieval & Data Improvements (CPU-side, parallel to v2/v3 training)

**Date:** 2026-07-15
**Status:** Approved
**Context:** v2 results (commit f869326) show the fine-tuned adapter has recovered
citing behaviour under RAG and now ties the base model on accuracy. The binding
constraints are no longer the model: (1) retrieval reached the gold section on only
85/132 citation-gold frozen-eval questions (full_hit@8 = 64.4%) and accuracy is 0%
whenever it misses; (2) 363/495 questions carry phrase-only gold (helplines, portals,
procedures) that exists in no retrievable document; (3) extraction at gold-in-context
tops out ~28%. All three are attackable on a CPU-only machine while training runs
elsewhere. Zero budget: free sources and free models only.

## Goals

- Raise frozen-eval retrieval full_hit@8 from 64.4% to ≥75% (dependency-free), with
  an optional dense stage measured for more.
- Make the 363 phrase-only-gold questions structurally answerable by giving the
  retriever a procedural knowledge base to draw from.
- Give the next training run (v3) data that teaches verbatim fact extraction from
  retrieved context.
- Everything measurable locally via `scripts/15_retrieval_recall.py` (no GPU);
  every work item lands as its own commit on `main` so the teammate can pull pieces
  as they land.

## Non-goals

- No changes to the frozen eval set (`data/eval/nyaya_eval_v0.jsonl`) — it stays frozen.
- No changes to training/eval k8s jobs or images (the dense stage ships disabled).
- No case-law corpus (later phase).

## Work item A — Retrieval recall push (`src/nyaya/retrieval.py`)

### A1. Domain-aware ambiguous-citation resolution

Today `StatuteIndex.referenced_keys()` drops a bare section reference ("Section 139
presumption of debt") unless exactly one act contains that section number, which
loses 21 gold facts and hurts real queries. Fix, in order of precedence:

1. If the caller supplies a `domain` hint (eval records carry `legal_domain`), map it
   to candidate acts (cheque_bounce → ni_act_1881, rti → rti_2005, cyber_law →
   it_act_2000, motor_vehicles → mv_act_1988, womens_protection → dv_act_2005 +
   posh_2013 + bns_2023, consumer_law → cpa_2019, labour_law → wages_code_2019,
   bns/bnss/bsa/constitutional_law → their acts) and restrict candidates to those acts.
2. Otherwise disambiguate by content: BM25-score the fact's remaining words against
   the candidate sections themselves; accept the top candidate only if its score is
   non-zero, else keep current drop behaviour.

`referenced_keys` gains an optional `domain: str | None = None` parameter; existing
call sites keep working. `scripts/15` passes the record's `legal_domain` when
resolving gold (gold resolution may use it — the *retrieval* call for the question
does not get the hint, since real users don't supply domains).

### A2. Synonym table expansion

Add clusters for the measured weak domains, calibrated only against train-split
questions (never the frozen eval):

- **Evidence/BSA** (recall 25%): electronic record, certificate, admissibility,
  witness examination, confession, dying declaration, burden of proof, presumption.
- **Consumer:** complaint filing, deficiency, refund/replacement, e-daakhil,
  district/state/national commission, unfair trade practice.
- **RTI:** application, PIO, first appeal, information commission, exemption, fee.
- **Labour:** wages, overtime, bonus, termination, gratuity.
- **Motor vehicles:** compensation, third-party insurance, hit and run, licence,
  challan, Good Samaritan.
- **Cheque bounce:** notice, presumption, drawer, legally enforceable debt.
- Additional Hindi/Hinglish equivalents for all of the above.

### A3. BM25 field improvements

- Index the `tags` field (schema documents that tags exist "to power retrieval";
  the tokenizer currently ignores them) and `punishment_summary`.
- Tags get a field bonus like titles do (weight chosen on train split).

### A4. Optional dense hybrid stage (built last, ships disabled)

- `DenseStage` using `sentence-transformers` with a free multilingual model
  (candidates: `BAAI/bge-m3`, `intfloat/multilingual-e5-small`; pick by local
  recall measurement and download size).
- Fusion: reciprocal-rank fusion of BM25 ranking and cosine ranking; exact-reference
  stage 1 is unchanged and always wins.
- Strictly opt-in: `load_statute_index(..., dense=False)` default and a `--dense`
  flag on `scripts/15` / `scripts/16`. `sentence-transformers` goes in an optional
  requirements extra, not the base requirements, so the k8s image is untouched.
- Embeddings for the statute DB are computed once on CPU and cached to disk
  (`data/canonical/.dense_cache/`, gitignored).

**Acceptance (A):** `scripts/15` reports full_hit@8 ≥ 0.75 without dense;
unresolved_fact_count drops from 21 to ≤ 5; all existing tests pass plus new tests
for domain-aware resolution and tags indexing. Dense stage reported separately with
its measured delta.

## Work item B — Procedural knowledge base (`data/canonical/procedures_kb.jsonl`)

The data README already promises a "procedure KB" in canonical; it was never built.

- **Schema:** reuse `StatuteSection` rows: `act_id="procedures_kb"`,
  `act_name="Official Procedural Guidance (India)"`, `section=<slug>` (e.g.
  `cyber-fraud-reporting`), `title`, `text` (the guidance, plain language),
  `tags` (lay + Hindi terms), `source_url` (official source). `load_statute_index`
  ingests it with zero code change.
- **Coverage (~60–100 snippets), driven by the measured failure topics** in
  `reports/error_analysis.json` top_missing_facts and the phrase-only gold set:
  1930 cyber helpline & cybercrime.gov.in; Zero FIR & e-FIR; FIR copy rights & refusal
  remedy (BNSS 173); arrest rights & grounds (BNSS 47/48); bail timelines,
  first-time offender release (BNSS 479), plea bargaining; women's helplines (181,
  112), DV complaint route, POSH complaint route; consumer complaint via e-daakhil,
  pecuniary jurisdiction tiers; RTI application steps, fees, timelines, appeals;
  cheque-bounce demand-notice timeline (30/15 days) and where to complain;
  MV accident: FIR, DAR, Motor Accident Claims Tribunal, Good Samaritan protection,
  hit-and-run compensation scheme; e-challan payment/contest.
- **Sourcing:** official portals only (cybercrime.gov.in, NALSA, india.gov.in,
  consumerhelpline.gov.in, rti.gov.in, morth.nic.in), fetched free; every row carries
  `source_url`. Where a snippet paraphrases a statute, the section is named in the text.
- **Prompt fit:** `format_context()` currently renders every row as "Section X of the
  {act_name}". Guidance rows render instead as "{title} — official guidance\n{text}",
  and the RAG prompt wording changes from "cite only sections above" to also permit
  naming official guidance sources. `build_rag_training_record` is unchanged.
- **Eval integrity:** snippets are written from official sources, not from eval
  answers; overlap with eval phrasing is inherent (both describe the same official
  facts) and acceptable — the eval measures whether the system can surface exactly
  these facts.

**Acceptance (B):** procedures_kb.jsonl validates against the schema (new test);
`scripts/15` re-run shows no regression on statute recall; a new lightweight check
reports what fraction of the 363 phrase-only questions now have at least one gold
phrase present in top-k retrieved text (baseline ~0).

## Work item C — Extraction training data (`scripts/19_generate_extraction_data.py`)

Rule-based generator (no LLM, zero cost) over `data/canonical/`:

- **Templates:** punishment extraction ("What is the punishment under Section {n} of
  the {act}?"), duration/deadline extraction ("Within how many days ..."), fine
  amounts, definition lookups ("What does {title} mean under {act}?"), old→new
  mapping ("Which BNS section replaced IPC {n}?" from law_mappings). Answers are
  composed from the section's own text — the exact numbers/durations verbatim —
  in the trained answer style (cite "Section {n} of the {act_name}").
- **Fact source:** regex extraction of durations/fines/dates from section text and
  the `punishment_summary` field; sections where extraction is ambiguous are skipped
  (precision over volume).
- **Language:** English first; Hindi/Hinglish variants for a template subset using
  the same fixed phrasings as the synonym table.
- **Volume:** target 1,500–3,000 records, capped per act for balance.
- **Schema/pipeline fit:** emits `TrainingRecord` JSONL into `data/generated/`
  with `metadata.source_sections`, `metadata.generator="rule_extraction_v1"`,
  `task_type="extraction_qa"` — flows through existing validate → dedup → split →
  RAG-format scripts unchanged.
- **Eval hygiene:** sections that any frozen-eval record's gold resolves to are
  EXCLUDED from generation (resolution done with the A1 parser, domain-hinted).
  This over-excludes rather than under-excludes.

**Acceptance (C):** generated file passes `scripts/05_validate_examples.py`; zero
records sourced from eval-gold sections (asserted in a test); spot-check of 20
random records reads correct.

## Measurement & handoff

- Before/after `reports/retrieval_recall.json` committed with each retrieval change;
  the old numbers stay in git history.
- New/extended tests in `tests/test_retrieval.py` (+ a KB schema test).
- Commit sequence: A1 → A2+A3 → B → C → A4(optional), each independently pullable.
- Teammate's follow-up (documented in commit messages): pull, rebuild RAG dataset
  (`scripts/18`) with improved retriever + KB, regenerate splits including
  extraction data, launch v3, re-run `scripts/16` RAG eval on the frozen set.

## Risks

- **Synonym/BM25 tuning overfits the frozen eval.** Mitigation: calibrate only on
  train-split questions; the frozen eval is measured, never optimized against
  (same discipline the codebase already documents).
- **KB snippets drift from official truth.** Mitigation: source_url on every row;
  paraphrase conservatively; statutory claims name their section.
- **Extraction templates produce stilted text that degrades style.** Mitigation:
  cap volume relative to the existing ~6.3k examples; teammate can down-weight.
- **Dense model too slow on cluster CPUs at eval time.** Mitigation: disabled by
  default; measured locally first; cache embeddings.
