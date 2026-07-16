# Retrieval & Data Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise frozen-eval retrieval recall from 64.4% to ≥75% full_hit@8, make the 363 phrase-only-gold eval questions retrievable via a procedural knowledge base, and generate extraction-focused training data for the v3 run — all CPU-only, all measurable locally.

**Architecture:** All retrieval work lands in `src/nyaya/retrieval.py` (pure-Python BM25 index, no dependencies). The procedures KB is a new `data/canonical/procedures_kb.jsonl` reusing the `StatuteSection` row shape so the existing loader ingests it unchanged. The extraction generator is a new standalone script emitting `TrainingRecord` JSONL into the existing validate→dedup→split pipeline. The dense stage is a separate lazily-imported module so the base package stays dependency-free.

**Tech Stack:** Python 3.14 (`C:/Python314/python.exe`), pytest, stdlib only (Task 7 optionally adds `sentence-transformers` behind an extras file).

## Global Constraints

- Zero budget: free sources and free models only. No paid APIs, no keys.
- `data/eval/nyaya_eval_v0.jsonl` is FROZEN. Never edit it, never train on it, never tune against it. Retrieval changes may be *measured* on it once per task via `scripts/15`, but synonym/weight choices are justified from statute vocabulary (`data/canonical/` texts), not by iterating on eval numbers.
- The user-facing retrieval path (`StatuteIndex.retrieve(question)`) never receives a `domain` hint — only gold-fact *resolution* does (real users don't announce their legal domain).
- Windows console: set `PYTHONIOENCODING=utf-8` when running scripts that print Devanagari (`$env:PYTHONIOENCODING = "utf-8"` in PowerShell).
- Python is at `C:/Python314/python.exe`; run tests as `C:/Python314/python.exe -m pytest`.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Do not modify the k8s manifests or `requirements.txt` (Task 7 uses a separate `requirements-dense.txt`).

---

### Task 1: Domain-aware ambiguous-citation resolution (A1)

Today `StatuteIndex.referenced_keys()` drops a bare section reference ("Section 139 presumption of debt") whenever more than one act contains that number. This loses 21 gold facts in `reports/retrieval_recall.json` (`unresolved_facts_sample`). Fix with two tiebreakers: an optional caller-supplied domain hint, then content overlap between the fact's words and the candidate sections.

**Files:**
- Modify: `src/nyaya/retrieval.py` (around lines 42 and 207–243)
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Produces: `StatuteIndex.referenced_keys(query: str, domain: str | None = None) -> list[str]` (signature extended, default keeps old behaviour); module-level dict `DOMAIN_ACTS: dict[str, tuple[str, ...]]`; private helper `StatuteIndex._pick_by_content(candidates: list[str], query: str) -> list[str]`.
- Consumed by: Task 2 (scripts pass `domain=record["legal_domain"]`), Task 6 (eval-section exclusion).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retrieval.py`:

```python
AMBIG_ROWS = ROWS + [
    {"act_id": "ni_act_1881", "act_name": "Negotiable Instruments Act, 1881",
     "section": "139", "title": "Presumption in favour of holder",
     "text": "It shall be presumed, unless the contrary is proved, that the holder "
     "of a cheque received the cheque for the discharge, in whole or in part, of "
     "any debt or other liability.", "chapter": "XVII"},
    {"act_id": "bnss_2023", "act_name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
     "section": "139", "title": "Power to declare certain publications forfeited",
     "text": "Where any newspaper or book contains any matter the publication of "
     "which is punishable, the State Government may declare it forfeited.",
     "chapter": "XI"},
]


class TestDomainAwareResolution:
    @pytest.fixture(scope="class")
    def ambig_index(self):
        return StatuteIndex(AMBIG_ROWS, MAPPINGS)

    def test_bare_ambiguous_section_still_drops_without_hints(self, ambig_index):
        # two acts contain a section 139 and nothing disambiguates -> no keys
        assert ambig_index.referenced_keys("Section 139") == []

    def test_domain_hint_resolves_bare_section(self, ambig_index):
        keys = ambig_index.referenced_keys("Section 139", domain="cheque_bounce")
        assert keys == ["ni_act_1881:139"]

    def test_content_words_resolve_bare_section(self, ambig_index):
        keys = ambig_index.referenced_keys("Section 139 presumption of debt")
        assert keys == ["ni_act_1881:139"]

    def test_named_act_beats_domain_hint(self, ambig_index):
        # an explicit act name is stronger evidence than the domain
        keys = ambig_index.referenced_keys("Section 139 of the BNSS",
                                           domain="cheque_bounce")
        assert keys == ["bnss_2023:139"]

    def test_unknown_domain_is_harmless(self, ambig_index):
        assert ambig_index.referenced_keys("Section 139", domain="nonsense") == []

    def test_content_tie_still_drops(self, ambig_index):
        # words matching neither candidate: content stage must not guess
        assert ambig_index.referenced_keys("Section 139 zebra flying") == []

    def test_unambiguous_section_unaffected(self, ambig_index):
        # only one act has a 318 -> resolves exactly as before, no hint needed
        assert ambig_index.referenced_keys("Section 318") == ["bns_2023:318"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Python314/python.exe -m pytest tests/test_retrieval.py::TestDomainAwareResolution -v`
Expected: FAIL — `TypeError: referenced_keys() got an unexpected keyword argument 'domain'` (and assertion failures for the content test).

- [ ] **Step 3: Implement**

In `src/nyaya/retrieval.py`, add after `_OLD_TO_NEW_ACT` (line 42):

```python
# eval-record legal_domain -> the act(s) a bare "Section <n>" most plausibly
# names in that domain. A resolution HINT only: it is consulted when no act is
# named in the text and the section number is ambiguous across acts. The
# user-facing retrieval path never passes a domain.
DOMAIN_ACTS = {
    "cheque_bounce": ("ni_act_1881",),
    "rti": ("rti_2005",),
    "cyber_law": ("it_act_2000",),
    "motor_vehicles": ("mv_act_1988",),
    "consumer_law": ("cpa_2019",),
    "labour_law": ("wages_code_2019",),
    "womens_protection": ("dv_act_2005", "posh_2013", "bns_2023"),
    "bns": ("bns_2023",),
    "bnss": ("bnss_2023",),
    "bsa": ("bsa_2023",),
    "constitutional_law": ("constitution_1950",),
}
```

Replace the `referenced_keys` method (keep its docstring, extend it):

```python
    def referenced_keys(self, query: str, domain: str | None = None) -> list[str]:
        """Resolve explicit statute references in `query` to index keys.
        Also used to parse gold citations out of eval required_facts.

        `domain` (an eval-record legal_domain) and the query's own content
        words act as tiebreakers for bare section numbers that exist in more
        than one act; with neither, ambiguous references drop as before.
        """
        query_lower = query.lower()
        families = []
        for family, variants in ACT_ALIASES.items():
            if any(re.search(alias_pattern(v), query_lower) for v in variants):
                families.append(family)
        keys = []
        for m in CITATION_PATTERN.finditer(query):
            number_match = _NUMBER.search(m.group(0))
            if not number_match:
                continue
            section = number_match.group(1).upper()
            marker = m.group(0).lower()
            # "Article 21" / "अनुच्छेद 21" names the Constitution by itself
            article_like = "art" in marker or "अनुच्छेद" in marker
            match_families = families or (
                ["constitution"] if article_like else list(_FAMILY_TO_ACT_ID))
            for family in match_families:
                mapped_family = _OLD_TO_NEW_ACT.get(family)
                if mapped_family:  # old-law reference -> official mapping
                    keys.extend(self.old_to_new.get((family, section), []))
                act_id = _FAMILY_TO_ACT_ID.get(family)
                if act_id and f"{act_id}:{section}" in self.by_key:
                    keys.append(f"{act_id}:{section}")
            # no act named: disambiguate by domain hint, then content overlap
            if not families and not article_like:
                candidates = [k for k in keys if k.endswith(f":{section}")]
                if len(candidates) > 1 and domain in DOMAIN_ACTS:
                    narrowed = [k for k in candidates
                                if k.split(":", 1)[0] in DOMAIN_ACTS[domain]]
                    if narrowed:
                        candidates = narrowed
                if len(candidates) > 1:
                    candidates = self._pick_by_content(candidates, query)
                if len(candidates) != 1:
                    keys = [k for k in keys if not k.endswith(f":{section}")]
                else:
                    keep = candidates[0]
                    keys = [k for k in keys
                            if k == keep or not k.endswith(f":{section}")]
        seen, ordered = set(), []
        for k in keys:
            if k not in seen:
                seen.add(k)
                ordered.append(k)
        return ordered

    def _pick_by_content(self, candidates: list[str], query: str) -> list[str]:
        """Among candidate keys sharing a section number, keep the single row
        the query's remaining words actually describe — the unique
        idf-weighted-overlap winner. Returns `candidates` unchanged when
        content cannot decide (all-zero scores or a tie for first)."""
        q_tokens = set(_tokens(expand_query(query)))
        scored = sorted(
            ((sum(self.idf.get(w, 0.0) for w in q_tokens
                  if w in self.tf[self.by_key[key]]), key)
             for key in candidates),
            reverse=True)
        if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            return [scored[0][1]]
        return candidates
```

- [ ] **Step 4: Run the full test file**

Run: `C:/Python314/python.exe -m pytest tests/test_retrieval.py -v`
Expected: ALL PASS (new class + every pre-existing test — the default-argument path must be behaviour-identical).

- [ ] **Step 5: Commit**

```bash
git add src/nyaya/retrieval.py tests/test_retrieval.py
git commit -m "Retrieval: domain + content tiebreakers for bare section references

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Pass the domain hint at gold-resolution call sites, measure

**Files:**
- Modify: `scripts/15_retrieval_recall.py:40-50` (gold_keys), `scripts/16_rag_eval.py:116`
- Output: `reports/retrieval_recall.json` (regenerated)

**Interfaces:**
- Consumes: `referenced_keys(fact, domain=...)` from Task 1.
- Produces: updated `reports/retrieval_recall.json` with `unresolved_fact_count ≤ 5` and a larger `gold_bearing` set — the baseline all later tasks are measured against.

- [ ] **Step 1: Update the two call sites**

In `scripts/15_retrieval_recall.py`, `gold_keys` (line 45):

```python
        keys = index.referenced_keys(fact, domain=record.get("legal_domain"))
```

In `scripts/16_rag_eval.py` (line 116):

```python
                gold.update(index.referenced_keys(fact, domain=rec.get("legal_domain")))
```

(Only gold parsing gets the hint. The retrieval calls for the *question* — `index.retrieve(rec["question"], ...)` in script 15 and the generate fn in script 16 — stay hint-free per Global Constraints.)

- [ ] **Step 2: Regenerate the recall report**

Run: `C:/Python314/python.exe scripts/15_retrieval_recall.py`
Expected: `unresolved_fact_count` drops from 21 toward ≤5; `gold_bearing` rises above 132. full_hit@8 may move either way slightly (newly resolvable gold questions are also newly *scoreable*, and BM25 may miss some) — record whatever it says; this task's deliverable is honest gold resolution, not the recall number.

- [ ] **Step 3: Sanity-check previously-unresolved facts**

Run:
```bash
C:/Python314/python.exe -c "
import sys; sys.path.insert(0, 'src')
from nyaya.retrieval import load_statute_index
idx = load_statute_index('data/canonical')
print(idx.referenced_keys('Section 139 presumption of debt', domain='cheque_bounce'))
print(idx.referenced_keys('Section 8 exemptions', domain='rti'))
print(idx.referenced_keys('Section 43A compensation', domain='cyber_law'))
print(idx.referenced_keys('धारा 138', domain='cheque_bounce'))
"
```
Expected: `['ni_act_1881:139']`, `['rti_2005:8']`, `['it_act_2000:43A']`, `['ni_act_1881:138']`. If any prints `[]`, debug before committing (check the act file actually contains that section; `43A` requires the `\d+[A-Za-z]{0,2}` capture, which it satisfies).

- [ ] **Step 4: Commit**

```bash
git add scripts/15_retrieval_recall.py scripts/16_rag_eval.py reports/retrieval_recall.json
git commit -m "Gold-fact resolution uses eval legal_domain; unresolved facts 21 -> <n>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
(Replace `<n>` with the measured number.)

---

### Task 3: Synonym clusters for weak domains + tags/punishment_summary field support (A2+A3)

BSA recall is 25% and consumer/labour/MV domains are thin because lay vocabulary ("WhatsApp chat as evidence", "salary not paid") shares no tokens with statutory phrasing. Every expansion below maps a lay term to phrasing that appears verbatim in the relevant canonical section texts — justify additions by grepping `data/canonical/`, never by eval scores. Also index the `tags` and `punishment_summary` fields: they are empty today (verified 2026-07-15: 0 of 2,458 rows populated) so this changes nothing now, but the procedures KB (Task 5) ships tagged rows and the schema documents tags as retrieval-powering.

**Files:**
- Modify: `src/nyaya/retrieval.py` (LEGAL_SYNONYMS dict, `StatuteIndex.__init__` lines 192–198)
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Produces: extended `LEGAL_SYNONYMS`; `StatuteIndex` indexes `tags` (with title-level bonus) and `punishment_summary` (body-level). No signature changes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retrieval.py`:

```python
class TestNewSynonymClusters:
    def _idx(self, *extra_rows):
        return StatuteIndex(ROWS + list(extra_rows), MAPPINGS)

    def test_electronic_evidence_reaches_bsa(self):
        idx = self._idx(
            {"act_id": "bsa_2023", "act_name": "Bharatiya Sakshya Adhiniyam, 2023",
             "section": "63", "title": "Admissibility of electronic records",
             "text": "Any information contained in an electronic record shall be "
             "admissible if accompanied by a certificate identifying the electronic "
             "record and describing the manner of its production.", "chapter": "V"})
        hits = idx.retrieve("Can I use a WhatsApp chat as proof in court?", k=2)
        assert any(h["section"] == "63" for h in hits)

    def test_unpaid_salary_reaches_wages_code(self):
        idx = self._idx(
            {"act_id": "wages_code_2019", "act_name": "Code on Wages, 2019",
             "section": "43", "title": "Responsibility for payment of various dues",
             "text": "Every employer shall pay all amounts of wages required to be "
             "paid under this Code to every employee employed by him.", "chapter": "V"})
        hits = idx.retrieve("My company has not paid my salary for two months", k=2)
        assert any(h["act_id"] == "wages_code_2019" for h in hits)

    def test_accident_compensation_reaches_mv_act(self):
        idx = self._idx(
            {"act_id": "mv_act_1988", "act_name": "Motor Vehicles Act, 1988",
             "section": "166", "title": "Application for compensation",
             "text": "An application for compensation arising out of an accident "
             "may be made to the Claims Tribunal by the person who has sustained "
             "the injury.", "chapter": "XII"})
        hits = idx.retrieve("Road accident me injury hui, muavza kaise milega?", k=2)
        assert any(h["section"] == "166" for h in hits)

    def test_otp_fraud_reaches_it_act(self):
        idx = self._idx(
            {"act_id": "it_act_2000", "act_name": "Information Technology Act, 2000",
             "section": "66D", "title": "Punishment for cheating by personation "
             "by using computer resource",
             "text": "Whoever, by means of any communication device or computer "
             "resource cheats by personation, shall be punished with imprisonment.",
             "chapter": "XI"})
        hits = idx.retrieve("Someone did OTP fraud on my phone and took money", k=2)
        assert any(h["section"] == "66D" for h in hits)


class TestFieldIndexing:
    def test_tags_are_indexed_with_title_weight(self):
        tagged = {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023",
                  "section": "999", "title": "Some provision",
                  "text": "generic words only here.", "chapter": "X",
                  "tags": ["gharelu hinsa"]}
        idx = StatuteIndex(ROWS + [tagged], MAPPINGS)
        hits = idx.retrieve("gharelu hinsa complaint", k=2)
        assert any(h["section"] == "999" for h in hits)

    def test_punishment_summary_is_indexed(self):
        row = {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023",
               "section": "998", "title": "Another provision",
               "text": "generic words only here.", "chapter": "X",
               "punishment_summary": "community service for petty theft"}
        idx = StatuteIndex(ROWS + [row], MAPPINGS)
        hits = idx.retrieve("community service petty punishment", k=2)
        assert any(h["section"] == "998" for h in hits)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Python314/python.exe -m pytest tests/test_retrieval.py::TestNewSynonymClusters tests/test_retrieval.py::TestFieldIndexing -v`
Expected: FAIL (retrieval misses the target rows; tags/punishment_summary are not tokenized).

- [ ] **Step 3: Implement field indexing**

In `StatuteIndex.__init__`, replace the `doc_tokens`/`title_tokens` construction:

```python
        self.doc_tokens = [
            _tokens(f"{r['act_name']} {r.get('title') or ''} {r.get('text') or ''} "
                    f"{r.get('punishment_summary') or ''} "
                    f"{' '.join(r.get('tags') or [])}")
            for r in rows
        ]
        # tags are curated retrieval keywords — they earn the same field bonus
        # as title words
        self.title_tokens = [
            set(_tokens(r.get("title") or ""))
            | set(_tokens(" ".join(r.get("tags") or [])))
            for r in rows
        ]
```

- [ ] **Step 4: Implement the synonym clusters**

Append to `LEGAL_SYNONYMS` (before the closing brace, after the existing Hindi block; every expansion phrase is statutory vocabulary present in the canonical texts):

```python
    # evidence / BSA
    "whatsapp": "electronic record admissible certificate",
    "screenshot": "electronic record admissible certificate",
    "cctv": "electronic record video admissible certificate",
    "call recording": "electronic record admissible certificate",
    "electronic evidence": "electronic record certificate admissible",
    "digital evidence": "electronic record certificate admissible",
    "proof in court": "evidence admissible electronic record document",
    "dying declaration": "statement written verbal person who is dead cause of his death",
    "burden of proof": "burden of proving fact lies on that person",
    "confession": "confession accused inducement threat coercion",
    "cross examination": "examination of witness cross-examination",
    "hostile witness": "witness examination party question",
    # consumer
    "defective": "defect deficiency goods service consumer",
    "warranty": "defect deficiency goods guarantee",
    "online shopping": "e-commerce unfair trade practice consumer deficiency",
    "wrong product": "deficiency defect goods unfair trade practice consumer",
    "overcharged": "unfair trade practice consumer price",
    "misleading advertisement": "misleading advertisement consumer",
    "product exchange": "replace goods defect deficiency consumer",
    # rti
    "pio": "public information officer request information",
    "information from government": "right to information public authority request",
    "rti fee": "request information fee prescribed",
    "rti appeal": "appeal central information commission state information commission",
    # labour / wages
    "salary": "wages payment employer employee",
    "salary not paid": "wages payment employer employee dues",
    "unpaid wages": "wages payment employer employee dues",
    "fired": "removal dismissal employer employee",
    "overtime": "wages overtime work hours",
    "minimum wage": "minimum rate of wages",
    "bonus": "bonus wages employee",
    # motor vehicles
    "accident": "accident compensation claims tribunal injury",
    "insurance claim": "insurance insurer compensation third party",
    "challan": "penalty fine offence licence",
    "driving licence": "licence driving motor vehicle",
    "drunk driving": "driving by a drunken person under the influence of drink",
    "helmet": "protective headgear",
    "good samaritan": "good samaritan accident victim emergency medical",
    # cheque / NI Act
    "stop payment": "cheque returned unpaid dishonour",
    "cheque notice": "notice in writing demand payment drawer cheque",
    # criminal procedure
    "police refuse fir": "information cognizable offence superintendent of police magistrate",
    "zero fir": "information cognizable offence irrespective of the area where the offence is committed",
    "police station": "officer in charge of a police station",
    # women
    "stalking": "follows a woman contacts attempts to contact",
    "domestic violence": "aggrieved person protection order residence shared household",
    "harassment at work": "sexual harassment workplace internal committee complaint",
    # cyber
    "hacking": "computer resource unauthorised access damage",
    "identity theft": "fraudulently dishonestly electronic signature password unique identification",
    "phishing": "cheats by personation computer resource communication device",
    "otp fraud": "cheats by personation computer resource cheating dishonestly password",
    "online fraud": "cheats by personation computer resource cheating dishonestly induces",
    "morphed photo": "publishes transmits obscene material electronic form privacy",
    # Hindi / Hinglish additions
    "तनख्वाह": "wages payment employer",
    "tankhwah": "wages payment employer",
    "वेतन": "wages payment employer",
    "नौकरी": "employment employer employee removal",
    "naukri se nikala": "removal dismissal employer employee",
    "एक्सीडेंट": "accident compensation claims tribunal",
    "दुर्घटना": "accident compensation claims tribunal",
    "durghatna": "accident compensation claims tribunal",
    "बीमा": "insurance insurer compensation",
    "beema": "insurance insurer compensation",
    "मुआवज़ा": "compensation",
    "muavza": "compensation",
    "चेक बाउंस": "cheque returned unpaid dishonour insufficiency of funds",
    "check bounce": "cheque returned unpaid dishonour insufficiency of funds",
    "सबूत": "evidence proof admissible document",
    "saboot": "evidence proof admissible document",
    "उपभोक्ता": "consumer complaint deficiency",
    "ठगी": "cheating dishonestly induces deliver property",
    "thagi": "cheating dishonestly induces deliver property",
    "ऑनलाइन फ्रॉड": "cheats by personation computer resource cheating",
```

CAUTION — duplicate keys: the existing table already has `"muavza"` and `"मुआवजा"` (compensation); the additions above deliberately omit those, and `"मुआवज़ा"` (with nukta) is a NEW distinct key. A duplicate key in a dict literal silently overrides instead of erroring, so before committing verify every added key is absent from the existing table (grep the file for each exact quoted key), and run this whole-dict check:

```bash
C:/Python314/python.exe -c "
import ast, sys
tree = ast.parse(open('src/nyaya/retrieval.py', encoding='utf-8').read())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'LEGAL_SYNONYMS':
        keys = [k.value for k in node.value.keys]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        print('DUPLICATE KEYS:', dupes) if dupes else print('no dupes,', len(keys), 'keys')
        sys.exit(1 if dupes else 0)
"
```
Expected: `no dupes, <count> keys`.

- [ ] **Step 5: Run the full test suite**

Run: `C:/Python314/python.exe -m pytest tests/ -v`
Expected: ALL PASS. If `test_title_outweighs_body_frequency` or other ranking tests regress, the tags bonus is not the cause (test rows carry no tags) — inspect which synonym expansion changed the ranking and adjust that expansion's wording, not the test.

- [ ] **Step 6: Regenerate the recall report and record**

Run: `C:/Python314/python.exe scripts/15_retrieval_recall.py`
Expected: full_hit@8 ≥ 0.70 (target after Task 3; ≥0.75 is the work-item A acceptance across Tasks 1–3). Record the by_domain movement (BSA should rise from 0.25). One measurement run — do not loop synonyms against this number.

- [ ] **Step 7: Commit**

```bash
git add src/nyaya/retrieval.py tests/test_retrieval.py reports/retrieval_recall.json
git commit -m "Retrieval: synonym clusters for weak domains; index tags + punishment_summary

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Phrase-coverage metric for the phrase-only gold set

363/495 frozen-eval questions have no citation-resolvable gold — their required_facts are phrases ("report on 1930", "Zero FIR"). Before building the procedures KB (Task 5), add the metric that will prove the KB works: what fraction of those questions have at least one required phrase verbatim in the top-k retrieved text. Committing the baseline first makes the KB's delta undeniable.

**Files:**
- Modify: `scripts/15_retrieval_recall.py`
- Output: `reports/retrieval_recall.json` (regenerated with new field)

**Interfaces:**
- Produces: `report["phrase_coverage"] = {"n": int, "any_fact_in_topk": float}` in the recall JSON.

- [ ] **Step 1: Implement**

Add to `scripts/15_retrieval_recall.py` after `gold_keys`:

```python
def phrase_coverage(records, index, k: int) -> dict:
    """For records with NO resolvable citation gold (the phrase-only set):
    fraction whose top-k retrieved text contains at least one required_fact
    verbatim (case-insensitive). The KB-coverage metric — statute recall@k
    cannot see these records at all."""
    covered = total = 0
    for rec in records:
        facts = [f for f in rec.get("required_facts", []) if f.strip()]
        if not facts:
            continue
        gold, _ = gold_keys(rec, index)
        if gold:
            continue
        total += 1
        blob = " ".join(
            f"{h.get('title') or ''} {h.get('text') or ''}"
            for h in index.retrieve(rec["question"], k=k)).lower()
        if any(f.lower() in blob for f in facts):
            covered += 1
    return {"n": total,
            "any_fact_in_topk": round(covered / total, 4) if total else 0.0}
```

In `main()`, after the `by_domain` block and before `measured_at`:

```python
    report["phrase_coverage"] = phrase_coverage(records, index, max_k)
```

- [ ] **Step 2: Run and record the baseline**

Run: `C:/Python314/python.exe scripts/15_retrieval_recall.py`
Expected: `phrase_coverage.n` ≈ 360–370; `any_fact_in_topk` low (expected under 0.10 — statute text occasionally contains a gold phrase like "grounds of arrest", but helplines/portals never). This baseline is the number Task 5 must beat.

- [ ] **Step 3: Commit**

```bash
git add scripts/15_retrieval_recall.py reports/retrieval_recall.json
git commit -m "Recall report: phrase-coverage metric for the phrase-only gold set (baseline)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Procedural knowledge base (B)

Create `data/canonical/procedures_kb.jsonl` — the "procedure KB" the data README has promised since day one. Rows reuse the `StatuteSection` shape so `load_statute_index()` ingests them with zero loader changes. Two small rendering changes make guidance rows present honestly in the RAG prompt.

**Files:**
- Create: `data/canonical/procedures_kb.jsonl`
- Create: `tests/test_procedures_kb.py`
- Modify: `src/nyaya/retrieval.py` (`format_context` line 299, `RAG_ANSWER_PROMPT` line 381)
- Output: `reports/retrieval_recall.json` (regenerated)

**Interfaces:**
- Produces: KB rows with `act_id="procedures_kb"`, `act_name="Official Procedural Guidance (India)"`, `section=<kebab-slug>`; `format_context` renders them as `"{title} — official guidance\n{text}"`.
- Consumes: field indexing from Task 3 (KB rows carry `tags`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_procedures_kb.py`:

```python
"""The procedures KB: schema, ingestion, and honest prompt rendering."""

import json
from pathlib import Path

import pytest

from nyaya.retrieval import format_context, load_statute_index
from nyaya.validators import load_statute_db

KB_PATH = Path(__file__).resolve().parents[1] / "data" / "canonical" / "procedures_kb.jsonl"
CANONICAL_DIR = KB_PATH.parent


@pytest.fixture(scope="module")
def kb_rows():
    assert KB_PATH.exists(), "data/canonical/procedures_kb.jsonl not built yet"
    with KB_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestKbSchema:
    def test_minimum_coverage(self, kb_rows):
        assert len(kb_rows) >= 60

    def test_constant_act_identity(self, kb_rows):
        assert {r["act_id"] for r in kb_rows} == {"procedures_kb"}
        assert {r["act_name"] for r in kb_rows} == {"Official Procedural Guidance (India)"}

    def test_slugs_unique_and_kebab(self, kb_rows):
        slugs = [r["section"] for r in kb_rows]
        assert len(slugs) == len(set(slugs))
        assert all(s == s.lower() and " " not in s for s in slugs)

    def test_every_row_has_content_and_source(self, kb_rows):
        for r in kb_rows:
            assert r["title"].strip() and len(r["text"].split()) >= 30, r["section"]
            assert r["source_url"].startswith("http"), r["section"]
            assert r.get("tags"), r["section"]  # lay/Hindi terms power retrieval


class TestKbIngestion:
    @pytest.fixture(scope="class")
    def index(self):
        return load_statute_index(CANONICAL_DIR)

    def test_cyber_fraud_query_surfaces_1930_guidance(self, index):
        hits = index.retrieve("I got scammed online, someone took money from my "
                              "account. Where do I report?", k=8)
        blob = " ".join(h["text"] for h in hits)
        assert "1930" in blob and "cybercrime.gov.in" in blob

    def test_kb_does_not_crowd_out_statute_lookup(self, index):
        hits = index.retrieve("What is the punishment under Section 103 of the BNS?", k=4)
        assert hits[0]["act_id"] == "bns_2023" and hits[0]["section"] == "103"

    def test_kb_rows_excluded_from_citation_whitelist(self):
        db = load_statute_db(CANONICAL_DIR)
        assert "procedures_kb" not in db  # guidance is not citable statute


class TestKbRendering:
    def test_guidance_renders_without_section_framing(self):
        row = {"act_id": "procedures_kb",
               "act_name": "Official Procedural Guidance (India)",
               "section": "cyber-fraud-reporting", "title": "Reporting cyber fraud",
               "text": "Call 1930 immediately."}
        out = format_context([row])
        assert out == "Reporting cyber fraud — official guidance\nCall 1930 immediately."
        assert "Section cyber-fraud-reporting" not in out

    def test_statute_rendering_unchanged(self):
        row = {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023",
               "section": "318", "title": "Cheating", "text": "Whoever deceives."}
        assert format_context([row]) == (
            "Section 318 of the Bharatiya Nyaya Sanhita, 2023 — Cheating\nWhoever deceives.")
```

- [ ] **Step 2: Run to verify failure**

Run: `C:/Python314/python.exe -m pytest tests/test_procedures_kb.py -v`
Expected: FAIL — KB file missing, and `format_context` renders guidance as "Section cyber-fraud-reporting of the …".

- [ ] **Step 3: Update rendering and prompt**

In `src/nyaya/retrieval.py`, replace `format_context`:

```python
def format_context(rows: list[dict]) -> str:
    """Verbatim statute/guidance block for the RAG prompt — the ONLY source of
    truth. Statutes render citably; procedures-KB rows render as named official
    guidance (they are not statute and must not look citable as sections)."""
    blocks = []
    for r in rows:
        if r["act_id"] == "procedures_kb":
            blocks.append(f"{r['title']} — official guidance\n{r['text']}")
        else:
            blocks.append(
                f"Section {r['section']} of the {r['act_name']} — {r['title']}\n{r['text']}"
            )
    return "\n\n".join(blocks)
```

Replace `RAG_ANSWER_PROMPT` (keep the exact phrase "do not cite any section not shown above" — a test asserts it):

```python
RAG_ANSWER_PROMPT = """\
Relevant provisions of current Indian law and official guidance (the ONLY \
sources you may rely on):

{context}

Using ONLY the material above where it is relevant, answer the citizen's
question below. Cite statute as "Section <n> of the <Act Name>" exactly as
given above; do not cite any section not shown above. Entries marked "official
guidance" give practical steps (helplines, portals, timelines) you may state
directly. If the material above does not cover the question, say so plainly and
give general guidance without inventing citations. Answer in the same language
as the question.

Question: {question}"""
```

Note for the v3 handoff (goes in this task's commit message): the prompt wording changed, so v2 adapters are mildly off-distribution under it — the teammate must rebuild the RAG dataset (`scripts/18`) and train v3 on the new prompt before comparing adapter numbers.

- [ ] **Step 4: Author the KB rows**

Create `data/canonical/procedures_kb.jsonl` with **60–100 rows**. Authoring rules:

- One JSON object per line, fields exactly: `act_id` (`"procedures_kb"`), `act_name` (`"Official Procedural Guidance (India)"`), `section` (unique kebab-case slug), `title`, `text` (30–150 words, plain language, English; name the statute section inline when one applies), `chapter` (topic group, e.g. `"cyber"`), `subsection` (null), `effective_date` (null), `replaces` (null), `punishment_summary` (null), `tags` (4–10 lay/Hindi/Hinglish search terms), `source_url` (official page the content was verified against).
- Verify every fact against the official source before writing it (WebFetch the URL; zero-budget sources only): cybercrime.gov.in, consumerhelpline.gov.in, edaakhil.nic.in, rti.gov.in, nalsa.gov.in, ncw.nic.in, morth.nic.in, india.gov.in, indiacode.nic.in.
- Do NOT open `data/eval/nyaya_eval_v0.jsonl` while authoring. Topic selection comes from the checklist below (derived from `reports/error_analysis.json` failure modes, which is fair — it describes topics, not answers).

Topic checklist (one row per bullet unless noted; ~70 total):

- **cyber (12):** reporting financial cyber fraud — 1930 helpline golden-hour; cybercrime.gov.in portal complaint steps; digital arrest scams — do not pay, no agency arrests by video call; sextortion response; social-media account hacking; morphed-image/NCII reporting; online job fraud; UPI wrong-transfer recovery; SIM-swap fraud; cyber cell vs local police station jurisdiction; evidence preservation for cyber complaints (screenshots, URLs, transaction IDs); IT Act vs BNS charges for online cheating overview.
- **police & FIR (12):** Zero FIR — any station regardless of jurisdiction; e-FIR online filing; FIR must be free, copy free (BNSS 173); remedy when police refuse FIR — SP then Magistrate (BNSS 173(4)/175(3)); rights on arrest — grounds in writing, inform relative (BNSS 47/48); women cannot be arrested after sunset save exceptions; 24-hour magistrate production rule; arrest of women by women officers; bail for bailable offences is a right; first-time offender undertrial release at one-third (BNSS 479); police complaint against police; missing-person report procedure.
- **women (8):** 181 women helpline & one-stop centres; DV complaint route — protection officer, Magistrate, protection/residence orders (DV Act 12/18/17); POSH internal committee complaint, 3-month window; dowry harassment complaint route; 112 emergency; free legal aid for all women (NALSA); maintenance claim routes (BNSS 144); workplace maternity rights overview.
- **consumer (8):** where to complain by value — district/state/national commission tiers; e-daakhil online filing steps; 1915 national consumer helpline; complaint prerequisites — bill, notice, limitation 2 years; e-commerce grievance officer escalation; misleading-ads complaint; product-liability overview; insurance-claim rejection escalation (ombudsman).
- **rti (7):** how to file RTI — written request, Rs 10 fee, PIO; 30-day reply clock, 48-hour life-liberty; first appeal within 30 days; second appeal to Information Commission; BPL fee exemption; exemptions overview (Section 8 RTI Act); RTI online portal for central bodies.
- **cheque & money (6):** cheque-bounce steps — demand notice within 30 days of return memo, drawer has 15 days, complaint within 1 month after (NI Act 138); where to file cheque-bounce complaint; interim compensation 20% (NI Act 143A); appeal deposit 20% (NI Act 148); legal-notice basics; money-recovery small-cause route.
- **motor vehicles (8):** accident FIR + Detailed Accident Report; claims tribunal (MACT) application (MV Act 166); hit-and-run compensation scheme (MV Act 161); Good Samaritan protections (MV Act 134A); third-party insurance mandatory; e-challan checking and contesting; licence/RC documents via DigiLocker are valid; drunk-driving penalties (MV Act 185).
- **courts & legal aid (9):** free legal aid eligibility & how to apply (NALSA/DLSA); Lok Adalat; plea bargaining basics (BNSS 289); how to engage an advocate / Bar Council complaints; e-courts case status checking; limitation periods overview; anticipatory bail basics (BNSS 482); mediation for family disputes; senior-citizen maintenance tribunal (MWPSC Act).

Three fully-worked example rows (write these verbatim as rows 1–3, then continue the pattern):

```json
{"act_id": "procedures_kb", "act_name": "Official Procedural Guidance (India)", "section": "cyber-fraud-reporting", "title": "Reporting online financial fraud (1930 and cybercrime.gov.in)", "text": "If money was taken from your account through an online scam, call the national cyber crime helpline 1930 immediately — reporting within the first few hours (the golden hour) lets banks freeze the fraudulent transaction chain. Also file a written complaint on the National Cyber Crime Reporting Portal at cybercrime.gov.in with transaction IDs, screenshots and the fraudster's number or UPI ID. You do not need to visit a police station first, and you may additionally file a Zero FIR at any police station. Online cheating is punishable under Section 318 of the Bharatiya Nyaya Sanhita, 2023 and Section 66D of the Information Technology Act, 2000.", "chapter": "cyber", "subsection": null, "effective_date": null, "replaces": null, "punishment_summary": null, "tags": ["1930", "cybercrime.gov.in", "online fraud", "otp fraud", "upi fraud", "scammed", "साइबर धोखाधड़ी", "paisa wapas", "scam report"], "source_url": "https://cybercrime.gov.in/"}
{"act_id": "procedures_kb", "act_name": "Official Procedural Guidance (India)", "section": "zero-fir", "title": "Zero FIR — filing an FIR at any police station", "text": "A Zero FIR can be registered at ANY police station regardless of where the offence happened; the station records it with serial number zero and transfers it to the station with jurisdiction. Section 173 of the Bharatiya Nagarik Suraksha Sanhita, 2023 obliges the officer in charge to record information about a cognizable offence irrespective of the area where it was committed, and information may also be given electronically (e-FIR). Registration is free and you are entitled to a free copy of the FIR. If the police refuse, send your complaint in writing to the Superintendent of Police, and thereafter you may approach the Magistrate.", "chapter": "police-fir", "subsection": null, "effective_date": null, "replaces": null, "punishment_summary": null, "tags": ["zero fir", "e-fir", "fir kaise kare", "police refuse fir", "एफआईआर", "shikayat darj"], "source_url": "https://www.indiacode.nic.in/handle/123456789/20099"}
{"act_id": "procedures_kb", "act_name": "Official Procedural Guidance (India)", "section": "cheque-bounce-notice-timeline", "title": "Cheque bounce — demand notice and complaint timeline", "text": "When a cheque is returned unpaid, the payee must send a written demand notice to the drawer within 30 days of receiving the bank's return memo. The drawer then has 15 days from receipt of the notice to pay. If payment is not made, a complaint under Section 138 of the Negotiable Instruments Act, 1881 must be filed within one month after that 15-day period ends, before the Magistrate where the payee's bank branch is located. Courts may order interim compensation up to 20 percent of the cheque amount under Section 143A of the Negotiable Instruments Act, 1881.", "chapter": "cheque-money", "subsection": null, "effective_date": null, "replaces": null, "punishment_summary": null, "tags": ["cheque bounce", "check bounce", "138 notice", "चेक बाउंस", "demand notice", "cheque return memo"], "source_url": "https://www.indiacode.nic.in/handle/123456789/2189"}
```

- [ ] **Step 5: Run the KB tests, then the full suite**

Run: `C:/Python314/python.exe -m pytest tests/test_procedures_kb.py tests/test_retrieval.py tests/test_evaluation.py -v`
Expected: ALL PASS. `TestRealStatuteDB::test_indexes_every_canonical_row` (asserts >2000 rows) still passes — the KB only adds rows.

- [ ] **Step 6: Measure**

Run: `C:/Python314/python.exe scripts/15_retrieval_recall.py`
Expected: `phrase_coverage.any_fact_in_topk` rises substantially above the Task 4 baseline (KB acceptance: it must at least triple the baseline; report the number honestly either way). Statute `full_hit@8` must NOT regress by more than 0.01 (KB rows competing in top-k) — if it does, that is a real finding: raise it with the user rather than quietly tuning.

- [ ] **Step 7: Commit**

```bash
git add data/canonical/procedures_kb.jsonl tests/test_procedures_kb.py src/nyaya/retrieval.py reports/retrieval_recall.json
git commit -m "Procedures KB: <n> official-guidance rows; guidance-aware RAG rendering

Phrase coverage <baseline> -> <new>; statute full_hit@8 <before> -> <after>.
NOTE for v3: RAG prompt wording changed — rebuild the RAG dataset (scripts/18)
and retrain before comparing adapter numbers.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
(Fill in the measured numbers.)

---

### Task 6: Extraction-focused training data generator (C)

Rule-based generator over `data/canonical/` producing verbatim-fact QA pairs (punishments, deadlines, mappings) in `TrainingRecord` shape. No LLM. Sections reachable from any frozen-eval record are excluded.

**Files:**
- Create: `scripts/19_generate_extraction_data.py`
- Create: `tests/test_extraction_data.py`
- Modify: `src/nyaya/validators.py:215-227` (register the new task type)

**Interfaces:**
- Consumes: `load_statute_index`, `referenced_keys(fact, domain=...)` (Task 1), `NYAYA_SYSTEM_PROMPT` from `nyaya.prompts`, `load_eval_records` from `nyaya.evaluation`.
- Produces: `data/generated/extraction_qa_v1.jsonl` (gitignored — the script is the deliverable) with `metadata.task_type` `"extraction_qa"` or `"law_mapping"`, `metadata.generator="rule_extraction_v1"`, `metadata.dataset_version="extraction_v1"`; importable functions `extract_punishment_years(text) -> str | None`, `extract_deadline_days(text) -> str | None`, `eval_excluded_keys(index, records) -> set[str]`, `generate_records(index, mappings, excluded) -> list[dict]`.

- [ ] **Step 1: Register the task type in validators**

In `src/nyaya/validators.py`, add `"extraction_qa"` to `_GROUNDED_TASK_TYPES`:

```python
_GROUNDED_TASK_TYPES = {
    "grounded_qa", "hindi_qa", "hinglish_qa", "terminology", "procedural",
    "law_mapping", "extraction_qa",
}
```

and to `MIN_WORDS_BY_TASK` (extraction answers are legitimately compact — the whole point is the exact fact, not an essay):

```python
MIN_WORDS_BY_TASK = {
    "law_mapping": 20,
    "terminology": 50,
    "safety_abstention": 1,
    "extraction_qa": 15,
}
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_extraction_data.py`:

```python
"""Rule-based extraction-QA generator: precision over volume, zero eval leakage."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "extraction_gen",
    Path(__file__).resolve().parents[1] / "scripts" / "19_generate_extraction_data.py")
extraction_gen = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(extraction_gen)

from nyaya.retrieval import StatuteIndex
from nyaya.validators import validate_example

ROWS = [
    {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023",
     "section": "318", "title": "Cheating",
     "text": "Whoever commits cheating shall be punished with imprisonment of either "
     "description for a term which may extend to three years, or with fine, or with both.",
     "chapter": "XVII"},
    {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023",
     "section": "999", "title": "Multi-clause provision",
     "text": "shall be punished with imprisonment which may extend to three years; "
     "and in aggravated cases with imprisonment which may extend to seven years.",
     "chapter": "X"},
    {"act_id": "rti_2005", "act_name": "Right to Information Act, 2005",
     "section": "7", "title": "Disposal of request",
     "text": "The request shall be disposed of within thirty days of the receipt "
     "of the request.", "chapter": "II"},
]
MAPPINGS = [
    {"old_act": "IPC", "old_section": "420", "new_act": "BNS",
     "new_section": "318", "note": None},
]


class TestFactExtraction:
    def test_punishment_years(self):
        assert extraction_gen.extract_punishment_years(ROWS[0]["text"]) == "three"

    def test_ambiguous_punishment_skipped(self):
        assert extraction_gen.extract_punishment_years(ROWS[1]["text"]) is None

    def test_deadline_days(self):
        assert extraction_gen.extract_deadline_days(ROWS[2]["text"]) == "thirty"

    def test_no_fact_returns_none(self):
        assert extraction_gen.extract_punishment_years("No numbers here.") is None


class TestGeneration:
    def test_generates_valid_records(self):
        index = StatuteIndex(ROWS, MAPPINGS)
        records = extraction_gen.generate_records(index, MAPPINGS, excluded=set())
        assert records, "generator produced nothing from extractable rows"
        # ipc whitelisted like the real pipeline (load_statute_db include_old_law=True)
        # — mapping answers legitimately mention the repealed section
        db = {"bns": {"318", "999"}, "rti": {"7"}, "ipc": {"420"}}
        for rec in records:
            ok, reasons = validate_example(rec, db, eval_records=[])
            assert ok, (rec["id"], reasons)

    def test_excluded_sections_produce_nothing(self):
        index = StatuteIndex(ROWS, MAPPINGS)
        excluded = {"bns_2023:318", "bns_2023:999", "rti_2005:7"}
        records = extraction_gen.generate_records(index, MAPPINGS, excluded=excluded)
        assert all(
            not (set(r["metadata"]["source_sections"]) & excluded) for r in records)
        # mapping records source from the NEW section — 318 excluded kills them too
        assert not any(r["metadata"]["task_type"] == "law_mapping" for r in records)

    def test_ids_deterministic(self):
        index = StatuteIndex(ROWS, MAPPINGS)
        a = extraction_gen.generate_records(index, MAPPINGS, excluded=set())
        b = extraction_gen.generate_records(index, MAPPINGS, excluded=set())
        assert [r["id"] for r in a] == [r["id"] for r in b]
```

- [ ] **Step 3: Run to verify failure**

Run: `C:/Python314/python.exe -m pytest tests/test_extraction_data.py -v`
Expected: FAIL at module load — `scripts/19_generate_extraction_data.py` does not exist.

- [ ] **Step 4: Implement the generator**

Create `scripts/19_generate_extraction_data.py`:

```python
"""Step 19 — rule-based extraction-QA training data (v3).

v2 showed extraction is a binding constraint: with the gold section IN
context, strict accuracy tops out ~21% — the model paraphrases instead of
copying the exact duration/fine/date. These examples teach verbatim copying:
question + the section, answer restating the exact fact with the citation.

No LLM anywhere (zero budget, deterministic). Precision over volume: a section
whose text matches a pattern more than once with different values is skipped.
Sections reachable from ANY frozen-eval record (gold facts, forbidden facts,
or the question itself) are excluded — over-exclusion is the correct failure
direction.

Usage:
    python scripts/19_generate_extraction_data.py
    python scripts/19_generate_extraction_data.py --cap-per-act 150
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.evaluation import load_eval_records
from nyaya.prompts import NYAYA_SYSTEM_PROMPT
from nyaya.retrieval import _FAMILY_TO_ACT_ID, load_statute_index

DATASET_VERSION = "extraction_v1"

_PUNISH = re.compile(
    r"imprisonment[^.;]{0,120}?may extend to ([a-z\- ]+?) years?", re.I)
_DAYS = re.compile(r"within ([a-z\-]+) days", re.I)

WORD2NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
            "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30,
            "forty-five": 45, "sixty": 60, "ninety": 90}


def _unique_match(pattern: re.Pattern, text: str) -> str | None:
    """The single distinct captured value, or None (absent or ambiguous)."""
    values = {m.group(1).strip().lower() for m in pattern.finditer(text)}
    return values.pop() if len(values) == 1 else None


def extract_punishment_years(text: str) -> str | None:
    return _unique_match(_PUNISH, text)


def extract_deadline_days(text: str) -> str | None:
    return _unique_match(_DAYS, text)


def eval_excluded_keys(index, records) -> set[str]:
    """Every statute key any frozen-eval record can reach — never train
    extraction on these."""
    excluded: set[str] = set()
    for rec in records:
        domain = rec.get("legal_domain")
        for fact in rec.get("required_facts", []) + rec.get("forbidden_facts", []):
            excluded.update(index.referenced_keys(fact, domain=domain))
        excluded.update(index.referenced_keys(rec["question"], domain=domain))
    return excluded


def _mk(record_id, question, answer, task_type, language, act_id, section):
    return {
        "id": record_id,
        "messages": [
            {"role": "system", "content": NYAYA_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "language": language, "legal_domain": act_id.rsplit("_", 1)[0],
            "task_type": task_type, "source_act": act_id,
            "source_sections": [f"{act_id}:{section}"],
            "generator": "rule_extraction_v1", "verified": True,
            "dataset_version": DATASET_VERSION,
        },
    }


def _punishment_records(row, n):
    act, sec, title = row["act_name"], row["section"], row["title"]
    years = extract_punishment_years(row.get("text") or "")
    if years is None:
        return
    yield _mk(
        f"extq_{n:05d}_en_pun", 
        f"What is the maximum imprisonment under Section {sec} of the {act}?",
        f"Under Section {sec} of the {act} ({title}), imprisonment may extend "
        f"to {years} years. Always check the full provision, since fines or "
        f"alternative punishments may also apply, and consult a licensed "
        f"advocate for anything consequential.",
        "extraction_qa", "english", row["act_id"], sec)
    num = WORD2NUM.get(years)
    if num is not None:
        yield _mk(
            f"extq_{n:05d}_hg_pun",
            f"Section {sec} {act} ke under maximum kitne saal ki saza ho sakti hai?",
            f"Section {sec} of the {act} ({title}) ke under imprisonment {num} "
            f"years tak extend ho sakti hai. Poora provision zaroor check karein "
            f"— fine ya alternative punishment bhi ho sakta hai. Kisi bhi "
            f"important case ke liye licensed advocate se consult karein.",
            "extraction_qa", "hinglish", row["act_id"], sec)


def _deadline_records(row, n):
    act, sec, title = row["act_name"], row["section"], row["title"]
    days = extract_deadline_days(row.get("text") or "")
    if days is None:
        return
    yield _mk(
        f"extq_{n:05d}_en_ddl",
        f"What is the time limit mentioned in Section {sec} of the {act}?",
        f"Section {sec} of the {act} ({title}) sets a time limit of {days} "
        f"days. Missing statutory time limits can be fatal to a claim, so act "
        f"promptly and consult a licensed advocate if the deadline is near.",
        "extraction_qa", "english", row["act_id"], sec)


def _mapping_records(mappings, index, excluded, start_n):
    act_names = {r["act_id"]: r["act_name"] for r in index.rows}
    n = start_n
    for m in mappings:
        new_family = m["new_act"].lower()
        new_act_id = _FAMILY_TO_ACT_ID.get(new_family)
        if not new_act_id or new_act_id not in act_names:
            continue
        key = f"{new_act_id}:{m['new_section'].upper()}"
        if key not in index.by_key or key in excluded:
            continue
        full = act_names[new_act_id]
        n += 1
        yield _mk(
            f"extq_{n:05d}_en_map",
            f"Which section of the {m['new_act']} corresponds to "
            f"{m['old_act']} Section {m['old_section']}?",
            f"{m['old_act']} Section {m['old_section']} corresponds to Section "
            f"{m['new_section']} of the {full}. The {m['old_act']} has been "
            f"repealed, so current matters should cite Section {m['new_section']} "
            f"of the {full}.",
            "law_mapping", "english", new_act_id, m["new_section"].upper())


def generate_records(index, mappings, excluded: set[str],
                     cap_per_act: int = 150) -> list[dict]:
    out, per_act = [], Counter()
    for n, row in enumerate(index.rows):
        key = f"{row['act_id']}:{row['section'].upper()}"
        if key in excluded or row["act_id"] == "procedures_kb":
            continue
        for rec in list(_punishment_records(row, n)) + list(_deadline_records(row, n)):
            if per_act[row["act_id"]] >= cap_per_act:
                break
            per_act[row["act_id"]] += 1
            out.append(rec)
    out.extend(_mapping_records(mappings, index, excluded, len(index.rows)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-per-act", type=int, default=150)
    parser.add_argument("--out", default=str(ROOT / "data" / "generated"
                                             / "extraction_qa_v1.jsonl"))
    args = parser.parse_args()

    index = load_statute_index(ROOT / "data" / "canonical")
    mappings_path = ROOT / "data" / "canonical" / "law_mappings.jsonl"
    with mappings_path.open(encoding="utf-8") as fh:
        mappings = [json.loads(line) for line in fh if line.strip()]

    excluded = eval_excluded_keys(index, load_eval_records())
    records = generate_records(index, mappings, excluded, args.cap_per_act)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_type = Counter(r["metadata"]["task_type"] for r in records)
    by_lang = Counter(r["metadata"]["language"] for r in records)
    print(json.dumps({"total": len(records), "excluded_eval_sections": len(excluded),
                      "by_task_type": dict(by_type), "by_language": dict(by_lang),
                      "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests**

Run: `C:/Python314/python.exe -m pytest tests/test_extraction_data.py tests/test_evaluation.py -v`
Expected: ALL PASS (validator changes must not break existing validation tests).

- [ ] **Step 6: Run the generator for real and eyeball**

Run: `C:/Python314/python.exe scripts/19_generate_extraction_data.py`
Expected: total in the 800–3,000 range (pattern strictness decides; if under 500, loosen nothing — report the number). Then print 5 random records and read them for correctness:

```bash
C:/Python314/python.exe -c "
import json, random
rows=[json.loads(l) for l in open('data/generated/extraction_qa_v1.jsonl',encoding='utf-8')]
random.seed(7)
for r in random.sample(rows, 5):
    print(r['messages'][1]['content']); print(r['messages'][2]['content']); print('---')
"
```
Verify each answer's number actually appears in the cited section (spot-check one against `data/canonical/`).

- [ ] **Step 7: Commit**

```bash
git add scripts/19_generate_extraction_data.py tests/test_extraction_data.py src/nyaya/validators.py
git commit -m "Extraction-QA generator: rule-based verbatim-fact training data for v3

<total> records (<by_type>), eval-reachable sections excluded (<n> keys).
Teammate: run scripts/19, then 05->06->07->18 to fold into the v3 dataset.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7 (optional, last): Dense hybrid retrieval stage (A4)

Build only if Tasks 1–6 are done and pushed. Ships disabled; base `requirements.txt` untouched.

**Files:**
- Create: `src/nyaya/dense.py`, `requirements-dense.txt`
- Modify: `src/nyaya/retrieval.py` (`StatuteIndex.retrieve`, `load_statute_index`), `scripts/15_retrieval_recall.py` (add `--dense` flag), `.gitignore`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Produces: `load_statute_index(canonical_dir, dense_model: str | None = None)`; when set, `StatuteIndex.retrieve` fuses BM25 and dense rankings with reciprocal-rank fusion (exact-reference stage 1 unchanged, always first). `nyaya.dense.DenseStage(rows, model_name, cache_dir).rank(query) -> list[int]` (row indices, best first).

- [ ] **Step 1: requirements + gitignore**

Create `requirements-dense.txt`:

```
sentence-transformers>=3.0
```

Append to `.gitignore`:

```
data/canonical/.dense_cache/
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_retrieval.py`:

```python
class TestDenseFusion:
    def test_rrf_fusion_merges_rankings(self):
        # pure-logic test of the fusion math — no model download
        from nyaya.retrieval import rrf_fuse
        fused = rrf_fuse([[0, 1, 2], [2, 0, 1]], c=60)
        assert fused[0] == 0          # rank 1 + rank 2 beats rank 3 + rank 1
        assert set(fused) == {0, 1, 2}

    def test_dense_disabled_by_default(self, index):
        assert getattr(index, "dense", None) is None


class TestDenseStage:
    def test_semantic_query_beats_lexical_miss(self, tmp_path):
        pytest.importorskip("sentence_transformers")  # dense extra not installed -> skip
        from nyaya.dense import DenseStage
        idx = StatuteIndex(ROWS, MAPPINGS)
        idx.dense = DenseStage(ROWS, cache_dir=tmp_path)
        # zero lexical overlap with Section 318's text; meaning overlaps
        hits = idx.retrieve("A conman fooled my dad into signing over his flat", k=2)
        assert any(h["section"] == "318" for h in hits)
```

- [ ] **Step 3: Run to verify failure**

Run: `C:/Python314/python.exe -m pytest tests/test_retrieval.py::TestDenseFusion -v`
Expected: FAIL — `ImportError: cannot import name 'rrf_fuse'`.

- [ ] **Step 4: Implement**

Create `src/nyaya/dense.py`:

```python
"""Optional dense retrieval stage (A4). Lazily imported — the base package
stays dependency-free; install requirements-dense.txt to enable.

multilingual-e5-small: free, ~470 MB, handles Devanagari — measured locally
before ever being enabled anywhere else. e5 models REQUIRE the "query: " /
"passage: " prefixes; scores are meaningless without them.
"""

import hashlib
from pathlib import Path

import numpy as np


class DenseStage:
    def __init__(self, rows: list[dict],
                 model_name: str = "intfloat/multilingual-e5-small",
                 cache_dir: str | Path | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        passages = [
            f"passage: {r.get('title') or ''}. {r.get('text') or ''}" for r in rows
        ]
        fingerprint = hashlib.sha256(
            (model_name + "\x00".join(passages)).encode()).hexdigest()[:16]
        cache_file = (Path(cache_dir) / f"{fingerprint}.npy") if cache_dir else None
        if cache_file and cache_file.exists():
            self.embeddings = np.load(cache_file)
        else:
            self.embeddings = self.model.encode(
                passages, normalize_embeddings=True, show_progress_bar=True)
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, self.embeddings)

    def rank(self, query: str) -> list[int]:
        q = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        sims = self.embeddings @ q
        return list(map(int, sims.argsort()[::-1]))
```

In `src/nyaya/retrieval.py`, add module-level:

```python
def rrf_fuse(rankings: list[list[int]], c: int = 60) -> list[int]:
    """Reciprocal-rank fusion: score(i) = sum over rankings of 1/(c + rank)."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] += 1.0 / (c + rank + 1)
    return [idx for idx, _s in sorted(scores.items(), key=lambda kv: -kv[1])]
```

In `StatuteIndex.__init__`, add `self.dense = None` (last line). Replace the BM25 loop in `retrieve` with fusion-aware selection:

```python
    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        picked = []
        for key in self.referenced_keys(query):
            picked.append(self.rows[self.by_key[key]])
            if len(picked) >= k:
                return picked
        chosen = {f"{r['act_id']}:{r['section'].upper()}" for r in picked}
        bm25_order = [i for _score, i in self._bm25(query)]
        if self.dense is not None:
            order = rrf_fuse([bm25_order, self.dense.rank(query)])
        else:
            order = bm25_order
        for i in order:
            row = self.rows[i]
            if f"{row['act_id']}:{row['section'].upper()}" in chosen:
                continue
            picked.append(row)
            if len(picked) >= k:
                break
        return picked
```

In `load_statute_index`, extend the signature and wire it:

```python
def load_statute_index(canonical_dir: str | Path,
                       dense_model: str | None = None) -> StatuteIndex:
    """Build a StatuteIndex from data/canonical/*.jsonl (mappings included).
    dense_model (e.g. "intfloat/multilingual-e5-small") enables the optional
    hybrid stage — requires requirements-dense.txt."""
    directory = Path(canonical_dir)
    rows, mappings = [], []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        if path.name.startswith("law_mappings"):
            mappings.extend(records)
        else:
            rows.extend(records)
    if not rows:
        raise FileNotFoundError(
            f"no statute JSONL found in {directory} — run scripts/03_build_corpus.py")
    index = StatuteIndex(rows, mappings)
    if dense_model:
        from .dense import DenseStage
        index.dense = DenseStage(rows, dense_model,
                                 cache_dir=directory / ".dense_cache")
    return index
```

In `scripts/15_retrieval_recall.py`, add the flag and wire it:

```python
    parser.add_argument("--dense", nargs="?", const="intfloat/multilingual-e5-small",
                        default=None, metavar="MODEL")
```
and change the index construction to `index = load_statute_index(args.canonical_dir, dense_model=args.dense)`; also record it: add `"dense_model": args.dense,` as the first key of `report`.

- [ ] **Step 5: Install the extra and run tests**

Run: `C:/Python314/python.exe -m pip install -r requirements-dense.txt`
then `C:/Python314/python.exe -m pytest tests/test_retrieval.py -v`
Expected: ALL PASS (first DenseStage test downloads ~470 MB once).

- [ ] **Step 6: Measure both ways**

Run: `C:/Python314/python.exe scripts/15_retrieval_recall.py` (baseline check — must be unchanged from Task 5) then `C:/Python314/python.exe scripts/15_retrieval_recall.py --dense`.
Expected: the dense run's `full_hit@8` and `by_language` hindi/hinglish rows are the decision data. Report both to the user; do NOT overwrite `reports/retrieval_recall.json` with the dense run — save it as `reports/retrieval_recall_dense.json` (copy the file before the second run, or run with the report path temporarily redirected; simplest is: run BM25-only last so the canonical report stays BM25-only, and `git add` the dense copy separately).

- [ ] **Step 7: Commit**

```bash
git add src/nyaya/dense.py src/nyaya/retrieval.py scripts/15_retrieval_recall.py requirements-dense.txt .gitignore tests/test_retrieval.py reports/retrieval_recall_dense.json
git commit -m "Optional dense hybrid retrieval (multilingual-e5-small + RRF), disabled by default

BM25-only full_hit@8 <x>; hybrid <y>. Enable with --dense after installing
requirements-dense.txt.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Final step: push and hand off

- [ ] Run the whole suite one last time: `C:/Python314/python.exe -m pytest tests/ -v` — ALL PASS.
- [ ] `git push origin main`
- [ ] Summarize for the teammate (in chat, and the commit messages already carry it): pull latest; rebuild the RAG dataset with `scripts/18` (new prompt + KB rows now flow into contexts); run `scripts/19` and fold `data/generated/extraction_qa_v1.jsonl` through `scripts/05 → 06 → 07`; retrain v3; re-run `scripts/16` on the frozen set; only enable `--dense` if the locally-measured delta justified it.
