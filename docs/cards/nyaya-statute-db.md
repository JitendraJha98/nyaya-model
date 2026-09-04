---
license: other
license_name: gov-india-public-domain
license_link: https://www.indiacode.nic.in/
language:
- en
- hi
tags:
- legal
- india
- indian-law
- bns
- bnss
- bsa
- statutes
- retrieval
pretty_name: Nyaya Statute DB — current Indian law, section by section
size_categories:
- 1K<n<10K
task_categories:
- text-retrieval
- question-answering
---

# Nyaya Statute DB

**Current Indian law as clean, section-level JSONL** — the Bharatiya Nyaya
Sanhita, Bharatiya Nagarik Suraksha Sanhita and Bharatiya Sakshya Adhiniyam as
in force after **1 July 2024**, plus the official old→new mapping tables, the
Constitution and fourteen other acts a citizen actually runs into.

Most Indian legal datasets predate the 2024 criminal-law overhaul and still
describe the IPC, CrPC and Evidence Act as current law. This one does not.

## Contents — 4,271 rows: 18 act files, 1 mapping table, 1 guidance file

| File | Rows | Act |
|---|---:|---|
| `bns_2023.jsonl` | 358 | Bharatiya Nyaya Sanhita, 2023 |
| `bnss_2023.jsonl` | 531 | Bharatiya Nagarik Suraksha Sanhita, 2023 |
| `bsa_2023.jsonl` | 170 | Bharatiya Sakshya Adhiniyam, 2023 |
| `constitution_1950.jsonl` | 498 | Constitution of India |
| `mv_act_1988.jsonl` | 257 | Motor Vehicles Act, 1988 |
| `ni_act_1881.jsonl` | 154 | Negotiable Instruments Act, 1881 (cheque bounce) |
| `it_act_2000.jsonl` | 122 | Information Technology Act, 2000 |
| `cpa_2019.jsonl` | 107 | Consumer Protection Act, 2019 |
| `wages_code_2019.jsonl` | 69 | Code on Wages, 2019 |
| `sma_1954.jsonl` | 57 | Special Marriage Act, 1954 |
| `hma_1955.jsonl` | 37 | Hindu Marriage Act, 1955 |
| `dv_act_2005.jsonl` | 37 | Protection of Women from Domestic Violence Act, 2005 |
| `rti_2005.jsonl` | 31 | Right to Information Act, 2005 |
| `posh_2013.jsonl` | 30 | POSH Act, 2013 |
| `tpa_1882.jsonl` | 135 | Transfer of Property Act, 1882 (from the India Code API) |
| `contract_act_1872.jsonl` | 192 | Indian Contract Act, 1872 (from the India Code API) |
| `pocso_2012.jsonl` | 47 | Protection of Children from Sexual Offences Act, 2012 (from the India Code API) |
| `jj_act_2015.jsonl` | 112 | Juvenile Justice (Care and Protection of Children) Act, 2015 (from the India Code API) |
| **`law_mappings.jsonl`** | **1,257** | **official IPC↔BNS / CrPC↔BNSS / IEA↔BSA mappings** |
| `procedures_kb.jsonl` | 70 | hand-written "how do I…" procedural guidance |

## Schema

```json
{
  "act_id": "bns_2023",
  "act_name": "Bharatiya Nyaya Sanhita, 2023",
  "section": "103",
  "title": "Punishment for murder",
  "text": "Whoever commits murder shall be punished with death or imprisonment for life...",
  "chapter": "...",
  "subsection": null,
  "effective_date": "2024-07-01",
  "replaces": ["IPC 302"],
  "punishment_summary": "death or imprisonment for life, and shall also be liable to fine",
  "tags": [],
  "source_url": "https://www.indiacode.nic.in/..."
}
```

`replaces` lists the old-law sections a BNS / BNSS / BSA section corresponds to,
derived from the official tables in `law_mappings.jsonl` (present on 346/358 BNS,
479/531 BNSS and 168/170 BSA rows; the mapping file itself remains the source of
truth for old→new lookups, e.g. "Section 302 IPC" → BNS 103). `punishment_summary`
is the section's first penalty clause where one exists (323 rows across all acts).
`tags` are curated keywords and are present only on the guidance notes.

## Intended use

Built as the retrieval corpus for an Indian legal question-answering system.
It suits any RAG setup that needs to cite Indian law by section, and is equally
usable as a plain lookup table.

Measured in that role (409 questions, BM25 + exact-citation lookup): the gold
section reaches the top 8 for **~81%** of questions with a resolvable citation.
Adding a cross-encoder reranker moves it to the **top 1** for 58.5% of them, up
from 45.8%.

## Limitations — read these

- **Coverage is 17 acts plus the Constitution, not all Indian law.** Anything outside them is absent,
  and absence is silent: a retriever will still return the nearest thing it has.
- **Statutes change.** `effective_date` is recorded, but nothing here tracks
  amendments after collection. Verify against
  [India Code](https://www.indiacode.nic.in/) before relying on any provision.
- **No case law.** Statutory text only. Judicial interpretation frequently
  determines what a section means in practice.
- **`procedures_kb.jsonl` is hand-written**, verified against act text but not
  drafted by an advocate.
- Some sections are time-sensitive at the time of collection — notably the
  BNS 106(2) deferral and the labour-code rollout.

## ⚖️ Not legal advice

This is primary legal text for research and information. The practice of law in
India is reserved to advocates enrolled under the Advocates Act, 1961. Consult a
licensed advocate for anything consequential. Free legal aid is available
through NALSA / DLSA under the Legal Services Authorities Act, 1987.

## Licence & provenance

Statutory text is Government of India material, **public domain in India under
Section 52(1)(q) of the Copyright Act, 1957**. Sourced from
[India Code](https://www.indiacode.nic.in/) and legislative.gov.in; per-row
`source_url` records where each section came from. Indian Kanoon HTML was not
scraped.

## Built with

[github.com/JitendraJha98/nyaya-model](https://github.com/JitendraJha98/nyaya-model)
— the extraction pipeline, the retriever, the reranker, and the evaluation
harness are all open, including the four measurement bugs found in the
project's own evaluation and written up in the README.
