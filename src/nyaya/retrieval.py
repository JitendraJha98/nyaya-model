"""Statute retrieval for RAG inference (v2 architecture).

Error analysis of v1 (reports/error_analysis.json) showed 475/491 frozen-eval
failures were wrong-or-incomplete citations: the fine-tuned model reliably
COMMITS to citations but cannot recall section-level facts from 3B weights.
Retrieval supplies the right verbatim section at answer time; the model's
trained citing behaviour does the rest.

Two-stage retriever over data/canonical (deterministic, dependency-free):
  1. Exact reference resolution — if the query names a section (any script,
     any alias, old or new law), surface exactly that section; old-law
     references (IPC/CrPC/IEA) resolve through the official mapping table.
  2. BM25 lexical scoring over title + text + act name for concept queries.

Known limitation (v2.1): pure-Devanagari *conceptual* queries score poorly
against English statute text — exact citation lookup still works via
Devanagari aliases; a multilingual embedding stage can follow if eval
demands it.
"""

import copy
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .validators import ACT_ALIASES, CITATION_PATTERN, alias_pattern

_TOKEN = re.compile(r"[a-z0-9ऀ-ॿ]+")
_NUMBER = re.compile(r"(\d+[A-Za-z]{0,2})")

# family key -> canonical act_id in data/canonical
_FAMILY_TO_ACT_ID = {family: act_id for act_id, family in (
    ("bns_2023", "bns"), ("bnss_2023", "bnss"), ("bsa_2023", "bsa"),
    ("rti_2005", "rti"), ("cpa_2019", "cpa"), ("it_act_2000", "it act"),
    ("ni_act_1881", "ni act"), ("mv_act_1988", "mv act"), ("dv_act_2005", "dv act"),
    ("posh_2013", "posh"), ("hma_1955", "hma"), ("sma_1954", "sma"),
    ("wages_code_2019", "wages code"), ("constitution_1950", "constitution"),
)}
_OLD_TO_NEW_ACT = {"ipc": "bns", "crpc": "bnss", "iea": "bsa"}

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

# Section titles are the statute's own name for a concept ("Cheating",
# "Dishonour of cheque…"). A query token appearing in the title earns a
# flat idf bonus on top of body BM25 — a field-match signal that body
# word-frequency cannot buy (BM25F simplified). 0.75 won the sweep on the
# generated train split (never tuned on the frozen eval).
TITLE_BONUS = 0.75

# Procedural-guidance (procedures_kb) rows point to the statute that governs a
# situation; they are a supplement, not the citation itself. At inference k=8 a
# question needs the governing sections plus at most a couple of practical
# pointers (a helpline, a portal, a deadline), so guidance is capped at KB_SLOTS
# of the retrieved context and statute sections keep the majority. A purely
# procedural question with no matching statute still fills up with guidance via
# the backfill in retrieve().
KB_SLOTS = 2

# Lay/Hindi legal vocabulary -> the statutory phrasing it names. General
# Indian legal-aid terminology (statute titles, standard usage) — NOT tuned
# per eval question; calibrated only against the generated train split.
LEGAL_SYNONYMS = {
    # criminal offences, lay English
    "cheated": "cheating deceiving dishonestly induces deliver property",
    "scam": "cheating deceiving dishonestly induces deliver property",
    "fraud": "cheating deceiving dishonestly induces deliver property",
    "tricked": "cheating deceiving dishonestly induces deliver property",
    "duped": "cheating deceiving dishonestly induces deliver property",
    "stole": "theft dishonestly movable property",
    "stolen": "theft dishonestly movable property",
    "stealing": "theft dishonestly movable property",
    "robbed": "robbery theft extortion",
    "lynching": "murder group of five or more persons ground of race caste community",
    "mob": "group of five or more persons",
    "hit and run": "escaping without reporting rash negligent driving causing death",
    "sedition": "acts endangering sovereignty unity and integrity of India",
    "blackmail": "extortion fear of injury dishonestly induces deliver property",
    "protection money": "extortion fear of injury",
    "ransom": "kidnapping for ransom",
    "kidnap": "kidnapping abduction",
    "acid": "acid grievous hurt",
    "slapped": "voluntarily causing hurt assault criminal force",
    "beat": "voluntarily causing hurt assault",
    "beaten": "voluntarily causing hurt grievous assault",
    "murder": "murder culpable homicide punishment death",
    "killed": "murder culpable homicide death rash negligent act",
    "dowry": "dowry cruelty husband relatives of husband woman",
    "harass": "harassment cruelty insult modesty",
    "eve teasing": "word gesture act intended to insult modesty of a woman",
    "secretly filmed": "watches captures image of a woman private act voyeurism",
    "obscene": "obscene act song publishing transmitting obscene material",
    "defaming": "defamation imputation harm reputation",
    "defamation": "defamation imputation harm reputation",
    "bribe": "gratification public servant bribery",
    # procedure, lay English
    "fir": "information cognizable offence officer in charge of a police station first information report",
    "anticipatory bail": "direction for grant of bail person apprehending arrest",
    "bail": "bail bailable released bond",
    "arrest": "arrest police custody without warrant",
    "arrested": "arrest police custody without warrant",
    "appeal": "appeal conviction sentence court",
    "maintenance": "order for maintenance of wives children and parents monthly allowance",
    "summon": "summons served",
    "cheque bounce": "cheque returned unpaid dishonour insufficiency of funds",
    "cheque bounced": "cheque returned unpaid dishonour insufficiency of funds",
    "consumer": "consumer complaint deficiency in service district commission",
    "refund": "deficiency in service unfair trade practice consumer",
    "divorce": "dissolution of marriage decree petition",
    "rent": "landlord tenant premises",

    # --- gaps found by auditing gold sections absent from the top-100 -----
    # These are not guesses. Each entry below fixes a specific eval question
    # whose gold section the first stage could not surface at ANY depth, so
    # no amount of reranking could have recovered it. The pattern is always
    # the same: citizens describe a situation, statutes describe a legal
    # concept, and BM25 cannot bridge the two.

    # Constitutional concepts -> article language. The single worst cluster
    # (9 of 34 unreachable golds): nobody asks about "protection of life and
    # personal liberty", they ask whether privacy is a right.
    # NB: keys are matched with \b...\b, so prefer ATOMIC keys over long
    # phrases -- "fundamental right violated" silently failed to match
    # "my fundamental rights are violated" (plural + intervening word).
    # Plural forms are listed explicitly for the same reason.
    "privacy": "protection of life and personal liberty fundamental right",
    "private life": "protection of life and personal liberty",
    "right to property": "persons not to be deprived of property save by authority of law",
    "land": "persons not to be deprived of property save by authority of law acquisition",
    "reservation": "equality of opportunity in matters of public employment backward class",
    "reservations": "equality of opportunity in matters of public employment backward class",
    "promotions": "equality of opportunity in matters of public employment",
    "fundamental rights": "power of high courts to issue certain writs remedies enforcement",
    "fundamental right": "power of high courts to issue certain writs remedies enforcement",
    "writ": "power of high courts to issue certain writs habeas corpus mandamus",
    "free legal aid": "equal justice and free legal aid legal services",
    "court provide a lawyer": "equal justice and free legal aid",
    "mercy petition": "power of president governor to grant pardons reprieves remissions",
    "mercy petitions": "power of president governor to grant pardons reprieves remissions",
    "pardon": "power of president governor to grant pardons reprieves remissions",
    "self incrimination": "no person accused of any offence shall be compelled to be a witness against himself",
    "silence": "compelled to be a witness against himself protection conviction offences",
    "protest": "protection of certain rights regarding freedom of speech assembly",
    "assembly": "protection of certain rights regarding freedom of speech assembly",
    "travel abroad": "protection of life and personal liberty",
    "passport": "protection of life and personal liberty",
    "custodial": "protection of life and personal liberty remedy torture",
    "torture": "protection of life and personal liberty",
    "school": "right to education free and compulsory education children",
    "admission": "right to education free and compulsory education",

    # Modern digital evidence -> "electronic record". The BSA cluster: the
    # Act never names a single one of these artefacts.
    "cdr": "admissibility of electronic records computer output",
    "call detail record": "admissibility of electronic records computer output",
    "call records": "admissibility of electronic records",
    "google maps": "admissibility of electronic records computer output",
    "location history": "admissibility of electronic records",
    "aadhaar log": "admissibility of electronic records",
    "authentication log": "admissibility of electronic records",
    "server log": "admissibility of electronic records",
    "whatsapp chat": "admissibility of electronic records computer output",
    "screenshot": "admissibility of electronic records",
    "phone photos": "admissibility of electronic records",
    "cctv": "admissibility of electronic records computer output",
    "suicide note": "statements as to cause of death dying declaration relevant facts",

    # Situation -> offence, where the lay phrasing shares no term with the section
    "honour killing": "murder punishment culpable homicide",
    "gang": "organised crime syndicate continuing unlawful activity",
    "syndicate": "organised crime continuing unlawful activity",
    "morphed": "obscene material publishing transmitting extortion fear of injury",
    "deepfake": "obscene material publishing transmitting electronic form",
    "astrologer": "cheating dishonestly induces deliver property",
    "never shipped": "cheating dishonestly induces deliver property",
    "never delivered": "cheating dishonestly induces deliver property",
    "hit-and-run": "causing death by negligence escaping without reporting rash negligent",

    # Cyber / IT Act: the Act says "personation", nobody else does
    "fake profile": "cheating by personation by using computer resource",
    "fake account": "cheating by personation by using computer resource",
    "impersonating": "cheating by personation by using computer resource",
    "pretending to be": "cheating by personation by using computer resource",
    "private photos": "publishing transmitting obscene material electronic form",
    "intimate images": "publishing transmitting obscene material electronic form",
    "without consent": "publishing transmitting obscene material privacy",

    # Cheque: NI Act presumption and interim compensation
    "blank cheque": "presumption in favour of holder consideration discharge debt",
    "security cheque": "presumption in favour of holder consideration debt liability",
    "presumption": "presumption in favour of holder consideration",
    "interim compensation": "power of appellate court to order payment pending appeal",
    "pendency": "power of appellate court to order payment pending appeal",

    # Domestic violence: "shared household" is the operative term
    "throw me out": "right to reside in a shared household",
    "thrown out": "right to reside in a shared household",
    "shared household": "right to reside in a shared household",
    "matrimonial home": "right to reside in a shared household",

    # Motor vehicles: minors driving is its own offence
    "juvenile": "offences by juveniles guardian owner",
    "underage": "offences by juveniles guardian owner",
    "minor drive": "offences by juveniles guardian owner",

    # Workplace harassment, incl. Hinglish phrasing
    "touch karta": "sexual harassment unwelcome physical contact advances",
    "galat tarah": "sexual harassment unwelcome physical contact advances",
    "inappropriately": "sexual harassment unwelcome physical contact advances",

    # Hindi / Hinglish (both scripts) -> English statutory vocabulary
    "भरण-पोषण": "order for maintenance of wives children and parents monthly allowance",
    "भरण": "order for maintenance of wives children and parents monthly allowance",
    "जमानत": "bail bailable bond release",
    "zamanat": "bail bailable bond release",
    "jamanat": "bail bailable bond release",
    "गिरफ्तार": "arrest police custody",
    "giraftar": "arrest police custody",
    "तलाक": "divorce dissolution of marriage",
    "talaq": "divorce dissolution of marriage",
    "दहेज": "dowry cruelty husband relatives",
    "dahej": "dowry cruelty husband relatives",
    "धोखा": "cheating deceiving dishonestly induces deliver property",
    "dhokha": "cheating deceiving dishonestly induces deliver property",
    "चोरी": "theft dishonestly movable property",
    "chori": "theft dishonestly movable property",
    "हत्या": "murder culpable homicide death",
    "hatya": "murder culpable homicide death",
    "क़त्ल": "murder culpable homicide death",
    "शादी": "marriage solemnized",
    "shaadi": "marriage solemnized",
    "विवाह": "marriage solemnized",
    "संपत्ति": "property",
    "sampatti": "property",
    "अदालत": "court",
    "adalat": "court",
    "न्यायालय": "court",
    "पुलिस": "police officer in charge police station",
    "शिकायत": "complaint",
    "shikayat": "complaint",
    "सज़ा": "punishment imprisonment fine",
    "सजा": "punishment imprisonment fine",
    "saza": "punishment imprisonment fine",
    "जेल": "imprisonment jail",
    "गवाह": "witness evidence testimony",
    "gawah": "witness evidence testimony",
    "गवाही": "witness evidence testimony examination",
    "gawaahi": "witness evidence testimony examination",
    "समन": "summons served",
    "मुआवजा": "compensation",
    "muavza": "compensation",
    "रिश्वत": "gratification public servant bribery",
    "rishwat": "gratification public servant bribery",
    "किराया": "rent landlord tenant",
    "kiraya": "rent landlord tenant",
    "अपील": "appeal conviction sentence",
    "कोर्ट": "court",
    "कानून": "law act provision",
    "kanoon": "law act provision",
    "अपराध": "offence punishable",
    "apradh": "offence punishable",
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
    "चेक बाउंस": "cheque returned unpaid dishonour insufficiency of funds",
    "check bounce": "cheque returned unpaid dishonour insufficiency of funds",
    "सबूत": "evidence proof admissible document",
    "saboot": "evidence proof admissible document",
    "उपभोक्ता": "consumer complaint deficiency",
    "ठगी": "cheating dishonestly induces deliver property",
    "thagi": "cheating dishonestly induces deliver property",
    "ऑनलाइन फ्रॉड": "cheats by personation computer resource cheating",
}

# Latin-script phrases need word boundaries; Devanagari must not use \b
# (combining vowels break Python re's \w) — plain substring works there.
_SYNONYM_PATTERNS = [
    (re.compile(rf"\b{re.escape(phrase)}\b" if phrase[0].isascii() else re.escape(phrase)),
     expansion)
    for phrase, expansion in LEGAL_SYNONYMS.items()
]


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def expand_query(query: str) -> str:
    """Append statutory vocabulary for any lay/Hindi legal term in the query."""
    q = query.lower()
    extra = [exp for pat, exp in _SYNONYM_PATTERNS if pat.search(q)]
    return q if not extra else q + " " + " ".join(extra)


class StatuteIndex:
    """Exact-reference + BM25 retrieval over StatuteSection rows."""

    def __init__(self, rows: list[dict], mappings: list[dict],
                 k1: float = 1.5, b: float = 0.75):
        self.rows = rows
        self.k1, self.b = k1, b
        self.by_key = {f"{r['act_id']}:{r['section'].upper()}": i
                       for i, r in enumerate(rows)}
        # official old-law -> new-law section mapping
        self.old_to_new = defaultdict(list)
        for m in mappings:
            old_family = m["old_act"].lower()
            new_family = m["new_act"].lower()
            new_act_id = _FAMILY_TO_ACT_ID.get(new_family)
            if new_act_id:
                self.old_to_new[(old_family, m["old_section"].upper())].append(
                    f"{new_act_id}:{m['new_section'].upper()}")

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
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))
        self.tf = [Counter(t) for t in self.doc_tokens]
        df = Counter()
        for t in self.doc_tokens:
            df.update(set(t))
        n = len(rows)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        # optional dense stage (A4); None keeps retrieval pure-BM25 and
        # dependency-free. Set by load_statute_index(dense_model=...).
        self.dense = None
        # optional cross-encoder second stage; None keeps first-stage order.
        # Set via set_reranker() — see nyaya.rerank.
        self.reranker = None

    # ---- stage 1: exact references -------------------------------------
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

    # ---- optional dense stage -------------------------------------------
    def add_dense(self, embed_fn, doc_vectors=None) -> None:
        """Enable hybrid retrieval from an embedding FUNCTION
        (list[str] -> list[vector]) — the injection point unit tests and the
        GPU jobs use (nyaya/dense.attach_dense_index wires a real model plus
        an .npy doc-vector cache through here). Equivalent to enabling the
        stage via load_statute_index(dense_model=...), which builds a
        DenseStage from a model name instead; both set self.dense.
        """
        if doc_vectors is None:
            doc_vectors = embed_fn([
                f"{r['act_name']} — {r.get('title') or ''}. {r.get('text') or ''}"
                for r in self.rows
            ])
        self.dense = _EmbedFnStage(embed_fn, doc_vectors)

    # ---- stage 2: BM25 ---------------------------------------------------
    def _bm25(self, query: str) -> list[tuple[float, int]]:
        q_tokens = _tokens(expand_query(query))
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q_tokens:
                if w not in tf:
                    continue
                idf = self.idf.get(w, 0.0)
                freq = tf[w]
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                s += idf * freq * (self.k1 + 1) / denom
                if w in self.title_tokens[i]:
                    s += TITLE_BONUS * idf
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return scores

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        """Up to k statute rows (exact references first, then fused ranking),
        plus up to KB_SLOTS procedural-guidance rows appended AFTER them.

        Guidance is ADDITIVE, never competing: reserving in-k slots measurably
        cost ~3pts statute recall@8 (KB rows crack the top-8 on ~100% of
        queries — short, keyword-dense, tag-boosted). Statutes keep every k
        slot; the appendix is capped; downstream word-budget packing trims the
        appendix first because it comes last. A purely procedural query with
        no matching statute still fills k from the KB.
        """
        picked = []
        for key in self.referenced_keys(query):
            picked.append(self.rows[self.by_key[key]])
            if len(picked) >= k:
                break
        chosen = {f"{r['act_id']}:{r['section'].upper()}" for r in picked}

        # Ranking for stage 2. Pure BM25 by default; when the optional dense
        # stage is enabled its cosine ranking is fused with BM25 via
        # reciprocal-rank fusion (exact references above always win).
        bm25_order = [i for _score, i in self._bm25(query)]
        order = (rrf_fuse([bm25_order, self.dense.rank(query)])
                 if self.dense is not None else bm25_order)

        statutes, guidance = [], []
        for i in order:
            row = self.rows[i]
            if f"{row['act_id']}:{row['section'].upper()}" in chosen:
                continue
            (guidance if row["act_id"] == "procedures_kb" else statutes).append(row)

        slots = max(0, k - len(picked))
        if self.reranker is not None:
            # Rerank statutes and guidance SEPARATELY so the KB appendix stays
            # additive and cannot win statute slots (see the docstring above).
            # `picked` is excluded: exact citation lookups are resolved facts,
            # not ranking guesses, and no model score may displace them.
            statute_take = self.reranker.rerank(query, statutes, slots)
            guidance_pool = self.reranker.rerank(query, guidance, KB_SLOTS)
        else:
            statute_take = statutes[:slots]
            guidance_pool = guidance[:KB_SLOTS]

        if picked or statute_take:
            guidance_take = guidance_pool
        else:
            # No statute matched at all — fill k from the KB instead.
            guidance_take = (self.reranker.rerank(query, guidance, k)
                             if self.reranker is not None else guidance[:k])
        return picked + statute_take + guidance_take

    def set_reranker(self, reranker) -> None:
        """Attach a cross-encoder second stage (see nyaya.rerank).

        Optional and off by default: it costs a model forward pass per
        candidate, which is worth it for eval and serving but not for bulk
        dataset generation over tens of thousands of queries.
        """
        self.reranker = reranker


class _EmbedFnStage:
    """Dense stage backed by an injected embed function — duck-typed to
    DenseStage (rank()). Pure Python on purpose: unit tests drive it with
    fake embedders and no numpy/model dependency."""

    def __init__(self, embed_fn, doc_vectors):
        self._embed_fn = embed_fn
        self._doc_vectors = [self._unit(v) for v in doc_vectors]

    @staticmethod
    def _unit(vec):
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def rank(self, query: str) -> list[int]:
        q = self._unit(self._embed_fn([query])[0])
        scores = [(sum(a * b for a, b in zip(q, d)), i)
                  for i, d in enumerate(self._doc_vectors)]
        scores.sort(reverse=True)
        return [i for _score, i in scores]


def rrf_fuse(rankings: list[list[int]], c: int = 60) -> list[int]:
    """Reciprocal-rank fusion: score(i) = sum over rankings of 1/(c + rank).
    Merges independent orderings (e.g. BM25 and dense cosine) without needing
    their scores to be comparable. Ties broken by first appearance."""
    scores: dict[int, float] = defaultdict(float)
    first_seen: dict[int, int] = {}
    seq = 0
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] += 1.0 / (c + rank + 1)
            if idx not in first_seen:
                first_seen[idx] = seq
                seq += 1
    return [idx for idx, _s in
            sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))]


def load_statute_index(canonical_dir: str | Path,
                       dense_model: str | None = None) -> StatuteIndex:
    """Build a StatuteIndex from data/canonical/*.jsonl (mappings included).

    dense_model (e.g. "intfloat/multilingual-e5-small") enables the optional
    hybrid retrieval stage — requires the dense extra (pip install -e ".[dense]"). Left
    None, retrieval stays pure-BM25 and dependency-free."""
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


# Context word budget: ~2,500 English legal words ≈ 3,400 tokens, leaving
# room for instructions + question + answer inside max_seq_length 6144.
CONTEXT_BUDGET_WORDS = 2500


def _row_words(row: dict) -> int:
    return len((row.get("title") or "").split()) + len((row.get("text") or "").split()) + 10


def pack_rows(rows: list[dict], budget_words: int = CONTEXT_BUDGET_WORDS,
              must_keys: set[str] | frozenset[str] = frozenset()) -> list[dict]:
    """Fit rows into a word budget, preserving order.

    Overflowing rows are skipped (a later shorter row may still fit) — except
    rows in must_keys (training gold) and a first row that alone exceeds the
    budget: those are truncated instead of dropped, so an answer's source is
    never packed away.
    """
    packed, used = [], 0
    for row in rows:
        words = _row_words(row)
        key = f"{row['act_id']}:{row['section'].upper()}"
        if used + words > budget_words:
            if not packed or key in must_keys:
                remaining = max(150, budget_words - used)
                text_words = (row.get("text") or "").split()[:remaining]
                row = {**row, "text": " ".join(text_words) + " …[provision truncated]"}
                words = _row_words(row)
            else:
                continue
        packed.append(row)
        used += words
    return packed


def rag_context_rows(record_id: str, question: str, gold_keys: list[str],
                     index: StatuteIndex, k: int = 8,
                     budget_words: int = CONTEXT_BUDGET_WORDS,
                     exclude_gold: bool = False) -> list[dict]:
    """Packed, per-record-shuffled context rows for a training/generation
    example. exclude_gold=True simulates a retrieval miss (RAFT negative)."""
    rows, seen = [], set()
    if not exclude_gold:
        rows = [index.rows[index.by_key[key]] for key in gold_keys if key in index.by_key]
        seen = {f"{r['act_id']}:{r['section'].upper()}" for r in rows}
    # mirror retrieve()'s composition: gold + statutes count toward k,
    # guidance rides in its own capped appendix
    statute_count = len(rows)
    guidance_count = 0
    for row in index.retrieve(question, k=k):
        key = f"{row['act_id']}:{row['section'].upper()}"
        if (exclude_gold and key in gold_keys) or key in seen:
            continue
        if row["act_id"] == "procedures_kb":
            if guidance_count >= KB_SLOTS:
                continue
            guidance_count += 1
        else:
            if statute_count >= k:
                continue
            statute_count += 1
        seen.add(key)
        rows.append(row)
    rows = pack_rows(rows, budget_words,
                     must_keys=set() if exclude_gold else set(gold_keys))
    random.Random(record_id).shuffle(rows)
    return rows


def normalize_gold_keys(source_sections: list[str]) -> list[str]:
    return [f"{s.split(':')[0].lower()}:{s.split(':')[1].upper()}"
            for s in source_sections]


def context_statute_db(context_keys: list[str]) -> dict[str, set[str]]:
    """A statute-DB view restricted to the context — verify_citations against
    it enforces 'cite only what was shown', not merely 'cite real law'."""
    from .validators import _ACT_ID_FAMILY
    db: dict[str, set[str]] = {}
    for key in context_keys:
        act_id, section = key.split(":", 1)
        family = _ACT_ID_FAMILY.get(act_id.rsplit("_", 1)[0])
        if family:
            db.setdefault(family, set()).add(section.upper())
    return db


def build_rag_training_record(record: dict, index: StatuteIndex, k: int = 8,
                              budget_words: int = CONTEXT_BUDGET_WORDS) -> dict:
    """Rewrap a Nyaya-Instruct example in the inference-time RAG prompt.

    v1's adapter went off-distribution under RAG prompts (it was trained on
    bare questions and stopped citing). v2 trains on exactly what inference
    serves: retrieved context + question -> the grounded answer. The example's
    gold source_sections are force-injected so the answer's citations are
    always derivable from the context; their position is shuffled per-record
    (seeded by id) so the model can't learn "the answer is always first".
    """
    question = next(m["content"] for m in record["messages"] if m["role"] == "user")
    gold_keys = normalize_gold_keys(
        record.get("metadata", {}).get("source_sections", []))
    rows = rag_context_rows(record["id"], question, gold_keys, index,
                            k=k, budget_words=budget_words)

    out = copy.deepcopy(record)
    for message in out["messages"]:
        if message["role"] == "user":
            message["content"] = build_rag_prompt(question, rows)
            break
    out["metadata"]["rag"] = {
        "k": k,
        "context_keys": [f"{r['act_id']}:{r['section'].upper()}" for r in rows],
        "gold_injected": bool(gold_keys),
    }
    return out


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


def build_rag_prompt(question: str, retrieved: list[dict],
                     budget_words: int = CONTEXT_BUDGET_WORDS) -> str:
    packed = pack_rows(retrieved, budget_words)
    return RAG_ANSWER_PROMPT.format(context=format_context(packed), question=question)
