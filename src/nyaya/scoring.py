"""Nyaya-Eval-v1 scoring — strict citations, partial-credit substance.

Why this exists
---------------
`evaluation.auto_strict_correct` (Eval-v0) is a conjunction over *every*
required fact, and ~85% of required facts are free-text phrases matched as
normalized substrings. Legally correct answers therefore fail routinely:

    required: "death or imprisonment for life"
    answer:   "imprisonment for life or death, plus fine"   -> v0 FAIL (word order)

    required: "five or more persons"
    answer:   "a group of 5 or more people"                 -> v0 FAIL (people/persons)

    required: "up to 3 years"
    answer:   "which may extend to three years"             -> v0 FAIL (phrasing)

    required: "ground of race, caste or community"
    answer:   "on grounds of race, caste, community"        -> v0 FAIL (ground/grounds)

Because of that, base / v3 / v4 land within a 2-answer spread of one another
(3.84% / 4.24% / 3.84% at n=495) — the metric cannot distinguish a good model
from a bad one, so it cannot be used to show that training helped.

What v1 changes
---------------
Two tracks, deliberately graded differently:

* CITATION track — stays strict, no partial credit. A legal model that cites
  the wrong section is simply wrong, and "close" section numbers are worthless.
  Reuses the Eval-v0 citation-context rule (a bare number in prose is not a
  citation) so citation numbers stay comparable across v0 and v1.

* SUBSTANCE track — per-fact partial credit with paraphrase tolerance:
  token-set containment (order-free), light stemming, numeral and legal-synonym
  normalization.

Forbidden facts (e.g. citing IPC 302 as *current* law) remain a hard fail and
zero the record — staleness is the one error class this project cannot tolerate.

Guard rails on the leniency
---------------------------
Partial credit is only as honest as its floor. Two rules keep it from turning
into free marks:

1. NUMERIC GUARD — every number in a fact must appear in the answer. A response
   that says "years" while missing the "7" scores 0 on "minimum 7 years". This
   is what stops "up to 10 years" from being satisfied by "up to 3 years".
2. OVERLAP FLOOR — below `PARTIAL_FLOOR` token overlap the fact scores 0, so
   incidental word collisions earn nothing.

The headline v1 metric is `fact_recall` (mean fraction of required facts
expressed). `all_facts` is retained as the v0-equivalent conjunction so the two
generations of results stay comparable in the same report.
"""

import re

from .evaluation import _SECTION_FACT, fact_present

# Below this token-overlap ratio a substance fact scores nothing.
PARTIAL_FLOOR = 0.6

# Devanagari-safe punctuation strip: Python's \w does not match Devanagari
# combining vowel signs, so [^\w\s] would shred Hindi. Same set as
# evaluation._PUNCT, kept local so scoring stays self-contained.
_PUNCT = re.compile(r"[.,;:!?\"'’‘“”`\-–—_/\\|।॥*#%&+=<>\[\]{}()]")

_STOPWORDS = frozenset(
    "the a an of to for in on at is are was were be been being and or with under "
    "by from as that this it its their his her they he she which who whom shall "
    "may can will would could should must have has had do does did not no any "
    "such other than then there here also more most only own same so up out"
    .split()
)

# Word numerals -> digits. Covers the compounds Eval-v0 explicitly gave up on
# ("twenty five"), handled by the multi-word pass in _fold_numerals.
_WORD_NUMERALS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_SCALES = {"thousand": 1_000, "lakh": 100_000, "lakhs": 100_000,
           "crore": 10_000_000, "crores": 10_000_000, "million": 1_000_000}

# Legal-register synonyms, mapped onto a canonical stem AFTER stemming.
# Deliberately small and domain-specific: a broad thesaurus would start
# matching facts the model never actually expressed.
_SYNONYMS = {
    "people": "person", "individual": "person", "citizen": "person",
    "human": "person", "member": "person",
    "jail": "imprisonment", "incarceration": "imprisonment",
    "imprison": "imprisonment", "prison": "imprisonment",
    "custodi": "custody",
    "penalti": "fine", "penalty": "fine", "monetari": "fine",
    "punish": "punishment", "punishabl": "punishment", "sentenc": "punishment",
    "offenc": "offence", "offens": "offence", "crime": "offence",
    "criminal": "offence",
    "advocat": "lawyer", "counsel": "lawyer", "attorney": "lawyer",
    "polic": "police", "cop": "police",
    "complain": "complaint", "grievanc": "complaint",
    "provis": "provision", "clause": "provision",
    "ground": "ground", "basi": "ground", "reason": "ground",
    "distinct": "separate", "specif": "separate", "standalon": "separate",
    "extend": "upto", "maximum": "upto", "max": "upto",
    "minimum": "atleast", "min": "atleast", "least": "atleast",
    "lifelong": "life", "lifetim": "life",
    "deceas": "death", "die": "death", "dead": "death", "kill": "death",
    "marriag": "marriage", "wed": "marriage",
    "year": "year", "yr": "year",
}


def _fold_numerals(text: str) -> str:
    """Turn spelled-out numbers into digits, including scaled compounds.

    "seven"        -> "7"
    "twenty five"  -> "25"
    "five thousand"-> "5000"
    "2 lakh"       -> "200000"
    """
    tokens = text.split()
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        value = _WORD_NUMERALS.get(tok)
        if value is None and tok.isdigit():
            value = int(tok)
        if value is None:
            out.append(tok)
            i += 1
            continue

        # "twenty five" -> 25 (tens followed by a unit)
        if (value >= 20 and value < 100 and value % 10 == 0
                and i + 1 < len(tokens)):
            unit = _WORD_NUMERALS.get(tokens[i + 1])
            if unit is not None and 1 <= unit <= 9:
                value += unit
                i += 1

        # "five thousand" / "2 lakh" -> scale it
        if i + 1 < len(tokens) and tokens[i + 1] in _SCALES:
            value *= _SCALES[tokens[i + 1]]
            i += 1

        out.append(str(value))
        i += 1
    return " ".join(out)


def _stem(token: str) -> str:
    """Crude suffix stripping, ASCII only.

    Devanagari is returned untouched: naive suffix rules would corrupt Hindi
    morphology, and Hindi facts are matched on their own surface forms.
    """
    if not token.isascii() or token.isdigit() or len(token) <= 3:
        return token
    for suffix, keep in (("ies", "y"), ("ing", ""), ("edly", ""), ("ed", ""),
                         ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "s" and token.endswith("ss"):
                break
            return token[: len(token) - len(suffix)] + keep
    return token


def normalize_tokens(text: str) -> list[str]:
    """Full normalization pipeline -> comparable token list."""
    text = text.lower().replace("§", " section ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)          # 10,000 -> 10000
    text = re.sub(r"\brs\.?\b", " rupees ", text)
    text = _PUNCT.sub(" ", text)
    text = _fold_numerals(re.sub(r"\s+", " ", text).strip())
    tokens = []
    for raw in text.split():
        if raw in _STOPWORDS:
            continue
        stem = _stem(raw)
        tokens.append(_SYNONYMS.get(stem, stem))
    return tokens


def _numbers(tokens) -> set[str]:
    return {t for t in tokens if t.isdigit()}


def is_citation_fact(fact: str) -> bool:
    """True if the fact asserts a specific section/article reference."""
    return bool(_SECTION_FACT.search(fact))


def score_fact(fact: str, response: str) -> dict:
    """Score one required fact in [0, 1] with a reason for the score.

    Citation facts are all-or-nothing. Substance facts get partial credit,
    subject to the numeric guard and the overlap floor.
    """
    if is_citation_fact(fact):
        ok = fact_present(fact, response)
        return {"fact": fact, "kind": "citation", "score": 1.0 if ok else 0.0,
                "reason": "citation matched" if ok else "citation missing/wrong"}

    fact_tokens = normalize_tokens(fact)
    resp_tokens = set(normalize_tokens(response))
    if not fact_tokens:
        return {"fact": fact, "kind": "substance", "score": 0.0,
                "reason": "fact normalized to nothing (unmatchable — curate it)"}

    # Numeric guard: a fact's numbers are load-bearing. "up to 10 years" must
    # not be satisfied by an answer that only says "up to 3 years".
    missing_numbers = _numbers(fact_tokens) - _numbers(resp_tokens)
    if missing_numbers:
        return {"fact": fact, "kind": "substance", "score": 0.0,
                "reason": f"missing number(s): {sorted(missing_numbers)}"}

    unique = list(dict.fromkeys(fact_tokens))
    hits = sum(1 for t in unique if t in resp_tokens)
    overlap = hits / len(unique)

    if overlap == 1.0:
        return {"fact": fact, "kind": "substance", "score": 1.0,
                "reason": "all content tokens present"}
    if overlap >= PARTIAL_FLOOR:
        return {"fact": fact, "kind": "substance", "score": round(overlap, 4),
                "reason": f"partial: {hits}/{len(unique)} content tokens"}
    return {"fact": fact, "kind": "substance", "score": 0.0,
            "reason": f"below floor: {hits}/{len(unique)} content tokens"}


def score_record(record: dict, response: str) -> dict:
    """Score one eval record. Forbidden facts zero everything."""
    facts = [score_fact(f, response) for f in record.get("required_facts", [])]
    violated = [f for f in record.get("forbidden_facts", [])
                if fact_present(f, response)]

    citations = [f for f in facts if f["kind"] == "citation"]
    substance = [f for f in facts if f["kind"] == "substance"]

    def mean(rows):
        return round(sum(r["score"] for r in rows) / len(rows), 4) if rows else None

    recall = mean(facts) or 0.0
    if violated:
        # Stale/forbidden law is a hard fail regardless of what else was right.
        recall = 0.0

    return {
        "id": record.get("id"),
        "language": record.get("language"),
        "legal_domain": record.get("legal_domain"),
        "task_type": record.get("task_type"),
        "difficulty": record.get("difficulty"),
        "is_safety_row": record.get("task_type") == "safety_abstention",
        "fact_scores": facts,
        "forbidden_violated": violated,
        "fact_recall": recall,
        "citation_recall": mean(citations),
        "substance_recall": mean(substance),
        # v0-equivalent conjunction, kept so old and new results stay comparable.
        "all_facts": bool(facts) and not violated and all(f["score"] == 1.0 for f in facts),
    }


def aggregate(scored: list[dict]) -> dict:
    """Aggregate scored records into the Eval-v1 metric block."""
    rows = [r for r in scored if not r["is_safety_row"]]
    safety = [r for r in scored if r["is_safety_row"]]
    n = len(rows)

    def mean(key, subset=None):
        vals = [r[key] for r in (subset or rows) if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def bucket(key):
        out = {}
        for r in rows:
            b = out.setdefault(r[key], {"n": 0, "fact_recall": 0.0, "all_facts": 0})
            b["n"] += 1
            b["fact_recall"] += r["fact_recall"]
            b["all_facts"] += int(r["all_facts"])
        for b in out.values():
            b["fact_recall"] = round(b["fact_recall"] / b["n"], 4)
            b["all_facts"] = round(b["all_facts"] / b["n"], 4)
        return out

    with_citations = [r for r in rows if r["citation_recall"] is not None]
    return {
        "scored_total": n,
        # Headline: mean fraction of required facts expressed.
        "fact_recall": mean("fact_recall"),
        "citation_accuracy": round(
            sum(r["citation_recall"] for r in with_citations) / len(with_citations), 4
        ) if with_citations else 0.0,
        "citation_rows": len(with_citations),
        "substance_recall": mean("substance_recall"),
        # v0-equivalent, for continuity with published numbers.
        "all_facts_accuracy": round(sum(r["all_facts"] for r in rows) / n, 4) if n else 0.0,
        "forbidden_violation_rate": round(
            sum(bool(r["forbidden_violated"]) for r in rows) / n, 4) if n else 0.0,
        "by_language": bucket("language"),
        "by_domain": bucket("legal_domain"),
        "by_difficulty": bucket("difficulty"),
        "by_task_type": bucket("task_type"),
        "safety_rows": len(safety),
    }


# --------------------------------------------------------------------------
# Fact linting — a scorer cannot rescue a fact that is not a phrase.
# --------------------------------------------------------------------------

# Facts phrased as propositions/instructions rather than quotable content:
# no answer wording reliably contains them, so they are noise in any
# string-matching metric and must be rewritten or dropped in Eval-v1.
_PROPOSITION_MARKERS = (
    "not retained", "not in force", "deferred", "is a distinct", "distinct offence",
    "introduced by", "option", "available", "applies", "still", "no longer",
    "must", "should", "can be", "is required",
)


def lint_fact(fact: str) -> list[str]:
    """Flag required facts that a string metric cannot fairly grade."""
    problems = []
    tokens = normalize_tokens(fact)
    if is_citation_fact(fact):
        return problems
    if not tokens:
        problems.append("empty-after-normalization")
    if len(tokens) > 6:
        problems.append("too-long: unlikely to appear verbatim; split it")
    lowered = fact.lower()
    if any(m in lowered for m in _PROPOSITION_MARKERS):
        problems.append("proposition: asserts a claim rather than quoting content")
    if len(tokens) == 1 and tokens[0].isdigit():
        problems.append("bare-number: ambiguous without units")
    return problems
