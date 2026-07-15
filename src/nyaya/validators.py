"""Validation pipeline pieces: citation verification, dedup, leakage detection.

Citation verification is deterministic and non-negotiable: if any cited section
does not resolve against the statute DB (data/canonical/*.jsonl, built by
scripts/03_build_corpus.py), drop the whole example.
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = ROOT / "data" / "canonical"

# Act-name aliases, shared with the evaluation harness. Keys are act families;
# values are the normalized surface forms accepted in citations/answers —
# including Devanagari forms, since Hindi answers legitimately cite acts in
# Devanagari ("भारतीय न्याय संहिता की धारा 318").
ACT_ALIASES = {
    "bns": ["bns", "bharatiya nyaya sanhita", "भारतीय न्याय संहिता"],
    "bnss": ["bnss", "bharatiya nagarik suraksha sanhita", "भारतीय नागरिक सुरक्षा संहिता"],
    "bsa": ["bsa", "bharatiya sakshya adhiniyam", "भारतीय साक्ष्य अधिनियम"],
    "rti": ["rti", "right to information act", "सूचना का अधिकार"],
    "ipc": ["ipc", "indian penal code", "भारतीय दंड संहिता"],
    "crpc": ["crpc", "code of criminal procedure", "criminal procedure code",
             "दंड प्रक्रिया संहिता"],
    "iea": ["iea", "indian evidence act", "evidence act", "भारतीय साक्ष्य अधिनियम 1872"],
    "ni act": ["ni act", "negotiable instruments act", "परक्राम्य लिखत अधिनियम"],
    "it act": ["it act", "information technology act", "सूचना प्रौद्योगिकी अधिनियम"],
    "mv act": ["mv act", "motor vehicles act", "motor vehicle act", "मोटर यान अधिनियम"],
    "cpa": ["cpa", "consumer protection act", "उपभोक्ता संरक्षण अधिनियम"],
    "dv act": ["dv act", "domestic violence act",
               "protection of women from domestic violence act", "घरेलू हिंसा अधिनियम"],
    "posh": ["posh", "posh act", "sexual harassment of women at workplace",
             "कार्यस्थल पर महिलाओं का लैंगिक उत्पीड़न"],
    "hma": ["hma", "hindu marriage act", "हिंदू विवाह अधिनियम"],
    "sma": ["sma", "special marriage act", "विशेष विवाह अधिनियम"],
    "wages code": ["wages code", "code on wages", "वेतन संहिता"],
    "constitution": ["constitution", "constitution of india", "संविधान"],
}

# act_id prefixes in data/canonical -> act family key above
_ACT_ID_FAMILY = {
    "bns": "bns",
    "bnss": "bnss",
    "bsa": "bsa",
    "rti": "rti",
    "cpa": "cpa",
    "it_act": "it act",
    "ni_act": "ni act",
    "mv_act": "mv act",
    "dv_act": "dv act",
    "posh": "posh",
    "hma": "hma",
    "sma": "sma",
    "wages_code": "wages code",
    "constitution": "constitution",
}

# Matches "Section 318", "Sections 34", "Sec. 173", "§420", "§ 420", "dhara 154",
# "धारा 154", "Article 21", "अनुच्छेद 21", "Section 66A", "Section 318(4)".
# The token after the marker must start with a digit so prose like "this
# section is important" never matches, and must end at a word boundary so
# glued shorthand like "34IPC" is cleanly skipped rather than half-captured
# as a plausible-but-wrong "34IP".
CITATION_PATTERN = re.compile(
    r"(?:\b(?:Sections?|Sec\.?|Articles?|Art\.?|dhara|धारा|अनुच्छेद)\s+|§\s*)"
    r"\d+[A-Za-z]{0,2}\b(?:\(\w+\))*",
    re.IGNORECASE,
)
_SECTION_NUMBER = re.compile(r"(\d+[A-Za-z]{0,2})")


def extract_citations(text: str) -> list[str]:
    """Regex-extract every section-number citation from an answer.

    Extraction only — resolving each citation to its act (and expanding
    enumerations like "Sections 34 and 120B") is verify_citations' job.
    """
    return [m.group(0) for m in CITATION_PATTERN.finditer(text)]


def load_statute_db(
    canonical_dir: str | Path | None = None, include_old_law: bool = False
) -> dict[str, set[str]]:
    """Load data/canonical/*.jsonl into {act_family: {section numbers}}.

    include_old_law=True additionally whitelists repealed-act sections
    (IPC/CrPC/IEA) listed in law_mappings.jsonl, so historical references
    like "IPC 420 was replaced by BNS 318" verify. Default excludes them:
    grounded generation should cite current law only.
    """
    directory = Path(canonical_dir) if canonical_dir else CANONICAL_DIR
    db: dict[str, set[str]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        if path.name.startswith("law_mappings"):
            continue  # handled below
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                act_id = row["act_id"]  # e.g. "bns_2023"
                family = _ACT_ID_FAMILY.get(act_id.rsplit("_", 1)[0])
                if family:
                    db.setdefault(family, set()).add(row["section"].upper())
    if include_old_law:
        mapping_file = directory / "law_mappings.jsonl"
        if mapping_file.exists():
            with mapping_file.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    family = row["old_act"].lower()
                    db.setdefault(family, set()).add(row["old_section"].upper())
    if not db:
        raise FileNotFoundError(
            f"no statute JSONL found in {directory} — run scripts/03_build_corpus.py"
        )
    return db


def alias_pattern(variant: str) -> str:
    """Regex for an act alias. Latin aliases get \\b guards; Devanagari ones
    must not — Python re's \\b misfires next to combining vowel signs (which
    are not \\w), so a \\b-wrapped "संहिता" can never match."""
    if variant.isascii():
        return rf"\b{re.escape(variant)}\b"
    return re.escape(variant)


def _acts_in(text_norm: str) -> list[tuple[int, str]]:
    """(position, family) for every act alias mention in normalized text."""
    hits = []
    for family, variants in ACT_ALIASES.items():
        for variant in variants:
            for m in re.finditer(alias_pattern(variant), text_norm):
                hits.append((m.start(), family))
    return sorted(hits)


def resolve_citations(text: str, statute_db: dict, window: int = 140) -> list[dict]:
    """Resolve each extracted citation to (act, section) against the DB.

    Act attribution: the nearest act-alias mention within `window` characters
    of the citation. Fallback: legal prose commonly names the act once and then
    cites bare sections ("…the DV Act, 2005 protects you. … Under Section 19…"),
    so when the WHOLE text mentions exactly one act family, bare citations
    attribute to it — hallucinated numbers still fail the section lookup, and
    multi-act texts stay strict. A citation with no attributable act, an act
    family absent from the DB, or a section the act does not contain is
    unresolved.
    """
    text_norm = text.lower()
    act_positions = _acts_in(text_norm)
    families_in_text = {family for _pos, family in act_positions}
    sole_act = next(iter(families_in_text)) if len(families_in_text) == 1 else None
    results = []
    for m in CITATION_PATTERN.finditer(text):
        citation = m.group(0)
        number_match = _SECTION_NUMBER.search(citation)
        section = number_match.group(1).upper() if number_match else None
        in_range = [
            # distance from the citation span (0 when the alias sits inside it)
            (max(m.start() - a_pos, a_pos - m.end(), 0), a_pos, family)
            for a_pos, family in act_positions
            if m.start() - window <= a_pos <= m.end() + window
        ]
        act = min(in_range)[2] if in_range else sole_act
        resolved = bool(
            act and section and act in statute_db and section in statute_db[act]
        )
        results.append(
            {"citation": citation, "section": section, "act": act, "resolved": resolved}
        )
    return results


def verify_citations(text: str, statute_db: dict) -> bool:
    """True only if EVERY citation in `text` resolves against the statute DB.

    A text with no citations passes vacuously (whether citation-less answers
    are acceptable is the caller's policy, not this gate's). Any unresolved
    citation -> reject the whole example.
    """
    return all(r["resolved"] for r in resolve_citations(text, statute_db))


def _norm_for_similarity(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def is_near_duplicate(a: str, b: str, threshold: float = 0.92) -> bool:
    """Near-duplicate check: normalized similarity ratio > threshold.

    Deterministic stdlib implementation (token-Jaccard prefilter +
    SequenceMatcher); swap in MinHash/embeddings later if scale demands."""
    na, nb = _norm_for_similarity(a), _norm_for_similarity(b)
    if na == nb:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return False
    jaccard = len(ta & tb) / len(ta | tb)
    if jaccard < 0.35:  # cheap prefilter — cannot be near-duplicate
        return False
    return SequenceMatcher(None, na, nb).ratio() > threshold


_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_GROUNDED_TASK_TYPES = {
    "grounded_qa", "hindi_qa", "hinglish_qa", "terminology", "procedural",
    "law_mapping", "extraction_qa",
}
_REQUIRED_METADATA = ("language", "task_type", "source_sections", "dataset_version")
# The 80-word plan floor was calibrated for essay-style grounded QA; mapping
# ("IPC 420 is now BNS 318…") and terminology answers are legitimately
# compact, and refusals are legitimately brief. Pilot evidence (2026-07-15):
# 31/47 length rejections were good mapping answers.
MIN_WORDS_BY_TASK = {
    "law_mapping": 20,
    "terminology": 50,
    "safety_abstention": 1,
    "extraction_qa": 15,
}


def validate_example(
    example: dict,
    statute_db: dict,
    eval_records: list[dict],
    min_words: int = 80,
    max_words: int = 600,
) -> tuple[bool, list[str]]:
    """The per-example gate for scripts/05 (docs/ROADMAP.md Step 8).

    Checks, in order: schema -> source_sections for grounded tasks ->
    citation verification (deterministic, non-negotiable) -> language/script
    consistency -> answer length (safety/abstention answers are exempt from
    the minimum — refusals are legitimately brief) -> eval-leakage.
    Returns (ok, reasons); every failed check is reported, not just the first.
    """
    reasons: list[str] = []

    messages = example.get("messages") or []
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if roles[:3] != ["system", "user", "assistant"] or not all(
        isinstance(m.get("content"), str) and m["content"].strip() for m in messages[:3]
    ):
        return False, ["schema: messages must be non-empty [system, user, assistant]"]
    metadata = example.get("metadata") or {}
    missing = [k for k in _REQUIRED_METADATA if k not in metadata]
    if missing or not example.get("id"):
        return False, [f"schema: missing {', '.join(missing) or 'id'}"]

    task_type = metadata["task_type"]
    question = messages[1]["content"]
    answer = messages[2]["content"]

    if task_type in _GROUNDED_TASK_TYPES and not metadata["source_sections"]:
        reasons.append("source_sections: grounded task with no split key")

    if not verify_citations(answer, statute_db):
        reasons.append("citation: unresolved citation against the statute DB")

    language = metadata["language"]
    if language == "hindi" and not _DEVANAGARI.search(answer):
        reasons.append("language: marked hindi but answer has no Devanagari")
    elif language == "hinglish" and _DEVANAGARI.search(question):
        reasons.append("language: marked hinglish but question uses Devanagari")

    words = len(answer.split())
    lower = MIN_WORDS_BY_TASK.get(task_type, min_words)
    if not (lower <= words <= max_words):
        reasons.append(f"length: {words} words outside [{lower}, {max_words}]")

    if detect_eval_leakage(example, eval_records):
        reasons.append("leakage: overlaps a Nyaya-Eval question/answer")

    return (not reasons), reasons


def detect_eval_leakage(
    example: dict, eval_records: list[dict], threshold: float = 0.85
) -> bool:
    """True if a training example overlaps Nyaya-Eval-v0 (question or answer
    similarity above threshold). Run on every dataset version before training."""
    user_texts = [m["content"] for m in example["messages"] if m["role"] == "user"]
    assistant_texts = [m["content"] for m in example["messages"] if m["role"] == "assistant"]
    for record in eval_records:
        for text in user_texts:
            if is_near_duplicate(text, record["question"], threshold):
                return True
        for text in assistant_texts:
            if is_near_duplicate(text, record["expected_answer"], threshold):
                return True
    return False
