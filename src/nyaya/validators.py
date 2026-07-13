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
# values are the normalized surface forms accepted in citations/answers.
ACT_ALIASES = {
    "bns": ["bns", "bharatiya nyaya sanhita"],
    "bnss": ["bnss", "bharatiya nagarik suraksha sanhita"],
    "bsa": ["bsa", "bharatiya sakshya adhiniyam"],
    "rti": ["rti", "right to information act"],
    "ipc": ["ipc", "indian penal code"],
    "crpc": ["crpc", "code of criminal procedure", "criminal procedure code"],
    "iea": ["iea", "indian evidence act", "evidence act"],
    "ni act": ["ni act", "negotiable instruments act"],
    "it act": ["it act", "information technology act"],
    "mv act": ["mv act", "motor vehicles act", "motor vehicle act"],
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
}

# Matches "Section 318", "Sections 34", "Sec. 173", "§420", "§ 420", "dhara 154",
# "धारा 154", "Section 66A", "Section 318(4)". The token after the marker must
# start with a digit so prose like "this section is important" never matches,
# and must end at a word boundary so glued shorthand like "34IPC" is cleanly
# skipped rather than half-captured as a plausible-but-wrong "34IP".
CITATION_PATTERN = re.compile(
    r"(?:\b(?:Sections?|Sec\.?|dhara|धारा)\s+|§\s*)\d+[A-Za-z]{0,2}\b(?:\(\w+\))*",
    re.IGNORECASE,
)
_SECTION_NUMBER = re.compile(r"(\d+[A-Za-z]{0,2})")


def extract_citations(text: str) -> list[str]:
    """Regex-extract every section-number citation from an answer.

    Extraction only — resolving each citation to its act (and expanding
    enumerations like "Sections 34 and 120B") is verify_citations' job.
    """
    return [m.group(0) for m in CITATION_PATTERN.finditer(text)]


def load_statute_db(canonical_dir: str | Path | None = None) -> dict[str, set[str]]:
    """Load data/canonical/*.jsonl into {act_family: {section numbers}}."""
    directory = Path(canonical_dir) if canonical_dir else CANONICAL_DIR
    db: dict[str, set[str]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        if path.name.startswith("law_mappings"):
            continue  # mapping table has its own loader
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                act_id = row["act_id"]  # e.g. "bns_2023"
                family = _ACT_ID_FAMILY.get(act_id.rsplit("_", 1)[0])
                if family:
                    db.setdefault(family, set()).add(row["section"].upper())
    if not db:
        raise FileNotFoundError(
            f"no statute JSONL found in {directory} — run scripts/03_build_corpus.py"
        )
    return db


def _acts_in(text_norm: str) -> list[tuple[int, str]]:
    """(position, family) for every act alias mention in normalized text."""
    hits = []
    for family, variants in ACT_ALIASES.items():
        for variant in variants:
            for m in re.finditer(rf"\b{re.escape(variant)}\b", text_norm):
                hits.append((m.start(), family))
    return sorted(hits)


def resolve_citations(text: str, statute_db: dict, window: int = 140) -> list[dict]:
    """Resolve each extracted citation to (act, section) against the DB.

    Act attribution: the nearest act-alias mention within `window` characters
    of the citation (searching the lowercased text). A citation with no act in
    range, an act family absent from the DB, or a section the act does not
    contain is unresolved.
    """
    text_norm = text.lower()
    act_positions = _acts_in(text_norm)
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
        act = min(in_range)[2] if in_range else None
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
