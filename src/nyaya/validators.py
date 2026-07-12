"""Validation pipeline pieces: citation verification, dedup, leakage detection.

Citation verification is deterministic and non-negotiable: if any cited section
does not resolve against the statute DB, drop the whole example.
"""

import re

# Matches "Section 318", "Sec. 173", "§420", "dhara 154", etc.
CITATION_PATTERN = re.compile(r"(Section|Sec\.?|§|dhara)\s+[\dA-Z()]+", re.IGNORECASE)


def extract_citations(text: str) -> list[str]:
    """Regex-extract every section citation + act mention from an answer."""
    return [m.group(0) for m in CITATION_PATTERN.finditer(text)]


def verify_citations(text: str, statute_db: dict) -> bool:
    """True only if EVERY citation in `text` resolves against the statute DB.

    TODO: implement act-aware resolution (map citation -> (act_id, section) ->
    lookup in statute DB). Any unresolved citation -> reject the whole example.
    """
    raise NotImplementedError


def is_near_duplicate(a: str, b: str, threshold: float = 0.92) -> bool:
    """Near-duplicate check — MinHash or embedding cosine similarity > threshold.

    TODO: implement (datasketch MinHash or sentence-transformers).
    """
    raise NotImplementedError


def detect_eval_leakage(example: dict, eval_records: list[dict]) -> bool:
    """True if a training example overlaps Nyaya-Eval-v0 (question or answer similarity).

    TODO: implement — run on every dataset version before training.
    """
    raise NotImplementedError
