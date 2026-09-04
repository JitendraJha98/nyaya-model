"""Rewrite a Hindi / Hinglish question into the English statutory vocabulary
the BM25 index is built on.

Why: 19 of 53 pure-Devanagari citizen questions retrieved zero statute sections
(reports/coverage_probe.json, Sept 2026). The index is English statute text
and the hand-written synonym table in retrieval.py cannot keep up. Any text
generator -- the 3B reader itself -- can translate a lay question into one line
of statutory English before retrieval. The original question is kept alongside
the rewrite, so exact citations and the Devanagari synonyms still match.

Usage:
    from nyaya.rewrite import rewrite_query
    query = rewrite_query(question, generate)      # generate: str -> str
    rows = index.retrieve(query, k=8)
"""
import re

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# Romanised Hindi function words. Two or more in a question is Hinglish.
_HINGLISH_MARKERS = re.compile(
    r"\b(kya|kaise|kaha|kahan|kab|kyu|kyun|nahi|nahin|hai|hain|tha|thi|karu|karun|karna|karne|"
    r"mera|meri|mere|mujhe|mujhko|hum|hamara|wapas|raha|rahi|rahe|kar|kiya|ho|hoga|hogi|gaya|gayi|"
    r"wala|wale|wali|bhai|bhaiya|thana|paisa|paise|saal|mahine|din|abhi|ab|toh|par|lekin|aur|ya|"
    r"bola|bol|diya|de|dena|lena|liya|chahiye|sakta|sakti|sakte|milega|milegi|jaye|jaun|jau)\b",
    re.IGNORECASE)

REWRITE_PROMPT = (
    "Rewrite the following Indian citizen's legal question as ONE LINE of plain English using the "
    "words an Indian statute would use (for example: 'cheating', 'dishonour of cheque', "
    "'information about a cognizable offence', 'maintenance', 'shared household', 'deficiency in "
    "service', 'rash and negligent driving'). Keep any section numbers and act names. Do not answer "
    "the question. Output only the rewritten line.\n\nQuestion: {question}\nRewritten:"
)


def needs_rewrite(question: str) -> bool:
    """Devanagari script, or two or more romanised-Hindi function words."""
    if _DEVANAGARI.search(question):
        return True
    return len(_HINGLISH_MARKERS.findall(question)) >= 2


def rewrite_query(question: str, generate) -> str:
    """Return the retrieval query for `question`.

    `generate(prompt: str) -> str` is any greedy text generator. English
    questions pass through untouched. For Hindi/Hinglish the first non-empty
    line of the generation is appended to the original question.
    """
    if not needs_rewrite(question):
        return question
    raw = generate(REWRITE_PROMPT.format(question=question)) or ""
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    rewritten = lines[0] if lines else ""
    rewritten = re.sub(r"^(rewritten|answer)\s*[:：]\s*", "", rewritten, flags=re.IGNORECASE).strip().strip('"')
    if not rewritten or len(rewritten) > 400:
        return question
    return f"{question} {rewritten}"
