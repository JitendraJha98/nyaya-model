"""Evaluation harness — build this BEFORE training run 1.

Layers (docs/ROADMAP.md "Evaluation plan"):
  1. Citation Accuracy Score (primary) — via validators.extract_citations
     + statute-DB verification. Until the statute DB exists (roadmap Step 6),
     this module computes what is honestly computable without it:
     required/forbidden-fact coverage, extracted citations, abstention flags
     and latency. DB-backed verification plugs into verify_citations later.
  2. Held-out-acts generalization.
  3. AIBE bar-exam MCQs.
  4. IL-TUR + BhashaBench-Legal deltas.
  5. Blind human eval (ship gate) + 50-question regression set.

`auto_strict_correct` is a deterministic proxy: every required fact present
and no forbidden fact present. Safety/abstention rows are excluded from the
strict aggregate (their grading is behavioural) and reported separately.
"""

import json
import re
import time
from pathlib import Path

from .dataset import load_jsonl
from .validators import ACT_ALIASES, alias_pattern, extract_citations

ROOT = Path(__file__).resolve().parents[2]
FROZEN_EVAL = ROOT / "data" / "eval" / "nyaya_eval_v0.jsonl"
DRAFT_EVAL = ROOT / "data" / "eval" / "nyaya_eval_v0_draft.jsonl"

_SECTION_FACT = re.compile(
    r"(?:section|sec\.?|§|dhara|धारा|अनुच्छेद|article|art\.?)\s*(\d+[A-Za-z]{0,2}(?:\(\w+\))*)",
    re.IGNORECASE,
)
_ABSTENTION_MARKERS = [
    "consult a licensed advocate", "consult a lawyer", "consult an advocate",
    "cannot help", "can't help", "cannot assist", "unable to assist",
    "i cannot guarantee", "cannot guarantee", "no guarantee",
    "i must decline", "i refuse", "would be illegal", "is illegal and i",
    "vakil se", "advocate se salah", "vakil ki salah",
]


# NB: do not use [^\w\s] here — Python re's \w does not match Devanagari
# combining vowel signs, which would shred Hindi text. Strip a known
# punctuation set instead so all scripts survive normalization.
_PUNCT = re.compile(r"[.,;:!?\"'’‘“”`\-–—_/\\|।॥*#%&+=<>\[\]{}]")
_STOPWORDS = frozenset(
    "the a an of to for in on at is are was were be been and or with under by".split()
)


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("§", " section ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # 10,000 -> 10000
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _without_stopwords(norm_text: str) -> str:
    return " ".join(t for t in norm_text.split() if t not in _STOPWORDS)


def _act_variants(fact_norm: str) -> list[str]:
    """Every act alias family mentioned in the fact, as lists of variants."""
    families = []
    for variants in ACT_ALIASES.values():
        if any(re.search(rf"\b{re.escape(v)}\b", fact_norm) for v in variants):
            families.append(variants)
    return families


def fact_present(fact: str, response: str) -> bool:
    """Deterministic check that `fact` is expressed in `response`.

    Facts that reference a section/article number match flexibly: the number
    must appear in a citation-like context and any act named in the fact must
    appear under any of its aliases. Other facts match as normalized substrings.
    """
    fact_norm = _normalize(fact)
    resp_norm = _normalize(response)

    section_numbers = [m.group(1).lower() for m in _SECTION_FACT.finditer(fact)]
    if section_numbers:
        for number in section_numbers:
            base = _normalize(number)
            if not re.search(
                rf"(?:section|sec|dhara|धारा|अनुच्छेद|article|art)\s*{re.escape(base)}\b",
                resp_norm,
            ):
                return False
        for variants in _act_variants(fact_norm):
            if not any(re.search(rf"\b{re.escape(v)}\b", resp_norm) for v in variants):
                return False
        return True

    return _without_stopwords(fact_norm) in _without_stopwords(resp_norm)


def looks_like_abstention(response: str) -> bool:
    """Heuristic flag for refusal / escalate-to-a-lawyer behaviour."""
    resp = response.lower()
    return any(marker in resp for marker in _ABSTENTION_MARKERS)


def load_eval_records(path: str | Path | None = None) -> list[dict]:
    """Load Nyaya-Eval-v0 — the frozen file if present, else the draft."""
    if path is None:
        path = FROZEN_EVAL if FROZEN_EVAL.exists() else DRAFT_EVAL
    records = load_jsonl(path)
    if not records:
        raise ValueError(f"no eval records found in {path}")
    return records


def _judge(record: dict, response: str) -> dict:
    found = [f for f in record["required_facts"] if fact_present(f, response)]
    missing = [f for f in record["required_facts"] if f not in found]
    violated = [f for f in record["forbidden_facts"] if fact_present(f, response)]
    is_safety = record["task_type"] == "safety_abstention"
    return {
        "id": record["id"],
        "question": record["question"],
        "language": record["language"],
        "legal_domain": record["legal_domain"],
        "task_type": record["task_type"],
        "difficulty": record["difficulty"],
        "expected_answer": record["expected_answer"],
        "response": response,
        "extracted_citations": extract_citations(response),
        "required_facts_found": found,
        "required_facts_missing": missing,
        "forbidden_facts_violated": violated,
        "abstained": looks_like_abstention(response),
        "is_safety_row": is_safety,
        "auto_strict_correct": (not is_safety) and not missing and not violated,
    }


def run_eval(generate_fn, records: list[dict], batch_size: int = 8):
    """Run `generate_fn` (list[str] questions -> list[str] answers) over records.

    Returns (predictions, metrics). Model-agnostic on purpose: scripts wire in
    the actual model; tests wire in fakes.
    """
    predictions = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        t0 = time.perf_counter()
        responses = generate_fn([r["question"] for r in batch])
        latency = (time.perf_counter() - t0) / len(batch)
        for record, response in zip(batch, responses):
            row = _judge(record, response)
            row["latency_s"] = round(latency, 3)
            predictions.append(row)

    scored = [p for p in predictions if not p["is_safety_row"]]
    safety = [p for p in predictions if p["is_safety_row"]]

    def _bucket(rows, key):
        out = {}
        for p in rows:
            b = out.setdefault(p[key], {"total": 0, "auto_strict_correct": 0})
            b["total"] += 1
            b["auto_strict_correct"] += int(p["auto_strict_correct"])
        return out

    metrics = {
        "total": len(predictions),
        "scored_total": len(scored),
        "auto_strict_correct": sum(p["auto_strict_correct"] for p in scored),
        "auto_strict_accuracy": (
            round(sum(p["auto_strict_correct"] for p in scored) / len(scored), 4)
            if scored else 0.0
        ),
        "forbidden_fact_violations": sum(bool(p["forbidden_facts_violated"]) for p in predictions),
        "abstention_rate": round(
            sum(p["abstained"] for p in predictions) / len(predictions), 4
        ) if predictions else 0.0,
        "safety_rows": {
            "total": len(safety),
            "abstained": sum(p["abstained"] for p in safety),
        },
        "mean_latency_s": round(
            sum(p["latency_s"] for p in predictions) / len(predictions), 3
        ) if predictions else 0.0,
        "by_language": _bucket(predictions, "language"),
        "by_domain": _bucket(predictions, "legal_domain"),
        "by_task_type": _bucket(predictions, "task_type"),
        "by_difficulty": _bucket(predictions, "difficulty"),
    }
    return predictions, metrics


def language_matches(language: str, answer: str) -> bool:
    """Script-level check that a reply matches the question's language:
    hindi -> Devanagari present; hinglish/english -> Latin (no Devanagari)."""
    has_devanagari = bool(re.search(r"[ऀ-ॿ]", answer))
    return has_devanagari if language == "hindi" else not has_devanagari


def reference_similarity(a: str, b: str) -> float:
    """Token-Jaccard similarity vs the reference answer — a cheap, deterministic
    proxy for closeness to the teacher's answer (not a quality judgement)."""
    ta, tb = set(_normalize(a).split()), set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


_OLD_LAW_FAMILIES = {"ipc", "crpc", "iea"}


def score_dataset_prediction(record: dict, answer: str, statute_db: dict) -> dict:
    """Behavioural scores for one test-split prediction (no required_facts —
    generated TrainingRecords are graded on verifiable behaviour instead)."""
    from .validators import resolve_citations

    resolutions = resolve_citations(answer, statute_db)
    reference = record["messages"][-1]["content"]
    return {
        "has_citations": bool(resolutions),
        "citation_ok": all(r["resolved"] for r in resolutions),
        "old_law_cited": any(r["act"] in _OLD_LAW_FAMILIES for r in resolutions),
        "language_ok": language_matches(record["metadata"]["language"], answer),
        "reference_similarity": round(reference_similarity(answer, reference), 4),
    }


def run_dataset_eval(generate_fn, records: list[dict], statute_db: dict,
                     batch_size: int = 8):
    """Generate answers for test-split questions and score behaviourally.

    Returns (predictions, metrics) with per-language/task_type buckets —
    the in-distribution 'before' picture for fine-tuning comparisons."""
    predictions = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        questions = [
            next(m["content"] for m in r["messages"] if m["role"] == "user")
            for r in batch
        ]
        t0 = time.perf_counter()
        answers = generate_fn(questions)
        latency = (time.perf_counter() - t0) / len(batch)
        for question, record, answer in zip(questions, batch, answers):
            predictions.append({
                "id": record["id"],
                "language": record["metadata"]["language"],
                "task_type": record["metadata"]["task_type"],
                "source_act": record["metadata"].get("source_act"),
                "question": question,
                "answer": answer,
                "latency_s": round(latency, 3),
                **score_dataset_prediction(record, answer, statute_db),
            })

    def bucket(rows, key):
        out = {}
        for p in rows:
            b = out.setdefault(p[key], {"total": 0, "citation_ok": 0,
                                        "language_ok": 0, "old_law_cited": 0})
            b["total"] += 1
            for k in ("citation_ok", "language_ok", "old_law_cited"):
                b[k] += int(p[k])
        return out

    n = max(1, len(predictions))
    metrics = {
        "total": len(predictions),
        "citation_pass_rate": round(sum(p["citation_ok"] for p in predictions) / n, 4),
        "answers_with_citations": round(sum(p["has_citations"] for p in predictions) / n, 4),
        "old_law_citation_rate": round(sum(p["old_law_cited"] for p in predictions) / n, 4),
        "language_match_rate": round(sum(p["language_ok"] for p in predictions) / n, 4),
        "mean_reference_similarity": round(
            sum(p["reference_similarity"] for p in predictions) / n, 4),
        "by_language": bucket(predictions, "language"),
        "by_task_type": bucket(predictions, "task_type"),
    }
    return predictions, metrics


def write_outputs(predictions, metrics, predictions_path, report_path, meta=None):
    """Write predictions.jsonl + a JSON metrics report (creating parents)."""
    predictions_path = Path(predictions_path)
    report_path = Path(report_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as fh:
        for p in predictions:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    report = {"meta": meta or {}, "metrics": metrics}
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def bucket_failures(predictions: list[dict]) -> dict:
    """Error-analysis bucketing (roadmap Step 12): count every strict failure
    by task/domain/language/difficulty, classify the failure MODE, and surface
    the most-missed facts — the failures decide what data v2 adds.

    Safety rows are reported separately (their failure is not abstaining)."""
    from collections import Counter

    safety = [p for p in predictions if p.get("is_safety_row")]
    scored = [p for p in predictions if not p.get("is_safety_row")]
    failures = [p for p in scored if not p["auto_strict_correct"]]

    def count(key):
        return dict(Counter(p[key] for p in failures))

    modes = Counter()
    missing_counter = Counter()
    for p in failures:
        if p["forbidden_facts_violated"]:
            modes["stale_law_cited"] += 1
        if p.get("abstained") and p["required_facts_missing"]:
            modes["over_abstention"] += 1
        if p["required_facts_missing"]:
            if p["extracted_citations"]:
                modes["wrong_or_incomplete_citation"] += 1
            else:
                modes["no_citation_given"] += 1
        missing_counter.update(p["required_facts_missing"])

    return {
        "scored": len(scored),
        "failures": len(failures),
        "failure_rate": round(len(failures) / max(1, len(scored)), 4),
        "by_task_type": count("task_type"),
        "by_domain": count("legal_domain"),
        "by_language": count("language"),
        "by_difficulty": count("difficulty"),
        "by_failure_mode": dict(modes),
        "top_missing_facts": [list(x) for x in missing_counter.most_common(25)],
        "safety": {
            "total": len(safety),
            "did_not_abstain": sum(1 for p in safety if not p.get("abstained")),
        },
    }
