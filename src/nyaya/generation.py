"""Grounded synthetic-data generation (roadmap Step 7) — pure logic.

The single most important rule (docs/ROADMAP.md): every example is generated
FROM verbatim statute text passed into the prompt; the teacher must never cite
from memory. This module builds the deterministic generation plan and parses
teacher responses into TrainingRecord rows; scripts/04_generate_examples.py
wires it to an OpenAI-compatible endpoint (vLLM serving the teacher model).

Slices (composition configured in configs/generation.yaml):
  grounded_qa / hindi_qa / hinglish_qa  statute-grounded citizen Q&A
  procedural                            "how do I..." grounded in statute text
  law_mapping                           IPC/CrPC/IEA -> BNS/BNSS/BSA bridging
  terminology                           "what does X mean" grounded
  safety_abstention                     refusal/no-prediction/escalation (ungrounded)

metadata.source_sections ("act_id:section") is the grouped-split key —
never dropped.
"""

import hashlib
import json
import random
import re

from .prompts import GROUNDED_QA_PROMPT, NYAYA_SYSTEM_PROMPT

LANGUAGE_DIRECTIVES = {
    "english": "",
    "hindi": (
        "\nIMPORTANT: Write ALL questions and answers in Hindi (Devanagari script). "
        "Keep statutory citations like 'Section 318 of the Bharatiya Nyaya Sanhita' "
        "recognisable (Devanagari 'धारा 318' with the act name is fine)."
    ),
    "hinglish": (
        "\nIMPORTANT: Write ALL questions and answers in natural Hinglish — Hindi-English "
        "code-switching in Latin script, the way an ordinary Indian citizen texts "
        "(e.g. 'police FIR nahi likh rahi, kya karu?'). Citations stay in English form."
    ),
}

MAPPING_PROMPT = """\
You are creating training data for a legal-information assistant for Indian citizens.

OFFICIAL CORRESPONDENCE (the ONLY source of truth):
Old law: Section {old_section} of the {old_act}
New law: Section {new_section} of the {new_act}
Text of the new provision:
{new_text}

Generate 4 realistic Q&A pairs about this old-law-to-new-law change:
- Q1: a citizen asking directly which new section replaced the old one
- Q2: a citizen who only knows the old section number describing their situation
- Q3: asking whether the old section still applies to a case from before July 2024
- Q4: asking what changed in substance, if anything

Rules for answers:
- Use ONLY the correspondence and text above. Do NOT add sections from memory.
- Cite as: "Section <n> of the <Act Name>" exactly.
- State clearly that the old act applies to offences before 1 July 2024.
- Plain language, 10th-standard reading level.
- End with a one-line "this is general information, not legal advice" note.
Output as JSON array of objects with "question" and "answer" fields.{language_directive}
"""

SAFETY_PROMPT = """\
You are creating SAFETY training data for a legal-information assistant for Indian citizens.
The assistant must refuse to help with illegal acts, must never guarantee case outcomes,
must ask for missing facts instead of guessing, and must escalate to professional help
(licensed advocate, DLSA free legal aid, 112/181/1930 helplines) where appropriate.

Generate 4 realistic Q&A pairs where the CORRECT behaviour is {behaviour}:
- questions should sound like real, sometimes emotionally charged citizen messages
- answers must be firm but kind, brief, explain WHY, and offer a lawful alternative
- do NOT invent statute section numbers in these answers

Output as JSON array of objects with "question" and "answer" fields.{language_directive}
"""

SAFETY_BEHAVIOURS = [
    "refusing to assist with an illegal or fraudulent request",
    "declining to guarantee or predict a case outcome",
    "asking for essential missing facts instead of answering definitively",
    "urging escalation to emergency help or a licensed advocate",
]

TERMINOLOGY_DIRECTIVE = (
    "\nFocus every question on the MEANING of legal terms appearing in the text "
    "(e.g. 'cognizable', 'proviso', 'abetment') — a citizen asking what a word means."
)

PROCEDURAL_DIRECTIVE = (
    "\nFocus every question on PROCEDURE — 'how do I…', step-by-step, what documents, "
    "where to go, what timelines apply — strictly grounded in the text above."
)


def _statute_key(row: dict) -> str:
    return f"{row['act_id']}:{row['section']}"


def _grounded_prompt(row: dict, language: str, extra_directive: str = "") -> str:
    statute_text = (
        f"Section {row['section']} — {row['title']}\n"
        f"[{row['act_name']}]\n{row['text']}"
    )
    return (
        GROUNDED_QA_PROMPT.format(statute_text=statute_text)
        + LANGUAGE_DIRECTIVES[language]
        + extra_directive
    )


def build_generation_plan(
    statute_rows: list[dict],
    mappings: list[dict],
    composition: dict,
    seed: int = 42,
) -> list[dict]:
    """Deterministic task list for the teacher, per the composition config.

    Statute-grounded slices sample sections (long enough to ground 4 QAs)
    round-robin across acts so no single act dominates. Every task records
    source_sections — the grouped-split key.
    """
    rng = random.Random(seed)
    usable = [r for r in statute_rows if len(r["text"]) >= 200]
    by_act: dict[str, list[dict]] = {}
    for row in usable:
        by_act.setdefault(row["act_id"], []).append(row)
    for rows in by_act.values():
        rng.shuffle(rows)

    def next_sections(n: int) -> list[dict]:
        picked, acts = [], sorted(by_act)
        i = 0
        while len(picked) < n and any(by_act.values()):
            act = acts[i % len(acts)]
            if by_act[act]:
                picked.append(by_act[act].pop())
            i += 1
        return picked

    plan: list[dict] = []
    counter = 0

    def add(task_type: str, language: str, prompt: str, source_act: str | None,
            source_sections: list[str]):
        nonlocal counter
        counter += 1
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        plan.append(
            {
                "task_id": f"gen_{counter:06d}_{digest}",
                "task_type": task_type,
                "language": language,
                "source_act": source_act,
                "source_sections": source_sections,
                "prompt": prompt,
            }
        )

    for task_type, spec in composition.items():
        count, language = spec["count"], spec.get("language", "english")
        if task_type in ("grounded_qa", "hindi_qa", "hinglish_qa", "terminology", "procedural"):
            extra = {"terminology": TERMINOLOGY_DIRECTIVE, "procedural": PROCEDURAL_DIRECTIVE}.get(task_type, "")
            for row in next_sections(count):
                add(task_type, language, _grounded_prompt(row, language, extra),
                    row["act_id"], [_statute_key(row)])
        elif task_type == "law_mapping":
            act_ids = {"BNS": "bns_2023", "BNSS": "bnss_2023", "BSA": "bsa_2023"}
            texts = {_statute_key(r): r for r in statute_rows}
            pool = list(mappings)
            rng.shuffle(pool)
            made = 0
            for m in pool:
                if made >= count:
                    break
                new_key = f"{act_ids.get(m['new_act'], m['new_act'].lower())}:{m['new_section']}"
                row = texts.get(new_key)
                if row is None:
                    continue
                prompt = MAPPING_PROMPT.format(
                    old_section=m["old_section"], old_act=m["old_act"],
                    new_section=m["new_section"], new_act=m["new_act"],
                    new_text=row["text"][:4000],
                    language_directive=LANGUAGE_DIRECTIVES[language],
                )
                add("law_mapping", language, prompt, row["act_id"], [new_key])
                made += 1
        elif task_type == "safety_abstention":
            for i in range(count):
                behaviour = SAFETY_BEHAVIOURS[i % len(SAFETY_BEHAVIOURS)]
                prompt = SAFETY_PROMPT.format(
                    behaviour=behaviour,
                    language_directive=LANGUAGE_DIRECTIVES[language],
                )
                add("safety_abstention", language, prompt, None, [])
        else:
            raise ValueError(f"unknown task_type in composition: {task_type}")
    return plan


_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def parse_teacher_response(raw: str, task: dict, dataset_version: str) -> list[dict]:
    """Teacher output -> TrainingRecord rows. Unparseable output -> [] (the
    task is retried/dropped by the driver; a 15-20% rejection rate is healthy)."""
    candidates = [m.group(1) for m in _JSON_BLOCK.finditer(raw)] + [raw]
    items = None
    for c in candidates:
        c = c.strip()
        start, end = c.find("["), c.rfind("]")
        if start == -1 or end <= start:
            continue
        try:
            parsed = json.loads(c[start:end + 1])
            if isinstance(parsed, list):
                items = parsed
                break
        except json.JSONDecodeError:
            continue
    if not items:
        return []

    records = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        records.append(
            {
                "id": f"{task['task_id']}_{i:02d}",
                "messages": [
                    {"role": "system", "content": NYAYA_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "language": task["language"],
                    "legal_domain": task.get("source_act") or "general",
                    "task_type": task["task_type"],
                    "source_act": task.get("source_act"),
                    "source_sections": task["source_sections"],
                    "generator": task.get("generator", "unknown"),
                    "verified": False,
                    "dataset_version": dataset_version,
                },
            }
        )
    return records
