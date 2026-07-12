"""Canonical record schemas for every dataset in the pipeline.

Metadata is never thrown away — it becomes extremely valuable during error analysis.
"""

from dataclasses import dataclass, field


@dataclass
class StatuteSection:
    """One row of the statute DB (data/canonical/) — the single source of truth.

    Example: act_id="bns_2023", section="318", title="Cheating",
    replaces={"act": "ipc_1860", "section": "415-420"}.
    The `tags` field includes Hindi/colloquial terms and old IPC numbers —
    it powers both retrieval and IPC-era queries.
    """

    act_id: str
    act_name: str
    section: str
    title: str
    text: str                       # full verbatim section text
    chapter: str | None = None
    subsection: str | None = None
    effective_date: str | None = None
    replaces: dict | None = None    # {"act": "ipc_1860", "section": "415-420"}
    punishment_summary: str | None = None
    tags: list[str] = field(default_factory=list)
    source_url: str | None = None


@dataclass
class LawMapping:
    """One row of the IPC<->BNS / CrPC<->BNSS mapping table (from official MHA tables)."""

    old_act: str        # "IPC"
    old_section: str    # "420"
    new_act: str        # "BNS"
    new_section: str    # "318"
    note: str | None = None


@dataclass
class TrainingRecord:
    """Universal training schema. TRL consumes `messages`; metadata is for us.

    metadata keys: language, legal_domain, task_type, source_act,
    source_sections (list — the split key!), generator, verified, dataset_version.
    """

    id: str
    messages: list[dict]            # [{"role": "system"|"user"|"assistant", "content": ...}]
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalRecord:
    """One Nyaya-Eval-v0 question (data/eval/nyaya_eval_v0.jsonl). Frozen — never train on it."""

    id: str
    question: str
    language: str                   # english | hindi | hinglish
    legal_domain: str
    task_type: str
    expected_answer: str
    required_facts: list[str] = field(default_factory=list)
    forbidden_facts: list[str] = field(default_factory=list)
    difficulty: str = "medium"      # easy | medium | hard
    source: str | None = None
    split: str = "test"
