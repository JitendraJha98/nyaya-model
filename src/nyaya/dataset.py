"""Dataset loading and formatting for TRL/Transformers.

- Loads JSONL of TrainingRecord (see schemas.py); TRL consumes the `messages` field.
- Applies Qwen's ChatML chat template.
- Grouped splitting is implemented here as grouped_split() and driven by
  scripts/07_create_splits.py: split by metadata.source_sections, never by row.
"""

import json
import random
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts (skips blank lines)."""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_trl_dataset(path: str | Path, max_examples: int | None = None):
    """JSONL of TrainingRecord -> datasets.Dataset with the `messages` column
    TRL's SFTTrainer consumes (conversational format; Qwen's chat template is
    applied by the trainer's tokenizer)."""
    from datasets import Dataset

    records = load_jsonl(path)
    if max_examples:
        records = records[:max_examples]
    if not records:
        raise ValueError(f"no training records in {path}")
    return Dataset.from_list([{"messages": r["messages"]} for r in records])


def grouped_split(
    records: list[dict],
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 42,
    holdout_acts: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Split by SOURCE SECTION, never by row.

    All examples sharing any metadata.source_sections key land in exactly one
    split. Records from `holdout_acts` (e.g. two entire acts, per the roadmap)
    go wholly to test — the honest generalization measure.
    """
    holdout = set(holdout_acts or [])
    heldout_rows = [r for r in records if r["metadata"].get("source_act") in holdout]
    rest = [r for r in records if r["metadata"].get("source_act") not in holdout]

    groups: dict[str, list[dict]] = {}
    for r in rest:
        key = "|".join(sorted(r["metadata"]["source_sections"])) or r["id"]
        groups.setdefault(key, []).append(r)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val = round(len(keys) * val_fraction)
    n_test = round(len(keys) * test_fraction)
    val_keys = set(keys[:n_val])
    test_keys = set(keys[n_val:n_val + n_test])

    splits = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        bucket = "val" if key in val_keys else "test" if key in test_keys else "train"
        splits[bucket].extend(groups[key])
    splits["test"].extend(heldout_rows)
    return splits
