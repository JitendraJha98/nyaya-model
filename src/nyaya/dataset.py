"""Dataset loading and formatting for TRL/Transformers.

- Loads JSONL of TrainingRecord (see schemas.py); TRL consumes the `messages` field.
- Applies Qwen's ChatML chat template.
- Grouped splitting is implemented here as grouped_split() and driven by
  scripts/07_create_splits.py: split by metadata.source_sections, never by row.

TODO: implement to_trl_dataset(), grouped_split().
"""

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts (skips blank lines)."""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
