"""Dataset loading and formatting for TRL/Transformers.

- Loads JSONL of TrainingRecord (see schemas.py); TRL consumes the `messages` field.
- Applies Qwen's ChatML chat template.
- Grouped splitting lives in scripts/07_create_splits.py: split by
  metadata.source_sections, never by row.

TODO: implement load_jsonl(), to_trl_dataset(), grouped_split().
"""
