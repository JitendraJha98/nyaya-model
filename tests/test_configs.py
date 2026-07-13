"""Sanity tests for configs/ — they are consumed by scripts, so keep them loadable."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _load(name):
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def test_hf_datasets_entries_are_well_formed():
    config = _load("hf_datasets.yaml")
    assert config["datasets"], "config must list at least one dataset"
    for entry in config["datasets"]:
        assert isinstance(entry["id"], str) and entry["id"]
        assert isinstance(entry["enabled"], bool)
        assert isinstance(entry["purpose"], str) and entry["purpose"]


def test_training_configs_have_required_keys():
    for name in ("smoke.yaml", "train_v1.yaml"):
        config = _load(name)
        assert config["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
        assert config["method"] == "lora"
        for key in ("run_name", "output_dir", "data", "lora", "training"):
            assert key in config, f"{name} missing {key}"


def test_no_quantization_anywhere():
    # Project decision (2026-07-13): full-precision bf16 LoRA, no QLoRA/4-bit.
    for name in ("smoke.yaml", "train_v1.yaml"):
        config = _load(name)
        assert "quantization" not in config, f"{name} still has a quantization block"
        assert "8bit" not in config["training"].get("optim", "")


def test_smoke_and_v1_share_hyperparameters():
    # The two configs are intentionally self-contained ("every experiment a config");
    # this guards against the copies drifting apart unintentionally.
    smoke, v1 = _load("smoke.yaml"), _load("train_v1.yaml")
    assert smoke["lora"] == v1["lora"]
    for key in ("max_seq_length", "learning_rate", "num_train_epochs", "lr_scheduler_type"):
        assert smoke["training"][key] == v1["training"][key]
