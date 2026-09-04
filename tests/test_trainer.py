"""Tests for the LoRA trainer's pure config logic — no GPU/heavy imports."""

import warnings
from pathlib import Path

import pytest
import yaml

from nyaya.trainer import _filter_to_signature, load_config, lora_kwargs, training_kwargs

ROOT = Path(__file__).resolve().parents[1]
ROOT_SMOKE = ROOT / "configs" / "smoke.yaml"


def _write_config(tmp_path, mutate=None):
    config = yaml.safe_load(ROOT_SMOKE.read_text(encoding="utf-8"))
    if mutate:
        mutate(config)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(config), encoding="utf-8")
    return p


class TestLoadConfig:
    def test_loads_smoke_config(self):
        config = load_config(ROOT_SMOKE)
        assert config["method"] == "lora"
        assert config["model_id"] == "Qwen/Qwen2.5-3B-Instruct"

    def test_rejects_quantization_block(self, tmp_path):
        p = _write_config(tmp_path, lambda c: c.update(quantization={"load_in_4bit": True}))
        with pytest.raises(ValueError, match="quantization"):
            load_config(p)

    def test_rejects_non_lora_method(self, tmp_path):
        p = _write_config(tmp_path, lambda c: c.update(method="qlora"))
        with pytest.raises(ValueError, match="lora"):
            load_config(p)

    def test_rejects_missing_keys(self, tmp_path):
        p = _write_config(tmp_path, lambda c: c.pop("training"))
        with pytest.raises(ValueError, match="training"):
            load_config(p)


class TestKwargMapping:
    def test_lora_kwargs(self):
        kwargs = lora_kwargs(load_config(ROOT_SMOKE))
        assert kwargs["r"] == 32
        assert kwargs["lora_alpha"] == 64
        assert kwargs["lora_dropout"] == 0.05
        assert "q_proj" in kwargs["target_modules"]
        assert kwargs["task_type"] == "CAUSAL_LM"

    def test_training_kwargs(self):
        config = load_config(ROOT_SMOKE)
        kwargs = training_kwargs(config)
        assert kwargs["output_dir"] == config["output_dir"]
        assert kwargs["learning_rate"] == pytest.approx(1e-4)
        assert kwargs["run_name"] == config["run_name"]
        assert "quantization" not in kwargs
        assert "8bit" not in kwargs["optim"]

    def test_precision_follows_the_hardware_not_just_the_config(self):
        """bf16: true must not force bf16 onto a GPU that lacks the cores.

        Only Ampere+ (sm_80) has bf16 tensor cores. On a Turing T4 the flag
        selects software emulation, which is several times slower -- that cost
        4.5h of GPU on an eval run before it was noticed. So the config states
        intent and the hardware decides, with fp16 as the CUDA fallback and
        neither flag set on CPU.
        """
        from nyaya.trainer import _cuda_available, _native_bf16

        kwargs = training_kwargs(load_config(ROOT_SMOKE))
        assert kwargs["bf16"] is _native_bf16()
        assert kwargs["fp16"] is (_cuda_available() and not _native_bf16())
        assert not (kwargs["bf16"] and kwargs["fp16"]), "cannot request both"

    def test_eval_strategy_follows_val_file(self):
        smoke = training_kwargs(load_config(ROOT_SMOKE))
        v1 = training_kwargs(load_config(ROOT / "configs" / "train_v1.yaml"))
        assert smoke.get("eval_steps") is None
        # cadence must sit well under the ~197 total steps of the v1 run
        assert v1["eval_steps"] == 50


class TestNeftuneAndDroppedKeys:
    """NEFTune was configured in v3/v4/v5 but never reached TRL: the kwargs
    mapper did not read the key and the signature filter silently discarded
    unknown keys. Those runs trained without it. Both halves are pinned here."""

    def test_neftune_passes_through_when_configured(self):
        config = load_config(ROOT / "configs" / "train_v3.yaml")
        assert training_kwargs(config)["neftune_noise_alpha"] == 5

    def test_neftune_absent_when_not_configured(self):
        config = load_config(ROOT_SMOKE)
        assert training_kwargs(config).get("neftune_noise_alpha") is None

    def test_filter_warns_on_every_dropped_key(self):
        class FakeConfig:
            def __init__(self, output_dir, max_length):
                pass

        with pytest.warns(UserWarning, match="neftune_noise_alpha"):
            out = _filter_to_signature(
                FakeConfig, {"output_dir": "x", "max_seq_length": 4, "neftune_noise_alpha": 5})
        assert out == {"output_dir": "x", "max_length": 4}

    def test_filter_is_silent_when_nothing_is_dropped(self):
        class FakeConfig:
            def __init__(self, output_dir, max_length):
                pass

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _filter_to_signature(FakeConfig, {"output_dir": "x", "max_seq_length": 4})
