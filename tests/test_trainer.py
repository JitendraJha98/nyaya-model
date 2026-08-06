"""Tests for the LoRA trainer's pure config logic — no GPU/heavy imports."""

import pytest
import yaml

from nyaya.trainer import load_config, lora_kwargs, training_kwargs

ROOT_SMOKE = "configs/smoke.yaml"


def _write_config(tmp_path, mutate=None):
    config = yaml.safe_load(open(ROOT_SMOKE, encoding="utf-8"))
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
        smoke = training_kwargs(load_config("configs/smoke.yaml"))
        v1 = training_kwargs(load_config("configs/train_v1.yaml"))
        assert smoke.get("eval_steps") is None
        # cadence must sit well under the ~197 total steps of the v1 run
        assert v1["eval_steps"] == 50
