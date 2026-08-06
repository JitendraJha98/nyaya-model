"""LoRA training wrapper: load a YAML config (configs/*.yaml), attach a LoRA
adapter to the full-precision bf16 base model, and run TRL SFTTrainer.

NO quantization — project decision (2026-07-13): the GPU pool (A100 80GB)
fits Qwen2.5-3B in bf16 with LoRA comfortably, so QLoRA/4-bit complexity
(bitsandbytes) is deliberately out. load_config() rejects configs that try.

Pure config mapping (load_config/lora_kwargs/training_kwargs) is imported and
tested everywhere; train() lazily imports torch/peft/trl so this module stays
importable on machines without a working GPU stack.

Config knobs are documented in configs/smoke.yaml and configs/train_v1.yaml.
Used by scripts/08_smoke_train.py and 09_train.py.
"""

from pathlib import Path

import yaml

REQUIRED_KEYS = ("run_name", "model_id", "output_dir", "data", "method", "lora", "training")


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _native_bf16() -> bool:
    """True only where the GPU has real bf16 tensor cores (Ampere, sm_80+).

    NOT torch.cuda.is_bf16_supported(): it defaults to including_emulation=True
    and answers True on a Turing T4, where bf16 runs in software several times
    slower. Import is local so this module stays importable without a GPU
    stack, which the rest of the file is careful to preserve.
    """
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 8


def pick_dtype():
    """bf16 on Ampere+, fp16 on older GPUs, fp32 on CPU."""
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if _native_bf16() else torch.float16


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ValueError(f"config missing keys: {', '.join(missing)}")
    if config["method"] != "lora":
        raise ValueError(
            f"method must be 'lora' (got {config['method']!r}) — Nyaya trains "
            "full-precision bf16 LoRA, no quantization"
        )
    if "quantization" in config:
        raise ValueError(
            "quantization block found — Nyaya trains full-precision bf16 LoRA; "
            "remove the block (no QLoRA/4-bit)"
        )
    return config


def lora_kwargs(config: dict) -> dict:
    """Map the config's lora block to peft.LoraConfig kwargs."""
    lora = config["lora"]
    return {
        "r": lora["r"],
        "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"],
        "target_modules": list(lora["target_modules"]),
        "task_type": "CAUSAL_LM",
        "bias": "none",
    }


def training_kwargs(config: dict) -> dict:
    """Map the config's training block to TRL SFTConfig/TrainingArguments kwargs."""
    t = config["training"]
    kwargs = {
        "output_dir": config["output_dir"],
        "run_name": config["run_name"],
        "num_train_epochs": t["num_train_epochs"],
        "learning_rate": float(t["learning_rate"]),
        "warmup_ratio": t["warmup_ratio"],
        "lr_scheduler_type": t["lr_scheduler_type"],
        "gradient_checkpointing": t["gradient_checkpointing"],
        # Config says bf16, but only Ampere+ has bf16 tensor cores. On a
        # Turing T4 the flag silently selects software emulation, which is
        # several times slower -- that cost 4.5h of GPU on an eval run before
        # it was spotted. Honour the config only where the hardware can.
        "bf16": t["bf16"] and _native_bf16(),
        "fp16": t["bf16"] and _cuda_available() and not _native_bf16(),
        "optim": t["optim"],
        "per_device_train_batch_size": t["per_device_train_batch_size"],
        "gradient_accumulation_steps": t["gradient_accumulation_steps"],
        "logging_steps": t["logging_steps"],
        "save_steps": t["save_steps"],
        "max_seq_length": t["max_seq_length"],
        "eval_steps": t.get("eval_steps"),
        "report_to": "none",
    }
    return kwargs


def _filter_to_signature(cls, kwargs: dict) -> dict:
    """Keep only kwargs the installed TRL/transformers version accepts.

    Handles cross-version renames (max_seq_length -> max_length,
    evaluation_strategy -> eval_strategy) without pinning."""
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters)
    out = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in accepted:
            out[key] = value
        elif key == "max_seq_length" and "max_length" in accepted:
            out["max_length"] = value
    return out


def train(config_path: str | Path) -> dict:
    """Run LoRA SFT per the config; returns trainer metrics."""
    import torch

    # Compat shim: newer TRL probes torch.distributed.tensor.DTensor in its
    # chunked-CE path; on torch 2.4 (the training image) the module exists but
    # lacks the attribute, crashing the first forward. A dummy class makes the
    # isinstance check False and TRL takes the ordinary single-GPU path.
    import torch.distributed.tensor as _dtensor_mod
    if not hasattr(_dtensor_mod, "DTensor"):
        class _NoDTensor:  # never instantiated
            pass
        _dtensor_mod.DTensor = _NoDTensor

    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from .dataset import to_trl_dataset

    config = load_config(config_path)
    data = config["data"]

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    dtype = pick_dtype()
    print(f"[train] loading {config['model_id']} as {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], dtype=dtype, device_map="auto"
    )
    model.config.use_cache = False  # required with gradient checkpointing

    train_dataset = to_trl_dataset(data["train_file"], max_examples=data.get("max_examples"))
    eval_dataset = to_trl_dataset(data["val_file"]) if data.get("val_file") else None

    kwargs = training_kwargs(config)
    if eval_dataset is None:
        kwargs.pop("eval_steps", None)
    else:
        kwargs["eval_strategy"] = "steps"
        kwargs["evaluation_strategy"] = "steps"  # older transformers name
    args = SFTConfig(**_filter_to_signature(SFTConfig, kwargs))

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LoraConfig(**lora_kwargs(config)),
        processing_class=tokenizer,
    )
    result = trainer.train()
    final_dir = Path(config["output_dir"]) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return dict(result.metrics)
