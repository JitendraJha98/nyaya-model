"""QLoRA training wrapper: load a YAML config (configs/*.yaml), build the
4-bit NF4 quantized Qwen2.5-3B-Instruct + LoRA adapter, and run TRL SFTTrainer.

Config knobs are documented in configs/smoke.yaml and configs/train_v1.yaml.

TODO: implement train(config_path) — used by scripts/08_smoke_train.py and 09_train.py.
"""
