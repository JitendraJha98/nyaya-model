---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
base_model: NyayaLabs98/nyaya-3b-v3
base_model_relation: quantized
language:
- en
- hi
library_name: llama.cpp
pipeline_tag: text-generation
tags:
- legal
- india
- indian-law
- bns
- gguf
- llama.cpp
- ollama
- non-commercial
---

# Nyaya-3B-v3 — GGUF builds for llama.cpp and Ollama

Quantised builds of [`NyayaLabs98/nyaya-3b-v3`](https://huggingface.co/NyayaLabs98/nyaya-3b-v3),
the reader model of the [Nyaya](https://github.com/JitendraJha98/nyaya-model) Indian
legal guidance system. Converted with llama.cpp (`convert_hf_to_gguf.py`, build b10795)
from the merged bf16 weights.

| File | Size | Use |
|---|---:|---|
| `nyaya-3b-v3-Q4_K_M.gguf` | 1.93 GB | default: laptops, 4 GB RAM headroom |
| `nyaya-3b-v3-Q8_0.gguf` | 3.29 GB | near-lossless |

> **⚖️ Not legal advice.** Nyaya provides legal *information*. The practice of law in
> India is reserved to advocates enrolled under the Advocates Act, 1961. Consult a
> licensed advocate for anything consequential. Free legal aid: NALSA / DLSA.

> **📋 Non-commercial.** Derived from `Qwen/Qwen2.5-3B-Instruct` (Qwen Research
> License). Research / non-commercial use only.

## Read this first

On the project's own benchmark this model is **statistically tied with the base
model it was fine-tuned from**, and the Nyaya system now uses `Qwen/Qwen3-4B-Instruct-2507`
(Apache-2.0, official GGUF builds exist) as its default reader: 52.0% fact recall against
33.8% for these weights under identical retrieval. These files remain for anyone who wants
the Nyaya-aligned 3B. The system's accuracy comes from **retrieval**: with
the right statute section in context the reader reaches 63% fact recall, without it
20%. Used bare, this model answers from memory like any 3B model and can cite the
wrong section. Pair it with the statute retriever:

```bash
pip install "git+https://github.com/JitendraJha98/nyaya-model"
nyaya ask "police FIR nahi likh rahi, kya karu?"    # prints the sections to paste into your prompt
```

## Ollama

```bash
huggingface-cli download NyayaLabs98/nyaya-3b-v3-GGUF nyaya-3b-v3-Q4_K_M.gguf Modelfile --local-dir nyaya-gguf
cd nyaya-gguf && ollama create nyaya -f Modelfile
ollama run nyaya "Police FIR nahi likh rahi, kya karu?"
```

The `Modelfile` carries the Nyaya system prompt (legal information, cite current law,
recommend an advocate) and `temperature 0.3`.

## llama.cpp

```bash
llama-server -m nyaya-3b-v3-Q4_K_M.gguf -c 8192
# then POST to http://127.0.0.1:8080/v1/chat/completions with the system prompt from the Modelfile
```

## Evaluation of the underlying weights

See the [`nyaya-3b-v3` card](https://huggingface.co/NyayaLabs98/nyaya-3b-v3): Nyaya-Eval-v1
fact recall 32.9% vs base 34.3% (tied, 95% CI spans zero); BhashaBench-Legal
1,500-question sample 45.2% vs base 47.8% (tied). Quantisation was not separately
evaluated; expect Q8_0 to match bf16 within noise and Q4_K_M to lose a little.
