# GPU smoke-test image: verifies Qwen2.5-3B-Instruct loads and runs inference
# in-cluster (scripts/01_download_model.py). Not a serving image — no API,
# no training — see docs/ROADMAP.md for what's implemented so far.

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# The base image already ships a CUDA-enabled torch build; installing
# requirements.txt as-is would let pip resolve a CPU-only torch wheel over it.
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > requirements.nogpu.txt \
    && pip install --no-cache-dir -r requirements.nogpu.txt

COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

RUN pip install --no-cache-dir --no-deps -e .

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf-cache

ENTRYPOINT ["python", "scripts/01_download_model.py"]
