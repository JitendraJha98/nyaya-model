"""Step 33 — Does query rewriting fix Hindi / Hinglish retrieval?

Measures nyaya.rewrite on two sets, before and after, with the same index:

  1. the Devanagari and Hinglish lines of data/raw/citizen_questions.txt
     (real questions, never used for tuning): how many retrieve ZERO statute
     sections in the top 8;
  2. the Hindi / Hinglish records of Eval-v1 public whose facts name a section:
     how many have every gold section in the top 8 (full hit).

The rewriter is the reader model itself (greedy, <= 48 new tokens). CPU is
fine: ~50 short generations. Writes reports/rewrite_measurement.json.

Usage:
    python scripts/33_measure_rewrite.py                       # base Qwen2.5-3B-Instruct
    python scripts/33_measure_rewrite.py --model <local merged dir or Hub id>
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.retrieval import load_statute_index  # noqa: E402
from nyaya.rewrite import needs_rewrite, rewrite_query  # noqa: E402

QUESTIONS = ROOT / "data" / "raw" / "citizen_questions.txt"
EVAL = ROOT / "data" / "eval" / "nyaya_eval_v1_public.jsonl"
REPORT = ROOT / "reports" / "rewrite_measurement.json"
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _generator(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float32 if not torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype,
                                                 device_map={"": 0} if torch.cuda.is_available() else None).eval()

    def generate(prompt: str) -> str:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True)
        enc = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def _endpoint_generator(base_url: str, model: str):
    """Any OpenAI-compatible chat endpoint: llama.cpp's llama-server, Ollama,
    vLLM, or a hosted API (NYAYA_TEACHER_API_KEY is sent if set)."""
    import os

    import requests

    headers = {"Content-Type": "application/json"}
    key = os.environ.get("NYAYA_TEACHER_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def generate(prompt: str) -> str:
        resp = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, timeout=300,
                             json={"model": model, "messages": [{"role": "user", "content": prompt}],
                                   "max_tokens": 48, "temperature": 0})
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return generate


def _statute_hits(index, query: str, k: int = 8) -> set[str]:
    return {f"{r['act_id']}:{r['section'].upper()}" for r in index.retrieve(query, k=k)
            if r["act_id"] != "procedures_kb"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                   help="HF id / local dir (transformers), or the model name to send to --endpoint")
    p.add_argument("--endpoint", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1 "
                                      "(llama-server with a GGUF); avoids loading the model in-process")
    p.add_argument("--limit", type=int, help="cap the number of citizen questions (smoke runs)")
    args = p.parse_args()

    index = load_statute_index(ROOT / "data" / "canonical")
    generate = _endpoint_generator(args.endpoint, args.model) if args.endpoint else _generator(args.model)

    citizen = [q.strip() for q in QUESTIONS.read_text(encoding="utf-8").splitlines() if q.strip()]
    citizen = [q for q in citizen if needs_rewrite(q)][: args.limit or None]
    t0 = time.time()
    rewrites = {q: rewrite_query(q, generate) for q in citizen}
    seconds_per_rewrite = round((time.time() - t0) / max(1, len(citizen)), 1)

    def bucket(q: str) -> str:
        return "devanagari" if _DEVANAGARI.search(q) else "hinglish"

    zero = {"before": {"devanagari": 0, "hinglish": 0}, "after": {"devanagari": 0, "hinglish": 0}}
    n = {"devanagari": 0, "hinglish": 0}
    samples = []
    for q in citizen:
        b = bucket(q)
        n[b] += 1
        before, after = _statute_hits(index, q), _statute_hits(index, rewrites[q])
        zero["before"][b] += not before
        zero["after"][b] += not after
        if len(samples) < 12 and not before:
            samples.append({"question": q, "rewritten": rewrites[q][len(q):].strip(),
                            "after_top": sorted(after)[:3]})

    eval_rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_rows = [r for r in eval_rows if r.get("language") in ("hindi", "hinglish")]
    full_before = full_after = gold_bearing = 0
    for r in eval_rows:
        gold = set()
        for fact in r.get("required_facts", []):
            gold.update(index.referenced_keys(fact, domain=r.get("legal_domain")))
        if not gold:
            continue
        gold_bearing += 1
        full_before += gold <= _statute_hits(index, r["question"])
        full_after += gold <= _statute_hits(index, rewrite_query(r["question"], generate))

    report = {
        "model": args.model,
        "endpoint": args.endpoint,
        "seconds_per_rewrite": seconds_per_rewrite,
        "citizen_questions": {"n": n, "zero_statute_hits_before": zero["before"],
                              "zero_statute_hits_after": zero["after"]},
        "eval_v1_public_hi_hinglish": {"gold_bearing": gold_bearing, "full_hit_at_8_before": full_before,
                                       "full_hit_at_8_after": full_after},
        "samples": samples,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=False, indent=2))
    for s in samples[:6]:
        print(f"  {s['question'][:60]!r} -> {s['rewritten'][:70]!r} -> {s['after_top']}")
    print(f"\n[rewrite] wrote {REPORT}")


if __name__ == "__main__":
    main()
