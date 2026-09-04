"""Step 13 (v2) — RAG eval: retrieval + v1 adapter on the frozen eval set.

The v2 architecture under test: the statute retriever surfaces top-k verbatim
sections, the v1 LoRA adapter (trained citing behaviour) answers from them.
Scored with the exact frozen-eval harness used for the 0.81% baseline and the
v1 checkpoint evals, so the three numbers are directly comparable.

Each prediction is annotated with what was retrieved and whether the gold
sections were in context, so accuracy can be split into retrieval-limited
vs model-limited — that split decides whether v2.1 is a better retriever
or a better prompt/model.

Outputs:
    reports/rag_eval.json
    outputs/rag-v2/<label>/predictions.jsonl

Usage:
    python scripts/16_rag_eval.py --adapter outputs/legal-3b-v1/checkpoint-50
    python scripts/16_rag_eval.py --adapter none            # RAG + base model ablation
    python scripts/16_rag_eval.py --no-rag                  # adapter only (sanity anchor)
"""

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Devanagari on cp1252 Windows consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jinja2 import TemplateError

from nyaya.evaluation import load_eval_records, run_eval
from nyaya.prompts import NYAYA_SYSTEM_PROMPT
from nyaya.retrieval import build_rag_prompt, load_statute_index

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def pick_dtype():
    """Native bf16 only on Ampere+ (sm_80); fp16 on older GPUs.

    Training ran on GPUs with native bf16, but evals now run on whatever free GPU
    is available, and Turing T4s (sm_75) have no bf16 tensor cores.

    Do NOT use torch.cuda.is_bf16_supported() here. It defaults to
    including_emulation=True and so returns True on a T4, where bf16 is
    emulated in software. That is not an error -- it just runs several times
    slower, silently. A Kaggle eval took >4.5h before this was spotted in a
    "[load] ... as torch.bfloat16" line that should have been impossible on
    the T4 it was running on.

    Compute capability is unambiguous, so check that instead.
    """
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def load_model(adapter_dir: str | None, model_id: str = MODEL_ID):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = pick_dtype()
    print(f"[load] {model_id} as {dtype}")
    # Pin to one device: device_map="auto" on a two-GPU Kaggle T4 box splits the
    # model and inputs across devices (docs/HANDOFF.md §5). A 3-4B model in fp16
    # fits a single 16 GB T4.
    device_map = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=device_map)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model


def chat_text(tokenizer, user_content: str) -> str:
    """Render system + user through the model's chat template.

    Some templates (the Gemma family) raise on a `system` role. Fold the system
    prompt into the user turn there so the same eval runs on any reader."""
    messages = [{"role": "system", "content": NYAYA_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (TypeError, ValueError, TemplateError):  # strict templates raise any of these
        merged = [{"role": "user", "content": f"{NYAYA_SYSTEM_PROMPT}\n\n{user_content}"}]
        return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)


def build_rag_generate_fn(tokenizer, model, index, k: int,
                          retrieval_log: dict, max_new_tokens: int = 384,
                          rewrite: bool = False, rewrite_log: dict | None = None):
    """generate_fn for run_eval: retrieve per question, answer from context.

    rewrite=True runs nyaya.rewrite first: Hindi/Hinglish questions are turned
    into one line of statutory English by the same model (a short greedy
    generation) and retrieval sees both the original and the rewrite. The
    rewritten query is recorded in rewrite_log[question] when a dict is given.
    """
    import torch

    from nyaya.rewrite import rewrite_query

    def plain_generate(prompt: str) -> str:
        text = chat_text(tokenizer, prompt)
        enc = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def generate(questions):
        prompts = []
        for q in questions:
            query = rewrite_query(q, plain_generate) if rewrite else q
            if rewrite_log is not None and query != q:
                rewrite_log[q] = query
            hits = index.retrieve(query, k=k) if index else []
            retrieval_log[q] = [f"{h['act_id']}:{h['section'].upper()}" for h in hits]
            prompts.append(build_rag_prompt(q, hits) if index else q)
        texts = [chat_text(tokenizer, p) for p in prompts]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)

    return generate


def build_endpoint_generate_fn(base_url: str, model: str, index, k: int,
                               retrieval_log: dict, max_new_tokens: int = 384,
                               rewrite: bool = False, rewrite_log: dict | None = None,
                               concurrency: int = 8, extra_body: dict | None = None,
                               timeout_s: int = 600):
    """generate_fn for run_eval against any OpenAI-compatible chat endpoint
    (vLLM, llama.cpp's llama-server, Ollama, or a hosted API).

    Same retrieval, same system prompt and same RAG prompt as the transformers
    path, greedy decoding (temperature 0), so a reader served this way is scored
    on the same footing as one loaded in-process. Retrieval runs in the calling
    thread (the dense stage is not thread-safe); only the HTTP calls of a batch
    run concurrently. NYAYA_TEACHER_API_KEY is sent as a bearer token when set.
    extra_body is merged into every request, e.g.
    {"chat_template_kwargs": {"enable_thinking": False}} for Qwen3 on vLLM.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    import requests

    from nyaya.rewrite import rewrite_query

    headers = {"Content-Type": "application/json"}
    key = os.environ.get("NYAYA_TEACHER_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = f"{base_url.rstrip('/')}/chat/completions"
    session = requests.Session()

    def chat(user_content: str, max_tokens: int) -> str:
        body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": NYAYA_SYSTEM_PROMPT},
                             {"role": "user", "content": user_content}]}
        body.update(extra_body or {})
        last_error = None
        for attempt in range(3):
            try:
                resp = session.post(url, headers=headers, json=body, timeout=timeout_s)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"] or ""
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"endpoint call failed after 3 attempts: {last_error}")

    def generate(questions):
        prompts = []
        for q in questions:
            query = rewrite_query(q, lambda prompt: chat(prompt, 48)) if rewrite else q
            if rewrite_log is not None and query != q:
                rewrite_log[q] = query
            hits = index.retrieve(query, k=k) if index else []
            retrieval_log[q] = [f"{h['act_id']}:{h['section'].upper()}" for h in hits]
            prompts.append(build_rag_prompt(q, hits) if index else q)
        with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(prompts)))) as pool:
            return list(pool.map(lambda prompt: chat(prompt, max_new_tokens), prompts))

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="outputs/legal-3b-v1/checkpoint-50",
                        help='LoRA adapter dir, or "none" for the base model')
    parser.add_argument("--no-rag", action="store_true",
                        help="skip retrieval (plain adapter, sanity anchor)")
    parser.add_argument("--dense", action="store_true",
                        help="hybrid retrieval: BM25 + multilingual-e5 (RRF)")
    parser.add_argument("--model", default=MODEL_ID,
                        help="base model id or a merged-model dir "
                             "(e.g. the DPO output)")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--canonical-dir", default=str(ROOT / "data" / "canonical"))
    parser.add_argument("--out-dir", default=str(ROOT))
    parser.add_argument("--label")
    args = parser.parse_args()

    adapter = None if args.adapter.lower() == "none" else args.adapter
    label = args.label or "_".join(filter(None, [
        "rag" if not args.no_rag else "norag",
        "dense" if args.dense else None,
        f"k{args.k}" if not args.no_rag else None,
        f"{Path(adapter).parent.name}-{Path(adapter).name}" if adapter
        else (f"{Path(args.model).parent.name}-merged" if args.model != MODEL_ID
              else "base")]))

    index = None if args.no_rag else load_statute_index(args.canonical_dir)
    if index is not None and args.dense:
        from nyaya.dense import attach_dense_index
        attach_dense_index(index,
                           cache_path=ROOT / "data" / "generated" / "e5_doc_vectors.npy")
    records = load_eval_records()
    if args.limit:
        records = records[: args.limit]

    gold_by_question = {}
    if index:
        for rec in records:
            gold = set()
            for fact in rec.get("required_facts", []):
                gold.update(index.referenced_keys(fact, domain=rec.get("legal_domain")))
            gold_by_question[rec["question"]] = gold

    print(f"[rag-eval] {label}: {len(records)} frozen questions, "
          f"adapter={adapter or 'base'}, k={'off' if args.no_rag else args.k}")
    tokenizer, model = load_model(adapter, model_id=args.model)
    retrieval_log: dict = {}
    generate = build_rag_generate_fn(tokenizer, model, index, args.k, retrieval_log)
    t0 = time.time()
    predictions, metrics = run_eval(generate, records, batch_size=args.batch_size)

    hit_split = {"gold_in_context": [0, 0], "gold_missing": [0, 0], "no_gold": [0, 0]}
    for p in predictions:
        retrieved = retrieval_log.get(p["question"], [])
        gold = gold_by_question.get(p["question"], set())
        p["retrieved"] = retrieved
        p["gold_sections"] = sorted(gold)
        p["gold_in_context"] = bool(gold) and gold <= set(retrieved)
        if p["is_safety_row"]:
            continue
        bucket = ("no_gold" if not gold
                  else "gold_in_context" if p["gold_in_context"] else "gold_missing")
        hit_split[bucket][0] += 1
        hit_split[bucket][1] += int(p["auto_strict_correct"])

    out = Path(args.out_dir) / "outputs" / "rag-v2" / label
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for p in predictions:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    report = {
        "label": label,
        "model": args.model,
        "adapter": adapter or "none",
        "rag": not args.no_rag,
        "k": None if args.no_rag else args.k,
        "frozen_eval_questions": len(records),
        "strict_accuracy": metrics["auto_strict_accuracy"],
        "strict_correct": metrics["auto_strict_correct"],
        "scored_total": metrics["scored_total"],
        "forbidden_fact_violations": metrics["forbidden_fact_violations"],
        "abstention_rate": metrics["abstention_rate"],
        "safety_rows": metrics["safety_rows"],
        "by_language": metrics["by_language"],
        "accuracy_by_retrieval": {
            name: {"n": n, "strict_correct": c,
                   "accuracy": round(c / n, 4) if n else None}
            for name, (n, c) in hit_split.items()
        },
        "eval_seconds": round(time.time() - t0),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path = Path(args.out_dir) / "reports" / "rag_eval.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if "runs" not in existing:
            existing = {"runs": {}}
    existing.setdefault("runs", {})[label] = report
    report_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "by_language"},
                     ensure_ascii=False, indent=2))
    print(f"[out] {report_path}")


if __name__ == "__main__":
    main()
