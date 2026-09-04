"""Step 15 (v3) — RAFT generation: teacher answers under the real RAG prompt.

v2 proved the adapter must be trained on inference-shaped prompts; RAFT goes
further and regenerates the ANSWERS under those prompts too: the teacher sees
retrieved context (gold + distractors, or gold deliberately withheld for a
miss_fraction of tasks) and writes the answer the student should give —
including honest "the provisions shown don't cover this" for misses.

Hard gate per answer (before it is written): every citation must resolve
against the CONTEXT (context_statute_db), not merely against real law.
Misses must cite nothing. scripts/05's validators still run downstream.

Resumable by task_id like scripts/04. For DPO pairs, --samples 2 generates
two candidates per task at temperature; pairing happens in scripts/21.

Usage:
    python scripts/20_generate_raft.py --splits train val test
    python scripts/20_generate_raft.py --splits train --samples 2 --sample-fraction 0.4
    TEACHER_BASE_URL=https://<openai-compatible-host>/v1 python scripts/20_generate_raft.py ...
    TEACHER_BASE_URL=http://127.0.0.1:8000/v1 TEACHER_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ \
        python scripts/20_generate_raft.py --version nyaya_instruct_v7   # local vLLM teacher
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.dataset import load_jsonl
from nyaya.generation import build_raft_plan, parse_raft_response
from nyaya.retrieval import context_statute_db, load_statute_index
from nyaya.validators import extract_citations, verify_citations

CONFIG = ROOT / "configs" / "generation.yaml"
OUT_DIR = ROOT / "data" / "generated"
RETRIES = 3
DEFAULT_VERSION = "nyaya_instruct_v4"


def call_teacher(prompt: str, teacher: dict, session: requests.Session,
                 temperature: float) -> str:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(teacher.get("api_key_env", ""), "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": teacher["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": teacher.get("max_tokens", 2048),
        "temperature": temperature,
    }
    last_error = None
    for attempt in range(RETRIES):
        try:
            resp = session.post(
                f"{teacher['base_url'].rstrip('/')}/chat/completions",
                json=payload, headers=headers, timeout=teacher.get("timeout_s", 300),
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError) as e:
            last_error = e
            if attempt < RETRIES - 1:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"teacher call failed after {RETRIES} attempts: {last_error}")


def gate(answer: str, task: dict) -> tuple[bool, str]:
    """Context containment: cite only what was shown; misses cite nothing;
    gold-bearing tasks must cite. Ungrounded tasks (safety) may skip citing
    but must still not cite outside the context."""
    citations = extract_citations(answer)
    if task["is_miss"]:
        return (not citations, "miss_task_cited" if citations else "ok")
    if task["source_sections"] and not citations:
        return False, "no_citation"
    context_db = context_statute_db(task["context_keys"])
    if citations and not verify_citations(answer, context_db):
        return False, "citation_outside_context"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--miss-fraction", type=float, default=0.10)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sample-fraction", type=float, default=1.0,
                        help="fraction of tasks that get the extra samples")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dense", action="store_true",
                        help="hybrid retrieval for distractor contexts "
                             "(match inference-time retrieval)")
    parser.add_argument("--version", default=DEFAULT_VERSION,
                        help="dataset version: names output files and scopes "
                             "resume-by-run_id")
    args = parser.parse_args()
    version = args.version

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    teacher = config["teacher"]
    teacher["base_url"] = os.environ.get("TEACHER_BASE_URL", teacher["base_url"])
    teacher["model"] = os.environ.get("TEACHER_MODEL", teacher["model"])

    index = load_statute_index(ROOT / "data" / "canonical")
    if args.dense:
        from nyaya.dense import (
            DEFAULT_ATTACH_MODEL,
            attach_dense_index,
            doc_vector_cache,
        )
        attach_dense_index(index, cache_path=doc_vector_cache(
            DEFAULT_ATTACH_MODEL, ROOT / "data" / "generated"))
    plan = []
    for split in args.splits:
        records = load_jsonl(ROOT / "data" / "splits" / f"{split}.jsonl")
        split_plan = build_raft_plan(records, index, k=args.k,
                                     miss_fraction=args.miss_fraction,
                                     seed=config["seed"])
        for task in split_plan:
            task["generator"] = teacher["model"]
            task["split"] = split  # v3 keeps v1's grouped-split assignment
        plan.extend(split_plan)

    # expand for multi-sample tasks (DPO candidates)
    expanded = []
    for i, task in enumerate(plan):
        n = args.samples if (i / max(1, len(plan))) < args.sample_fraction else 1
        for s in range(n):
            t = dict(task)
            t["sample_id"] = s
            t["run_id"] = f"{task['task_id']}_s{s}"
            expanded.append(t)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{version}_raw.jsonl"
    rejects_file = OUT_DIR / f"{version}_rejected.jsonl"
    done = set()
    if out_file.exists():
        done |= {r["metadata"]["run_id"] for r in load_jsonl(out_file)}
    if rejects_file.exists():
        done |= {r["run_id"] for r in load_jsonl(rejects_file)}
    pending = [t for t in expanded if t["run_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[plan] {len(expanded)} calls | done {len(done)} | running {len(pending)} "
          f"| teacher {teacher['model']} @ {teacher['base_url']}")

    session = requests.Session()
    stats = {"ok": 0, "gate_fail": 0, "trivial": 0, "failed": 0}
    started = time.time()

    def run_task(task):
        temp = 0.2 if task["sample_id"] == 0 else args.temperature
        return task, call_teacher(task["prompt"], teacher, session, temp)

    with out_file.open("a", encoding="utf-8") as fh, \
         rejects_file.open("a", encoding="utf-8") as rej, \
         ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(run_task, t) for t in pending]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                task, raw = future.result()
            except RuntimeError as e:
                stats["failed"] += 1
                print(f"  [{i}/{len(pending)}] FAILED: {e}")
                continue
            recs = parse_raft_response(raw, task, version)
            if not recs:
                stats["trivial"] += 1
                rej.write(json.dumps({"run_id": task["run_id"], "reason": "trivial",
                                      "raw": raw[:500]}, ensure_ascii=False) + "\n")
                continue
            rec = recs[0]
            ok, reason = gate(rec["messages"][-1]["content"], task)
            if not ok:
                stats["gate_fail"] += 1
                # full answer kept: gate failures are DPO "rejected" candidates
                rej.write(json.dumps({"run_id": task["run_id"],
                                      "task_id": task["task_id"],
                                      "split": task["split"], "reason": reason,
                                      "answer": rec["messages"][-1]["content"]},
                                     ensure_ascii=False) + "\n")
            else:
                stats["ok"] += 1
                rec["metadata"]["run_id"] = task["run_id"]
                rec["metadata"]["sample_id"] = task["sample_id"]
                rec["metadata"]["split"] = task["split"]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 100 == 0 or i == len(pending):
                rate = i / max(1, time.time() - started)
                print(f"  [{i}/{len(pending)}] ok={stats['ok']} gate_fail={stats['gate_fail']} "
                      f"trivial={stats['trivial']} failed={stats['failed']} "
                      f"({rate:.2f} calls/s)", flush=True)
                fh.flush()
                rej.flush()

    report = {"stats": stats, "calls": len(pending), "k": args.k,
              "miss_fraction": args.miss_fraction, "samples": args.samples,
              "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (ROOT / "reports" / "raft_generation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
