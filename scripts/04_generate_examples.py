"""Step 4 — Generate training examples (Nyaya-Instruct-v1), grounded only.

Every example is generated FROM verbatim statute text passed into the prompt
(src/nyaya/generation.py builds the plan; the teacher must never cite from
memory). Output records use the universal TrainingRecord schema with
metadata.source_sections — the grouped-split key — always populated for
grounded slices.

Teacher: any OpenAI-compatible endpoint (configs/generation.yaml). Default is
the in-cluster vLLM serving Gemma 4 31B-IT via
    kubectl port-forward svc/nyaya-teacher 8000:8000 -n askdata-ng

Resumable: completed task_ids are recovered from the output file on restart.
Prints a live citation-gate preview (verify_citations against the statute DB)
— the authoritative gate runs in scripts/05_validate_examples.py.

Usage:
    python scripts/04_generate_examples.py --composition pilot
    python scripts/04_generate_examples.py --composition full
    python scripts/04_generate_examples.py --composition pilot --limit 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyaya.dataset import load_jsonl
from nyaya.generation import build_generation_plan, parse_teacher_response
from nyaya.validators import load_statute_db, verify_citations

CONFIG = ROOT / "configs" / "generation.yaml"
OUT_DIR = ROOT / "data" / "generated"
REPORTS = ROOT / "reports"
RETRIES = 3


def call_teacher(prompt: str, teacher: dict, session: requests.Session) -> str:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(teacher.get("api_key_env", ""), "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": teacher["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": teacher.get("max_tokens", 2048),
        "temperature": teacher.get("temperature", 0.8),
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


def load_statute_rows() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "data" / "canonical").glob("*.jsonl")):
        if path.name == "law_mappings.jsonl":
            continue
        rows.extend(load_jsonl(path))
    return rows


def completed_task_ids(out_file: Path) -> set[str]:
    if not out_file.exists():
        return set()
    done = set()
    for record in load_jsonl(out_file):
        done.add(record["id"].rsplit("_", 1)[0])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--limit", type=int, help="only run the first N pending tasks")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    teacher = config["teacher"]
    version = config["dataset_version"]
    composition = config[f"{args.composition}_composition"]

    statute_rows = load_statute_rows()
    mappings = load_jsonl(ROOT / "data" / "canonical" / "law_mappings.jsonl")
    statute_db = load_statute_db(include_old_law=True)

    plan = build_generation_plan(statute_rows, mappings, composition, seed=config["seed"])
    for task in plan:
        task["generator"] = teacher["model"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{version}_raw.jsonl"
    done = completed_task_ids(out_file)
    pending = [t for t in plan if t["task_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[plan] {len(plan)} tasks | done {len(done)} | running {len(pending)} "
          f"| teacher {teacher['model']} @ {teacher['base_url']}")

    session = requests.Session()
    stats = {"tasks_ok": 0, "tasks_unparseable": 0, "tasks_failed": 0,
             "records": 0, "gate_preview_pass": 0, "gate_preview_fail": 0}
    started = time.time()
    with out_file.open("a", encoding="utf-8") as fh:
        for i, task in enumerate(pending, 1):
            try:
                raw = call_teacher(task["prompt"], teacher, session)
            except RuntimeError as e:
                stats["tasks_failed"] += 1
                print(f"  [{i}/{len(pending)}] {task['task_id']} FAILED: {e}")
                continue
            records = parse_teacher_response(raw, task, version)
            if not records:
                stats["tasks_unparseable"] += 1
                print(f"  [{i}/{len(pending)}] {task['task_id']} unparseable output")
                continue
            for record in records:
                answer = record["messages"][-1]["content"]
                key = ("gate_preview_pass"
                       if verify_citations(answer, statute_db) else "gate_preview_fail")
                stats[key] += 1
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            stats["tasks_ok"] += 1
            stats["records"] += len(records)
            if i % 10 == 0 or i == len(pending):
                rate = stats["gate_preview_pass"] / max(1, stats["records"])
                print(f"  [{i}/{len(pending)}] records={stats['records']} "
                      f"gate-preview pass={rate:.0%} "
                      f"({(time.time() - started) / i:.1f}s/task)", flush=True)

    REPORTS.mkdir(exist_ok=True)
    report = {
        "dataset_version": version,
        "composition": args.composition,
        "teacher": teacher["model"],
        "stats": stats,
        "output": str(out_file.relative_to(ROOT)),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (REPORTS / "generation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] {stats} -> {out_file.relative_to(ROOT)}")
    if stats["records"]:
        pass_rate = stats["gate_preview_pass"] / stats["records"]
        print(f"[gate preview] {pass_rate:.1%} of answers verify against the statute DB "
              f"(authoritative gate: scripts/05_validate_examples.py)")


if __name__ == "__main__":
    main()
