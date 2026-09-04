"""Step 38 — Fill the reserved statute-DB fields from data the repo already has.

  replaces            <- law_mappings.jsonl (official IPC->BNS, CrPC->BNSS, IEA->BSA
                         tables): list of "IPC 302"-style strings, or null
  punishment_summary  <- the first "shall be punished with ..." clause of the
                         section text (<= 160 chars), or null

Until Sept 2026 both fields were null in all 2,528 statute rows while the
dataset card showed them populated. `tags` stays as it is (curated keywords,
present only on the guidance notes).

Idempotent; rewrites the three 2023 act files in place. Re-run after any
rebuild by scripts/03. Then re-upload the dataset:
    huggingface-cli upload NyayaLabs98/nyaya-statute-db data/canonical . --repo-type dataset --include "*.jsonl"

Usage:
    python scripts/38_enrich_statute_db.py
"""
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "data" / "canonical"
NEW_ACT_TO_ID = {"BNS": "bns_2023", "BNSS": "bnss_2023", "BSA": "bsa_2023"}
_PENALTY = re.compile(r"shall be punish(?:ed|able) with ([^.;]{5,160})", re.IGNORECASE)


def _section_key(s: str) -> tuple:
    m = re.match(r"(\d+)([A-Za-z]*)", s)
    return (int(m.group(1)), m.group(2)) if m else (10**9, s)


def enrich(row: dict, mappings: list[dict]) -> dict:
    out = dict(row)
    olds = sorted(
        {f"{m['old_act']} {m['old_section']}" for m in mappings
         if NEW_ACT_TO_ID.get(m["new_act"]) == row["act_id"]
         and m["new_section"].upper() == row["section"].upper()},
        key=lambda s: _section_key(s.split(" ", 1)[1]))
    out["replaces"] = olds or None
    m = _PENALTY.search(row.get("text") or "")
    out["punishment_summary"] = m.group(1).strip() if m else None
    return out


def main() -> None:
    mappings = [json.loads(l) for l in (CANON / "law_mappings.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for act_id in NEW_ACT_TO_ID.values():
        path = CANON / f"{act_id}.jsonl"
        rows = [enrich(json.loads(l), mappings) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{act_id}: {sum(1 for r in rows if r['replaces'])}/{len(rows)} rows with replaces, "
              f"{sum(1 for r in rows if r['punishment_summary'])} with punishment_summary")
    # other acts: only punishment_summary applies
    for path in sorted(CANON.glob("*.jsonl")):
        if path.stem in NEW_ACT_TO_ID.values() or path.stem in ("law_mappings", "procedures_kb"):
            continue
        rows = [enrich(json.loads(l), []) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        n = sum(1 for r in rows if r["punishment_summary"])
        if n:
            print(f"{path.stem}: {n} rows with punishment_summary")


if __name__ == "__main__":
    main()
