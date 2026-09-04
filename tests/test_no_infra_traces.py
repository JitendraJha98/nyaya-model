"""v1-v4 were trained on a private cluster that no longer exists. Nothing in
the tree may reference it: the manifests were deleted in Sept 2026 and every
remaining mention of that environment was reworded to be hardware-generic."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(
    r"askdata-ng|azurecr\.io|kubectl|\bA100\b|/pvc/|hf-model-cache"
    r"|inference-server|nyaya-teacher",
    re.IGNORECASE,
)
PATTERNS = ("*.py", "*.yaml", "*.yml", "*.md", "*.txt", "*.toml", "*.ipynb", "*.json", "*.cfg")
# docs/plans/ is skipped because the plan that removed these references has to
# name them; it moves to docs/archive/ once executed.
SKIP_PREFIXES = ("docs/archive/", "docs/plans/", "outputs/", "data/", "reports/eval_v1_kaggle_run.log")


def _scan_files():
    for pattern in PATTERNS:
        for path in ROOT.rglob(pattern):
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(".git/") or rel.startswith(SKIP_PREFIXES) or "/.venv" in f"/{rel}":
                continue
            if path.name == Path(__file__).name:
                continue
            yield path, rel


def test_no_decommissioned_cluster_references():
    hits = []
    for path, rel in _scan_files():
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if FORBIDDEN.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()[:100]}")
    assert not hits, "references to the decommissioned cluster:\n" + "\n".join(hits)


def test_no_kubernetes_manifests_or_cluster_dockerfile():
    assert not (ROOT / "k8s").exists()
    assert not (ROOT / "Dockerfile").exists()
