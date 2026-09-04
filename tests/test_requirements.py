"""The code calls APIs that only exist above these versions (audit, Sept 2026):
from_pretrained(dtype=) needs transformers>=4.56, SFTTrainer(processing_class=)
needs trl>=0.12, torch.cuda.OutOfMemoryError needs torch>=2.5. The floors used to
sit below all three, so a fresh install at the floors could not run the code."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOORS = {"transformers": (4, 56), "trl": (0, 12), "torch": (2, 5)}


def _floor(text: str, pkg: str):
    m = re.search(rf"^{pkg}>=(\d+)\.(\d+)", text, re.MULTILINE)
    assert m, f"{pkg} has no >= floor"
    return int(m.group(1)), int(m.group(2))


def test_training_floors_match_the_apis_the_code_uses():
    text = (ROOT / "requirements-train.txt").read_text(encoding="utf-8")
    for pkg, floor in FLOORS.items():
        assert _floor(text, pkg) >= floor, f"{pkg} floor below {floor}"


def test_core_requirements_do_not_pull_the_training_stack():
    core = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for heavy in ("torch", "trl", "wandb", "gradio", "boto3", "transformers"):
        assert not re.search(rf"^{heavy}\b", core, re.MULTILINE), f"{heavy} belongs in an extra"


def test_oom_handler_uses_the_portable_exception_name():
    src = (ROOT / "scripts" / "26_eval_v1_run.py").read_text(encoding="utf-8")
    assert "torch.cuda.OutOfMemoryError" in src
    assert "except torch.OutOfMemoryError" not in src


def test_every_from_pretrained_uses_the_dtype_keyword():
    """transformers 4.56 renamed torch_dtype -> dtype; the tree must use one name."""
    offenders = []
    for path in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "src" / "nyaya").glob("*.py")) + [ROOT / "app.py"]:
        if "torch_dtype=" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert not offenders, f"still using torch_dtype=: {offenders}"
