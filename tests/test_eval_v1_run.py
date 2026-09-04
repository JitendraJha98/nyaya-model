"""Pure-logic guards on the Eval-v1 runner (scripts/26) and the RAG helpers it
reuses from scripts/16. No model is loaded."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load("eval_v1_run", "26_eval_v1_run.py")


@pytest.fixture(scope="module")
def helpers():
    return _load("rag_eval_16", "16_rag_eval.py")


def test_results_path_follows_out_dir(runner, tmp_path):
    """An --out-dir run used to scatter: predictions under out_dir, metrics in the repo."""
    assert runner._results_path(tmp_path) == tmp_path / "reports" / "eval_v1_results.json"


class _SystemlessTokenizer:
    """Mimics a Gemma-style template that rejects the system role."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.calls.append([m["role"] for m in messages])
        if any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        return "<user>" + messages[0]["content"] + "<assistant>"


class _NormalTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|assistant:"


def test_chat_text_folds_system_prompt_when_template_rejects_it(helpers):
    tok = _SystemlessTokenizer()
    text = helpers.chat_text(tok, "police FIR nahi likh rahi")
    assert tok.calls == [["system", "user"], ["user"]]
    assert "police FIR nahi likh rahi" in text
    assert "Nyaya" in text, "the system prompt must survive, folded into the user turn"


def test_chat_text_keeps_the_system_role_when_supported(helpers):
    text = helpers.chat_text(_NormalTokenizer(), "hello")
    assert text.startswith("system:You are Nyaya")
    assert "user:hello" in text
