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


def test_dense_cache_path_is_per_model(runner):
    """The e5-base doc vectors live in the shared e5_doc_vectors.npy; any other
    embedder must get its own cache, or its run silently reuses e5-base vectors."""
    default = runner._dense_cache_path(runner.DEFAULT_DENSE_MODEL)
    assert default.name == "e5_doc_vectors.npy"
    other = runner._dense_cache_path("NyayaLabs98/nyaya-embed-v1")
    assert other != default and other.parent == default.parent
    assert other.name == "dense_vectors_NyayaLabs98__nyaya-embed-v1.npy"
    local = runner._dense_cache_path("/kaggle/working/models/nyaya-embed-v1")
    assert local.name == "dense_vectors_nyaya-embed-v1.npy"


class _FakeIndex:
    def retrieve(self, query, k=8):
        return [{"act_id": "bnss_2023", "section": "173", "act_name": "BNSS", "title": "FIR",
                 "text": "Every information relating to a cognizable offence..."}]


def test_endpoint_generate_fn_posts_system_and_rag_prompt(helpers, monkeypatch):
    """The endpoint path must send the same system prompt and RAG prompt the
    in-process path renders, greedy, with the requested token budget."""
    import requests

    seen = []

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "Under Section 173 of the BNSS..."}}]}

    def fake_post(self, url, headers=None, json=None, timeout=None):
        seen.append((url, json))
        return _Resp()

    monkeypatch.setattr(requests.Session, "post", fake_post)
    log = {}
    generate = helpers.build_endpoint_generate_fn(
        "http://127.0.0.1:8000/v1/", "teacher", _FakeIndex(), 8, log, max_new_tokens=512,
        concurrency=2, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    out = generate(["police FIR nahi likh rahi", "what is Section 173 BNSS"])
    assert out == ["Under Section 173 of the BNSS..."] * 2
    assert log["police FIR nahi likh rahi"] == ["bnss_2023:173"]
    url, body = seen[0]
    assert url == "http://127.0.0.1:8000/v1/chat/completions"
    assert body["model"] == "teacher" and body["temperature"] == 0 and body["max_tokens"] == 512
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == helpers.NYAYA_SYSTEM_PROMPT
    assert "Section 173" in body["messages"][1]["content"] and "police FIR" in body["messages"][1]["content"]
