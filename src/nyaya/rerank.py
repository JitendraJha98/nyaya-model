"""Cross-encoder reranking — the second stage the retriever never had.

Why
---
Measured on Nyaya-Eval-v1, accuracy is dominated by whether the right statute
reaches the context window:

    gold statute retrieved   63.2% fact_recall   (n=94)
    gold statute missed      17.1% fact_recall   (n=43)

full_hit@8 is 65.3%, so roughly a third of questions are answered without the
section they need. Fixing that is worth far more than more fine-tuning, which
measured as a statistical tie against the base model.

BM25 and dense embeddings are both *bi-encoders*: query and document are scored
independently, so nothing ever compares them jointly. A cross-encoder reads
(query, passage) together and is much better at "does this passage actually
answer this question" — at a cost that only makes sense on a short list.

So: retrieve wide (top-N by the existing fused ranking), rerank narrow
(top-k by cross-encoder). Exact citation lookups are NOT reranked — when the
user names "Section 103 BNS" that is a resolved fact, not a guess, and no
model score should be allowed to displace it.

Statutes and procedural-guidance rows are reranked separately so the KB
appendix stays additive, matching the contract in StatuteIndex.retrieve.

Usage:
    from nyaya.rerank import CrossEncoderReranker
    index.set_reranker(CrossEncoderReranker())      # default model
    index.retrieve(question, k=8)                   # now reranked
"""

from pathlib import Path

# Multilingual by design: the eval is English/Hindi/Hinglish and Hinglish is
# the weakest retrieval slice (full_hit@8 33%), so an English-only reranker
# would improve the metric while making the hardest case no better.
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

# How deep to pull candidates before reranking. Recall@50 is the ceiling the
# reranker can reach; anything the first stage never surfaced is unreachable.
DEFAULT_DEPTH = 50


def passage_text(row: dict, max_chars: int = 1600) -> str:
    """Render a statute/KB row for the cross-encoder.

    Title carries most of the retrieval signal ("Punishment for murder") and is
    cheap, so it always leads; body text is truncated because cross-encoders
    degrade past their max sequence length and long statutes would otherwise
    push the title out of the window.
    """
    title = (row.get("title") or "").strip()
    body = (row.get("text") or "").strip()
    act = (row.get("act_id") or "").upper()
    section = (row.get("section") or "").upper()
    head = f"{act} Section {section}: {title}".strip()
    return f"{head}\n{body[:max_chars]}"


class CrossEncoderReranker:
    """Scores (query, passage) jointly. Lazy-loads the model on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL, depth: int = DEFAULT_DEPTH,
                 batch_size: int = 16, max_length: int = 512,
                 device: str | None = None):
        self.model_name = model_name
        self.depth = depth
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device).eval()

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance score per passage. Higher is better."""
        if not passages:
            return []
        import torch

        self._load()
        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            chunk = passages[start:start + self.batch_size]
            enc = self._tokenizer([query] * len(chunk), chunk, padding=True,
                                  truncation=True, max_length=self.max_length,
                                  return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            # Cross-encoder rerankers emit a single relevance logit; a
            # 2-class checkpoint would put relevance in column 1.
            col = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
            scores.extend(col.float().cpu().tolist())
        return scores

    def rerank(self, query: str, rows: list[dict], k: int) -> list[dict]:
        """Return the k best rows for the query, best first."""
        if not rows or k <= 0:
            return []
        candidates = rows[: self.depth]
        scored = list(zip(self.score(query, [passage_text(r) for r in candidates]),
                          range(len(candidates))))
        # Sort by score desc, original rank asc — a stable tiebreak keeps the
        # first-stage order when the reranker is indifferent, so a useless
        # reranker degrades to "no change" rather than to noise.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [candidates[i] for _s, i in scored[:k]]


class CachedReranker:
    """Wraps a reranker with an on-disk score cache.

    Recall sweeps re-score the same (query, passage) pairs across k values and
    across runs; on CPU that is the entire cost of the experiment. Cache is
    keyed by model name so switching checkpoints cannot read stale scores.
    """

    def __init__(self, inner: CrossEncoderReranker, cache_path: str | Path):
        self.inner = inner
        self.cache_path = Path(cache_path)
        self.depth = inner.depth
        self._cache: dict[str, float] = {}
        if self.cache_path.exists():
            import json
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except ValueError:
                self._cache = {}
        self._dirty = False

    def _key(self, query: str, passage: str) -> str:
        import hashlib
        raw = f"{self.inner.model_name}\x00{query}\x00{passage}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def score(self, query: str, passages: list[str]) -> list[float]:
        keys = [self._key(query, p) for p in passages]
        missing = [i for i, key in enumerate(keys) if key not in self._cache]
        if missing:
            fresh = self.inner.score(query, [passages[i] for i in missing])
            for i, value in zip(missing, fresh):
                self._cache[keys[i]] = value
            self._dirty = True
        return [self._cache[key] for key in keys]

    def rerank(self, query: str, rows: list[dict], k: int) -> list[dict]:
        if not rows or k <= 0:
            return []
        candidates = rows[: self.depth]
        scored = list(zip(self.score(query, [passage_text(r) for r in candidates]),
                          range(len(candidates))))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [candidates[i] for _s, i in scored[:k]]

    def flush(self) -> None:
        if not self._dirty:
            return
        import json
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        self._dirty = False
