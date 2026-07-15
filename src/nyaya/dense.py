"""Optional dense retrieval stage (A4). Lazily imported — the base package
stays dependency-free; install requirements-dense.txt to enable.

multilingual-e5-small: free, ~470 MB, handles Devanagari (the Hindi/Hinglish
queries pure BM25 struggles with). Measured locally before being enabled
anywhere. e5 models REQUIRE the "query: " / "passage: " prefixes; cosine
scores are meaningless without them.
"""

import hashlib
from pathlib import Path

import numpy as np


class DenseStage:
    """Cosine ranking of statute rows for a query, fused with BM25 by the
    caller (retrieval.rrf_fuse). Row embeddings are computed once and cached
    to disk keyed by (model, corpus) so repeated runs skip re-encoding."""

    def __init__(self, rows: list[dict],
                 model_name: str = "intfloat/multilingual-e5-small",
                 cache_dir: str | Path | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        passages = [
            f"passage: {r.get('title') or ''}. {r.get('text') or ''}" for r in rows
        ]
        fingerprint = hashlib.sha256(
            ("\x00".join([model_name, *passages])).encode("utf-8")).hexdigest()[:16]
        cache_file = (Path(cache_dir) / f"{fingerprint}.npy") if cache_dir else None
        if cache_file and cache_file.exists():
            self.embeddings = np.load(cache_file)
        else:
            self.embeddings = self.model.encode(
                passages, normalize_embeddings=True, show_progress_bar=True)
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, self.embeddings)

    def rank(self, query: str) -> list[int]:
        """Row indices ordered by cosine similarity to the query, best first."""
        q = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        sims = self.embeddings @ q
        return [int(i) for i in sims.argsort()[::-1]]
