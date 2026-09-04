"""Optional dense retrieval stage (lazily imported — the base package stays
dependency-free; install the dense extra (pip install -e ".[dense]") to enable).

Two entry points, both setting StatuteIndex.dense so retrieve() fuses the
cosine ranking with BM25 via RRF:

- DenseStage(rows, model_name): built by load_statute_index(dense_model=...);
  content-fingerprinted on-disk embedding cache. Default e5-small (~470 MB,
  CPU-friendly, handles the Devanagari queries pure BM25 struggles with).
- attach_dense_index(index, cache_path): the GPU-job path — model reused for
  batch query embedding, doc vectors cached to a named .npy. Default e5-base, the model behind the committed frozen-eval
  recall numbers (reports/retrieval_recall_dense.json).

e5 models REQUIRE the "query: " / "passage: " prefixes; cosine scores are
meaningless without them. That contract lives here, not in callers.
"""

import hashlib
from pathlib import Path

DEFAULT_ATTACH_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_STAGE_MODEL = "intfloat/multilingual-e5-small"


class DenseStage:
    """Cosine ranking of statute rows for a query, fused with BM25 by the
    caller (retrieval.rrf_fuse). Row embeddings are computed once and cached
    to disk keyed by (model, corpus) so repeated runs skip re-encoding."""

    def __init__(self, rows: list[dict],
                 model_name: str = DEFAULT_STAGE_MODEL,
                 cache_dir: str | Path | None = None):
        import numpy as np
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


def attach_dense_index(index, cache_path: str | Path | None = None,
                       model_name: str = DEFAULT_ATTACH_MODEL,
                       device: str | None = None):
    """Enable hybrid retrieval on a StatuteIndex via index.add_dense.

    Doc vectors load from cache_path (.npy) when present, otherwise they are
    computed and cached there. Returns the SentenceTransformer so callers can
    reuse it (e.g. for batch query embedding).
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def embed_queries(texts):
        return model.encode([f"query: {t}" for t in texts],
                            batch_size=32, normalize_embeddings=True).tolist()

    doc_vectors = None
    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.exists():
        doc_vectors = np.load(cache)
        if len(doc_vectors) != len(index.rows):
            doc_vectors = None  # statute DB changed — recompute
    if doc_vectors is None:
        docs = [f"passage: {r['act_name']} — {r.get('title') or ''}. "
                f"{r.get('text') or ''}" for r in index.rows]
        doc_vectors = model.encode(docs, batch_size=32, normalize_embeddings=True)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, doc_vectors)

    index.add_dense(embed_queries, doc_vectors=doc_vectors.tolist())
    return model
