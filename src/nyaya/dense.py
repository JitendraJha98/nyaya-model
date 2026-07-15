"""Dense embedding stage for hybrid retrieval (optional dependency).

sentence-transformers is imported lazily so the core retrieval module and the
test suite never require it. E5-family models need the "query: "/"passage: "
prefixes — that contract lives here, not in callers.
"""

from pathlib import Path

DEFAULT_MODEL = "intfloat/multilingual-e5-base"


def attach_dense_index(index, cache_path: str | Path | None = None,
                       model_name: str = DEFAULT_MODEL, device: str | None = None):
    """Enable hybrid retrieval on a StatuteIndex.

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
