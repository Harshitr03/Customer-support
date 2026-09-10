"""Embedding index + numpy cosine retrieval over the corpus KB."""
import functools
import logging
import time

import numpy as np
import pandas as pd

from . import config, data_prep, llm_client

logger = logging.getLogger(__name__)


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def build_index() -> None:
    config.ensure_dirs()
    vp = config.KB_DIR / "kb_vectors.npy"
    mp = config.KB_DIR / "kb_meta.parquet"
    if vp.exists() and mp.exists():
        logger.info("index cache hit, skipping build")
        return
    logger.info("index cache miss, building")
    start = time.monotonic()
    corpus, _ = data_prep.load_pools()
    texts = corpus["customer_open"].tolist()
    vecs = llm_client.embed(texts)
    np.save(vp, _normalize(vecs))
    corpus[["customer_open", "spotify_reply"]].reset_index(drop=True).to_parquet(mp)
    elapsed = time.monotonic() - start
    logger.info(
        "index built: %d texts embedded, shape=%s, elapsed=%.1fs",
        len(texts), vecs.shape, elapsed,
    )
    _load_index.cache_clear()


@functools.lru_cache(maxsize=1)
def _load_index():
    vecs = np.load(config.KB_DIR / "kb_vectors.npy")
    meta = pd.read_parquet(config.KB_DIR / "kb_meta.parquet").to_dict("records")
    return vecs, meta


def retrieve(message: str, k: int = 4) -> list[dict]:
    vecs, meta = _load_index()
    q = _normalize(llm_client.embed([message]))[0]
    scores = vecs @ q
    idx = np.argsort(-scores)[:k]
    results = [{**meta[i], "score": float(scores[i])} for i in idx]
    logger.debug("retrieve: k=%d top_score=%.4f", k, results[0]["score"] if results else float("nan"))
    return results
