"""Gemini wrapper with on-disk caching. Network calls isolated in _raw_*."""
import hashlib
import json
import logging
import os
import time

import numpy as np

from . import config

logger = logging.getLogger(__name__)

_client = None

# Retryable HTTP statuses: 429 (rate limit) and any 5xx (server error).
_RETRYABLE_MAX_ATTEMPTS = 5
_RETRYABLE_BASE_DELAY = 2.0
_RETRYABLE_MAX_DELAY = 60.0


def _get_client():
    global _client
    if _client is None:
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set (see .env.example)")
        _client = genai.Client(api_key=key)
    return _client


def _is_retryable(exc: Exception) -> bool:
    from google.genai import errors as genai_errors
    if isinstance(exc, genai_errors.APIError):
        code = exc.code
        return code == 429 or (code is not None and code >= 500)
    return False


def _retry_with_backoff(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying on rate-limit/server errors with
    exponential backoff. Anything else (e.g. auth errors) is re-raised
    immediately."""
    delay = _RETRYABLE_BASE_DELAY
    for attempt in range(1, _RETRYABLE_MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == _RETRYABLE_MAX_ATTEMPTS or not _is_retryable(exc):
                raise
            code = getattr(exc, "code", None)
            logger.warning(
                "retryable error (code=%s) on attempt %d/%d, retrying in %.1fs",
                code, attempt, _RETRYABLE_MAX_ATTEMPTS, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, _RETRYABLE_MAX_DELAY)


def _cache_path(kind: str, key: str):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:32]
    return config.CACHE_DIR / f"{kind}_{h}.json"


def _raw_generate(prompt: str, temperature: float, model: str, json_mode: bool) -> str:
    from google.genai import types

    cfg = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json" if json_mode else "text/plain",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    def call():
        resp = _get_client().models.generate_content(model=model, contents=prompt, config=cfg)
        return resp.text or ""

    return _retry_with_backoff(call)


def _raw_embed(texts: list[str]) -> np.ndarray:
    from google.genai import types

    cfg = types.EmbedContentConfig(output_dimensionality=config.EMBED_DIM)

    def call():
        resp = _get_client().models.embed_content(model=config.EMBED_MODEL, contents=texts, config=cfg)
        return np.array([e.values for e in resp.embeddings], dtype=np.float32)

    return _retry_with_backoff(call)


def generate(prompt: str, *, json_mode: bool = False, temperature: float = 0.2,
             model: str | None = None) -> str:
    model = model or config.GEN_MODEL
    key = json.dumps({"m": model, "p": prompt, "t": temperature, "j": json_mode})
    path = _cache_path("gen", key)
    if path.exists():
        logger.debug("generate: cache hit (model=%s)", model)
        return json.loads(path.read_text())["text"]
    logger.debug("generate: cache miss (model=%s)", model)
    text = _raw_generate(prompt, temperature, model, json_mode)
    path.write_text(json.dumps({"text": text}))
    return text


def _embed_cache_key(text: str) -> str:
    return f"{config.EMBED_MODEL}::{config.EMBED_DIM}::{text}"


def embed(texts: list[str]) -> np.ndarray:
    out: list[np.ndarray | None] = [None] * len(texts)
    missing_idx, missing_txt = [], []
    for i, t in enumerate(texts):
        p = _cache_path("emb", _embed_cache_key(t))
        if p.exists():
            out[i] = np.array(json.loads(p.read_text()), dtype=np.float32)
        else:
            missing_idx.append(i)
            missing_txt.append(t)

    batch_size = max(1, config.EMBED_BATCH)
    if missing_txt:
        n_batches = (len(missing_txt) + batch_size - 1) // batch_size
        logger.info(
            "embed: %d texts, %d cached, %d to fetch in %d batch(es)",
            len(texts), len(texts) - len(missing_txt), len(missing_txt), n_batches,
        )
    for start in range(0, len(missing_txt), batch_size):
        chunk_idx = missing_idx[start:start + batch_size]
        chunk_txt = missing_txt[start:start + batch_size]
        logger.debug("embed: sending batch of %d texts", len(chunk_txt))
        vecs = _raw_embed(chunk_txt)
        for j, i in enumerate(chunk_idx):
            out[i] = vecs[j]
            _cache_path("emb", _embed_cache_key(texts[i])).write_text(
                json.dumps(vecs[j].tolist()))

    return np.vstack(out)
