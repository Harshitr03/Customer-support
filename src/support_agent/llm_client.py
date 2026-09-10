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
_RETRY_DELAY_SLACK_S = 1.0
_RETRY_DELAY_CAP_S = 120.0

# Process-wide timestamp (time.monotonic()) of when the last real embed
# batch was sent, so free-tier quota pacing applies across separate embed()
# calls too, not just within one.
_last_embed_batch_at: float | None = None

# Task 13: every cache filename this process has read (cache or replay hit)
# or written, for `touched_cache_files()` / `export_touched_cache()`.
_touched_files: set[str] = set()


class OfflineModeError(RuntimeError):
    """Raised by _raw_generate/_raw_embed when SUPPORT_AGENT_OFFLINE=1 and
    no cached or replayed response exists for the call, instead of ever
    touching the network."""


def _check_offline_guard() -> None:
    """Checked inside the raw network functions themselves (not generate()/
    embed()) so it can't be bypassed by any caching path."""
    if os.environ.get("SUPPORT_AGENT_OFFLINE") == "1":
        raise OfflineModeError(
            "offline mode: no cached response for this call — run with --live"
        )


def touched_cache_files() -> list[str]:
    """Every cache filename this process has read (cache or replay hit) or
    written to config.CACHE_DIR, for --export-cache."""
    return sorted(_touched_files)


def export_touched_cache() -> tuple[int, int]:
    """Copy every file in touched_cache_files() from config.CACHE_DIR into
    config.REPLAY_CACHE_DIR, skipping files already present there. The only
    place that writes into REPLAY_CACHE_DIR -- generate()/embed() never do.
    Returns (n_files_copied, n_bytes_copied)."""
    config.REPLAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    n_bytes = 0
    for name in touched_cache_files():
        src = config.CACHE_DIR / name
        dst = config.REPLAY_CACHE_DIR / name
        if dst.exists() or not src.exists():
            continue
        data = src.read_bytes()
        dst.write_bytes(data)
        n_copied += 1
        n_bytes += len(data)
    return n_copied, n_bytes


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


def _parse_retry_delay(exc: Exception) -> float | None:
    """Extract the server-suggested retry delay (seconds) from an APIError's
    details, if present. The genai SDK exposes the raw error body as
    `exc.details`, e.g. `{'error': {..., 'details': [..., {'@type':
    '.../RetryInfo', 'retryDelay': '47s'}]}}` (some responses omit the outer
    'error' wrapper). Returns None if absent or unparseable."""
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    err = details.get("error", details)
    if not isinstance(err, dict):
        return None
    sub_details = err.get("details")
    if not isinstance(sub_details, list):
        return None
    for d in sub_details:
        if not isinstance(d, dict):
            continue
        if not str(d.get("@type", "")).endswith("RetryInfo"):
            continue
        delay = d.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


def _retry_with_backoff(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying on rate-limit/server errors.
    When the error carries a server-suggested retryDelay (RetryInfo detail),
    sleep that long (plus a small slack, capped) instead of guessing;
    otherwise fall back to exponential backoff. Anything else (e.g. auth
    errors) is re-raised immediately."""
    delay = _RETRYABLE_BASE_DELAY
    for attempt in range(1, _RETRYABLE_MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == _RETRYABLE_MAX_ATTEMPTS or not _is_retryable(exc):
                raise
            code = getattr(exc, "code", None)
            retry_delay = _parse_retry_delay(exc)
            if retry_delay is not None:
                wait = min(retry_delay + _RETRY_DELAY_SLACK_S, _RETRY_DELAY_CAP_S)
                logger.warning(
                    "retryable error (code=%s) on attempt %d/%d, server "
                    "requested retryDelay=%.1fs, retrying in %.1fs",
                    code, attempt, _RETRYABLE_MAX_ATTEMPTS, retry_delay, wait,
                )
            else:
                wait = delay
                logger.warning(
                    "retryable error (code=%s) on attempt %d/%d, retrying in %.1fs",
                    code, attempt, _RETRYABLE_MAX_ATTEMPTS, wait,
                )
                delay = min(delay * 2, _RETRYABLE_MAX_DELAY)
            time.sleep(wait)


def _cache_path(kind: str, key: str):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:32]
    return config.CACHE_DIR / f"{kind}_{h}.json"


def _raw_generate(prompt: str, temperature: float, model: str, json_mode: bool) -> str:
    _check_offline_guard()
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
    _check_offline_guard()
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
        _touched_files.add(path.name)
        return json.loads(path.read_text())["text"]
    replay_path = config.REPLAY_CACHE_DIR / path.name
    if replay_path.exists():
        logger.debug("generate: replay cache hit (model=%s)", model)
        _touched_files.add(path.name)
        return json.loads(replay_path.read_text())["text"]
    logger.debug("generate: cache miss (model=%s)", model)
    text = _raw_generate(prompt, temperature, model, json_mode)
    path.write_text(json.dumps({"text": text}))
    _touched_files.add(path.name)
    return text


def _embed_cache_key(text: str) -> str:
    return f"{config.EMBED_MODEL}::{config.EMBED_DIM}::{text}"


def _pace_embed_batch() -> None:
    """Block until at least config.EMBED_BATCH_INTERVAL_S seconds have
    passed since the previous real embed batch was sent (process-wide), to
    stay under the free-tier per-minute embed quota. The first batch in a
    process goes immediately."""
    global _last_embed_batch_at
    now = time.monotonic()
    if _last_embed_batch_at is not None:
        wait = config.EMBED_BATCH_INTERVAL_S - (now - _last_embed_batch_at)
        if wait > 0:
            logger.info(
                "pacing: waiting %.0fs before next embed batch (free-tier quota)",
                wait,
            )
            time.sleep(wait)
    _last_embed_batch_at = time.monotonic()


def embed(texts: list[str]) -> np.ndarray:
    out: list[np.ndarray | None] = [None] * len(texts)
    missing_idx, missing_txt = [], []
    for i, t in enumerate(texts):
        p = _cache_path("emb", _embed_cache_key(t))
        if p.exists():
            out[i] = np.array(json.loads(p.read_text()), dtype=np.float32)
            _touched_files.add(p.name)
            continue
        replay_p = config.REPLAY_CACHE_DIR / p.name
        if replay_p.exists():
            out[i] = np.array(json.loads(replay_p.read_text()), dtype=np.float32)
            _touched_files.add(p.name)
            continue
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
        _pace_embed_batch()
        vecs = _raw_embed(chunk_txt)
        for j, i in enumerate(chunk_idx):
            out[i] = vecs[j]
            cache_p = _cache_path("emb", _embed_cache_key(texts[i]))
            cache_p.write_text(json.dumps(vecs[j].tolist()))
            _touched_files.add(cache_p.name)

    return np.vstack(out)
