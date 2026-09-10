import numpy as np
import pytest

from support_agent import llm_client as lc


def test_generate_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(lc.config, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_raw(prompt, temperature, model, json_mode):
        calls["n"] += 1
        return "hello"

    monkeypatch.setattr(lc, "_raw_generate", fake_raw)
    assert lc.generate("hi") == "hello"
    assert lc.generate("hi") == "hello"  # second call served from cache
    assert calls["n"] == 1


def test_embed_caches_per_text(tmp_path, monkeypatch):
    monkeypatch.setattr(lc.config, "CACHE_DIR", tmp_path)

    def fake_raw(texts):
        return np.array([[float(len(t)), 1.0] for t in texts])

    monkeypatch.setattr(lc, "_raw_embed", fake_raw)
    v1 = lc.embed(["ab", "cde"])
    assert v1.shape == (2, 2)
    v2 = lc.embed(["ab", "cde"])  # fully cached
    assert np.allclose(v1, v2)


def test_embed_chunks_by_embed_batch(tmp_path, monkeypatch):
    """Controller ruling: embed() must send missing texts to _raw_embed in
    chunks of at most config.EMBED_BATCH, caching each chunk as it arrives,
    while keeping output rows aligned with input order."""
    monkeypatch.setattr(lc.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(lc.config, "EMBED_BATCH", 2)

    calls = []

    def fake_raw(texts):
        calls.append(list(texts))
        return np.array([[float(len(t)), float(i)] for i, t in enumerate(texts)])

    monkeypatch.setattr(lc, "_raw_embed", fake_raw)

    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    out = lc.embed(texts)

    assert [len(c) for c in calls] == [2, 2, 1]
    assert out.shape == (5, 2)
    # row alignment: first column of each row equals len(text)
    for i, t in enumerate(texts):
        assert out[i, 0] == float(len(t))

    # a crash mid-way keeps completed chunks: first chunk's texts are now cached
    calls.clear()
    lc.embed(["a", "bb"])
    assert calls == []  # fully served from cache, no new _raw_embed call


def test_retry_with_backoff_retries_then_succeeds(monkeypatch):
    from google.genai import errors as genai_errors

    sleeps = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: sleeps.append(s))

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise genai_errors.ClientError(429, {"message": "rate limited"}, None)
        return "ok"

    result = lc._retry_with_backoff(flaky)
    assert result == "ok"
    assert attempts["n"] == 3
    assert len(sleeps) == 2  # slept between the two failed attempts


def test_retry_with_backoff_logs_warning_on_retry(monkeypatch, caplog):
    from google.genai import errors as genai_errors

    monkeypatch.setattr(lc.time, "sleep", lambda s: None)

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise genai_errors.ClientError(429, {"message": "rate limited"}, None)
        return "ok"

    with caplog.at_level("WARNING", logger=lc.logger.name):
        result = lc._retry_with_backoff(flaky)
    assert result == "ok"
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_retry_with_backoff_raises_non_retryable_immediately(monkeypatch):
    from google.genai import errors as genai_errors

    sleeps = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: sleeps.append(s))

    attempts = {"n": 0}

    def broken():
        attempts["n"] += 1
        raise genai_errors.ClientError(401, {"message": "bad auth"}, None)

    with pytest.raises(genai_errors.ClientError):
        lc._retry_with_backoff(broken)
    assert attempts["n"] == 1
    assert sleeps == []
