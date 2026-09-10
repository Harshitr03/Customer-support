import numpy as np
import pytest

from support_agent import llm_client as lc


@pytest.fixture(autouse=True)
def _no_embed_pacing(monkeypatch):
    """Controller ruling F2 paces real embed batches to config.EMBED_BATCH_
    INTERVAL_S (61s by default), tracked via a process-wide timestamp. Tests
    that don't specifically exercise pacing shouldn't sleep for real or be
    affected by state another test left behind."""
    monkeypatch.setattr(lc.config, "EMBED_BATCH_INTERVAL_S", 0.0)
    monkeypatch.setattr(lc, "_last_embed_batch_at", None)


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


def test_retry_with_backoff_honors_server_retry_delay(monkeypatch):
    """Controller ruling F3: a 429 whose details carry a RetryInfo
    retryDelay must be honored (plus ~1s slack) instead of the exponential
    schedule, so we don't exhaust retries while the quota window is still
    open."""
    from google.genai import errors as genai_errors

    sleeps = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: sleeps.append(s))

    resp_json = {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota...",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": "EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier"}],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "47s"},
            ],
        }
    }

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise genai_errors.ClientError(429, resp_json, None)
        return "ok"

    result = lc._retry_with_backoff(flaky)
    assert result == "ok"
    assert attempts["n"] == 2
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(48.0, abs=0.01)  # 47s + 1s slack


def test_retry_with_backoff_falls_back_to_exponential_without_retry_delay(monkeypatch):
    """An error whose details don't carry a RetryInfo (or has no details at
    all) must keep using the existing exponential backoff schedule."""
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
    assert sleeps == [2.0, 4.0]  # unchanged exponential schedule (base=2, x2)


def test_embed_paces_real_batches_for_free_tier_quota(tmp_path, monkeypatch):
    """Controller ruling F2: separate real embed batches must be spaced at
    least config.EMBED_BATCH_INTERVAL_S apart, tracked process-wide, so a
    100-per-minute free-tier quota isn't exceeded across embed() calls."""
    monkeypatch.setattr(lc.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(lc.config, "EMBED_BATCH", 1)
    monkeypatch.setattr(lc.config, "EMBED_BATCH_INTERVAL_S", 61.0)
    monkeypatch.setattr(lc, "_last_embed_batch_at", None)

    clock = {"t": 1000.0}
    monkeypatch.setattr(lc.time, "monotonic", lambda: clock["t"])

    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        clock["t"] += s

    monkeypatch.setattr(lc.time, "sleep", fake_sleep)

    def fake_raw(texts):
        return np.array([[float(len(t))] for t in texts])

    monkeypatch.setattr(lc, "_raw_embed", fake_raw)

    lc.embed(["a", "b", "c"])  # batch size 1 -> 3 separate real batches

    # First batch goes immediately; batches 2 and 3 each wait ~61s.
    assert len(sleeps) == 2
    assert sleeps[0] == pytest.approx(61.0, abs=0.01)
    assert sleeps[1] == pytest.approx(61.0, abs=0.01)


def test_embed_fully_cached_call_never_paces(tmp_path, monkeypatch):
    """Cached texts must never trigger a pacing wait, even with a real
    previous batch timestamp set and a long interval configured."""
    monkeypatch.setattr(lc.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(lc.config, "EMBED_BATCH_INTERVAL_S", 61.0)

    def fake_raw(texts):
        return np.array([[float(len(t))] for t in texts])

    monkeypatch.setattr(lc, "_raw_embed", fake_raw)

    lc.embed(["x", "y"])  # first (real) call, primes the cache and the pacer

    sleeps = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: sleeps.append(s))
    lc.embed(["x", "y"])  # fully cached: no _raw_embed call, no pacing wait
    assert sleeps == []
