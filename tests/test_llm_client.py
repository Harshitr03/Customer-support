import json

import numpy as np
import pytest

from support_agent import llm_client as lc


@pytest.fixture(autouse=True)
def _no_embed_pacing(monkeypatch):
    """Real embed batches are paced to config.EMBED_BATCH_
    INTERVAL_S (61s by default), tracked via a process-wide timestamp. Tests
    that don't specifically exercise pacing shouldn't sleep for real or be
    affected by state another test left behind."""
    monkeypatch.setattr(lc.config, "EMBED_BATCH_INTERVAL_S", 0.0)
    monkeypatch.setattr(lc, "_last_embed_batch_at", None)


@pytest.fixture(autouse=True)
def _reset_touched_files(monkeypatch):
    """touched_cache_files() is a module-level set; don't let one
    test's reads/writes bleed into the next."""
    monkeypatch.setattr(lc, "_touched_files", set())


@pytest.fixture(autouse=True)
def _offline_env_isolated(monkeypatch):
    """SUPPORT_AGENT_OFFLINE must never leak between tests. Start
    every test with it unset; monkeypatch restores whatever it was
    (including unset) after the test regardless of what ran in between."""
    monkeypatch.delenv("SUPPORT_AGENT_OFFLINE", raising=False)


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
    """embed() must send missing texts to _raw_embed in
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
    """A 429 whose details carry a RetryInfo
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


def test_retry_with_backoff_raises_quota_exhausted_immediately_on_perday_429(monkeypatch):
    """A 429 whose QuotaFailure details carry a quotaId containing
    "PerDay" must raise QuotaExhaustedError at once -- no sleep, no further
    attempts -- because retrying within the same day can't help."""
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
                    "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
                },
            ],
        }
    }

    attempts = {"n": 0}

    def always_daily_quota():
        attempts["n"] += 1
        raise genai_errors.ClientError(429, resp_json, None)

    with pytest.raises(lc.QuotaExhaustedError) as exc_info:
        lc._retry_with_backoff(always_daily_quota)

    assert attempts["n"] == 1
    assert sleeps == []
    msg = str(exc_info.value)
    assert "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in msg
    assert "midnight" in msg.lower() and "pacific" in msg.lower()
    assert "cache" in msg.lower()


def test_retry_with_backoff_still_retries_on_perminute_429(monkeypatch):
    """A 429 for a PER-MINUTE quota (no "PerDay" in the quotaId) must
    keep using the normal retry path, not QuotaExhaustedError."""
    from google.genai import errors as genai_errors

    monkeypatch.setattr(lc.time, "sleep", lambda s: None)

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


def test_embed_paces_real_batches_for_free_tier_quota(tmp_path, monkeypatch):
    """Separate real embed batches must be spaced at
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


# ---------------------------------------------------------------------------
# Replay cache, offline guard, touched-files tracking, export
# ---------------------------------------------------------------------------

def test_generate_replay_fallback_no_network_and_replay_readonly(tmp_path, monkeypatch):
    cache_dir, replay_dir = tmp_path / "cache", tmp_path / "replay"
    cache_dir.mkdir()
    replay_dir.mkdir()
    monkeypatch.setattr(lc.config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(lc.config, "REPLAY_CACHE_DIR", replay_dir)

    key = json.dumps({"m": lc.config.GEN_MODEL, "p": "hi", "t": 0.2, "j": False})
    path = lc._cache_path("gen", key)
    (replay_dir / path.name).write_text(json.dumps({"text": "replayed"}))

    def fail_raw(*a, **k):
        raise AssertionError("must not call the network on a replay hit")

    monkeypatch.setattr(lc, "_raw_generate", fail_raw)

    before = set(p.name for p in replay_dir.iterdir())
    assert lc.generate("hi") == "replayed"
    assert set(p.name for p in replay_dir.iterdir()) == before  # replay dir untouched
    assert not path.exists()  # nothing written into CACHE_DIR either


def test_embed_replay_fallback_no_network(tmp_path, monkeypatch):
    cache_dir, replay_dir = tmp_path / "cache", tmp_path / "replay"
    cache_dir.mkdir()
    replay_dir.mkdir()
    monkeypatch.setattr(lc.config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(lc.config, "REPLAY_CACHE_DIR", replay_dir)

    key = lc._embed_cache_key("hello")
    path = lc._cache_path("emb", key)
    (replay_dir / path.name).write_text(json.dumps([1.0, 2.0, 3.0]))

    def fail_raw(texts):
        raise AssertionError("must not call the network on a replay hit")

    monkeypatch.setattr(lc, "_raw_embed", fail_raw)

    vecs = lc.embed(["hello"])
    assert np.allclose(vecs[0], [1.0, 2.0, 3.0])
    assert not path.exists()


def test_generate_offline_guard_raises_clear_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setattr(lc.config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(lc.config, "REPLAY_CACHE_DIR", tmp_path / "replay")
    monkeypatch.setenv("SUPPORT_AGENT_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="offline mode"):
        lc.generate("uncached prompt")


def test_embed_offline_guard_raises_clear_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setattr(lc.config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(lc.config, "REPLAY_CACHE_DIR", tmp_path / "replay")
    monkeypatch.setenv("SUPPORT_AGENT_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="offline mode"):
        lc.embed(["uncached text"])


def test_offline_guard_does_not_block_cache_hits(tmp_path, monkeypatch):
    """Offline mode only guards the network fallback -- a value already in
    CACHE_DIR or REPLAY_CACHE_DIR must still be served."""
    cache_dir, replay_dir = tmp_path / "cache", tmp_path / "replay"
    cache_dir.mkdir()
    replay_dir.mkdir()
    monkeypatch.setattr(lc.config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(lc.config, "REPLAY_CACHE_DIR", replay_dir)

    key = json.dumps({"m": lc.config.GEN_MODEL, "p": "hi", "t": 0.2, "j": False})
    path = lc._cache_path("gen", key)
    (replay_dir / path.name).write_text(json.dumps({"text": "replayed"}))

    monkeypatch.setenv("SUPPORT_AGENT_OFFLINE", "1")
    assert lc.generate("hi") == "replayed"


def test_touched_cache_files_records_writes_and_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(lc.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(lc.config, "REPLAY_CACHE_DIR", tmp_path / "replay")
    monkeypatch.setattr(lc, "_raw_generate", lambda *a, **k: "hello")

    assert lc.touched_cache_files() == []
    lc.generate("hi")  # cache miss -> write
    touched_after_write = set(lc.touched_cache_files())
    assert len(touched_after_write) == 1

    monkeypatch.setattr(lc, "_touched_files", set())  # simulate a fresh process
    lc.generate("hi")  # now a cache hit
    assert set(lc.touched_cache_files()) == touched_after_write


def test_generate_does_not_cache_empty_response(tmp_path, monkeypatch, caplog):
    """An empty or whitespace-only generate() response must never be
    written to disk -- caching it would make every future call for that
    same prompt replay the empty string forever instead of retrying."""
    monkeypatch.setattr(lc.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(lc, "_raw_generate", lambda *a, **k: "   ")

    with caplog.at_level("WARNING", logger=lc.logger.name):
        result = lc.generate("hi")

    assert result == "   "
    assert any(r.levelname == "WARNING" for r in caplog.records)
    assert list(tmp_path.iterdir()) == []  # nothing written to CACHE_DIR


def test_generate_empty_response_is_retried_on_next_call(tmp_path, monkeypatch):
    monkeypatch.setattr(lc.config, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_raw(*a, **k):
        calls["n"] += 1
        return "" if calls["n"] == 1 else "real text"

    monkeypatch.setattr(lc, "_raw_generate", fake_raw)
    assert lc.generate("hi") == ""
    assert lc.generate("hi") == "real text"  # not served from a bogus empty cache entry
    assert calls["n"] == 2


def test_generate_ignores_replay_cache_when_no_replay_env_set(tmp_path, monkeypatch):
    """SUPPORT_AGENT_NO_REPLAY=1 must disable the replay-cache lookup so
    --live --no-replay genuinely calls the network for anything not in the
    local disk cache, even if a replay entry exists."""
    cache_dir, replay_dir = tmp_path / "cache", tmp_path / "replay"
    cache_dir.mkdir()
    replay_dir.mkdir()
    monkeypatch.setattr(lc.config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(lc.config, "REPLAY_CACHE_DIR", replay_dir)
    monkeypatch.setenv("SUPPORT_AGENT_NO_REPLAY", "1")

    key = json.dumps({"m": lc.config.GEN_MODEL, "p": "hi", "t": 0.2, "j": False})
    path = lc._cache_path("gen", key)
    (replay_dir / path.name).write_text(json.dumps({"text": "stale replay"}))

    monkeypatch.setattr(lc, "_raw_generate", lambda *a, **k: "fresh from network")

    assert lc.generate("hi") == "fresh from network"


def test_embed_ignores_replay_cache_when_no_replay_env_set(tmp_path, monkeypatch):
    cache_dir, replay_dir = tmp_path / "cache", tmp_path / "replay"
    cache_dir.mkdir()
    replay_dir.mkdir()
    monkeypatch.setattr(lc.config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(lc.config, "REPLAY_CACHE_DIR", replay_dir)
    monkeypatch.setenv("SUPPORT_AGENT_NO_REPLAY", "1")

    key = lc._embed_cache_key("hello")
    path = lc._cache_path("emb", key)
    (replay_dir / path.name).write_text(json.dumps([9.0, 9.0, 9.0]))

    monkeypatch.setattr(lc, "_raw_embed", lambda texts: np.array([[1.0, 2.0, 3.0]]))

    out = lc.embed(["hello"])
    assert np.allclose(out, [[1.0, 2.0, 3.0]])


def test_export_touched_cache_writes_new_files(tmp_path, monkeypatch):
    cache_dir, replay_dir = tmp_path / "cache", tmp_path / "replay"
    cache_dir.mkdir()
    replay_dir.mkdir()
    monkeypatch.setattr(lc.config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(lc.config, "REPLAY_CACHE_DIR", replay_dir)

    (cache_dir / "gen_aaa.json").write_text('{"text": "a"}')
    monkeypatch.setattr(lc, "_touched_files", {"gen_aaa.json"})

    n_copied, n_bytes = lc.export_touched_cache()

    assert n_copied == 1
    assert (replay_dir / "gen_aaa.json").read_text() == '{"text": "a"}'
    assert n_bytes == len((cache_dir / "gen_aaa.json").read_bytes())


def test_export_touched_cache_overwrites_stale_files_and_skips_identical(tmp_path, monkeypatch, caplog):
    """A replay file must be refreshed when the local cache's bytes for
    that same filename have changed (e.g. the prompt template or model
    changed since the replay cache was committed) -- silently keeping the
    stale replay file forever would make offline runs replay outdated
    responses. An identical replay file is left untouched (skip, don't
    rewrite for no reason)."""
    cache_dir, replay_dir = tmp_path / "cache", tmp_path / "replay"
    cache_dir.mkdir()
    replay_dir.mkdir()
    monkeypatch.setattr(lc.config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(lc.config, "REPLAY_CACHE_DIR", replay_dir)

    (cache_dir / "gen_stale.json").write_text('{"text": "new value"}')
    (replay_dir / "gen_stale.json").write_text('{"text": "old stale value"}')
    (cache_dir / "gen_same.json").write_text('{"text": "unchanged"}')
    (replay_dir / "gen_same.json").write_text('{"text": "unchanged"}')

    monkeypatch.setattr(lc, "_touched_files", {"gen_stale.json", "gen_same.json"})

    with caplog.at_level("INFO", logger=lc.logger.name):
        n_copied, n_bytes = lc.export_touched_cache()

    # only the stale (differing) file was actually written
    assert n_copied == 1
    assert (replay_dir / "gen_stale.json").read_text() == '{"text": "new value"}'
    assert (replay_dir / "gen_same.json").read_text() == '{"text": "unchanged"}'
    assert n_bytes == len((cache_dir / "gen_stale.json").read_bytes())

    # counts of new/updated/unchanged are logged
    messages = " ".join(r.message for r in caplog.records)
    assert "1 updated" in messages or "updated=1" in messages
    assert "1 unchanged" in messages or "unchanged=1" in messages
    assert "0 new" in messages or "new=0" in messages
