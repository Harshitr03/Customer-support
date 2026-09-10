import numpy as np
import pandas as pd
import pytest

from support_agent import retrieve


@pytest.fixture(autouse=True)
def _clear_index_memo():
    """The lru_cache on _load_index is process-wide; make sure no test
    leaks a memoized index into the next one."""
    retrieve._load_index.cache_clear()
    yield
    retrieve._load_index.cache_clear()


def test_cosine_topk(monkeypatch, tmp_path):
    # 3 KB rows; query closest to row 1
    vecs = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
    meta = [{"customer_open": f"m{i}", "spotify_reply": f"r{i}"} for i in range(3)]
    monkeypatch.setattr(retrieve, "_load_index", lambda: (retrieve._normalize(vecs), meta))
    monkeypatch.setattr(retrieve.llm_client, "embed",
                        lambda ts: np.array([[0.1,0.9,0.0]], dtype=np.float32))
    res = retrieve.retrieve("anything", k=2)
    assert res[0]["spotify_reply"] == "r1"
    assert len(res) == 2
    assert res[0]["score"] >= res[1]["score"]


def test_build_index_invalidates_load_index_cache_after_real_rebuild(monkeypatch, tmp_path):
    """Review finding F1: build_index()'s real-build path must clear the
    _load_index lru_cache, or a retrieve() that ran earlier in the same
    process keeps serving the pre-rebuild index forever."""
    kb_dir = tmp_path / "kb"
    monkeypatch.setattr(retrieve.config, "KB_DIR", kb_dir)
    monkeypatch.setattr(retrieve.config, "ensure_dirs", lambda: kb_dir.mkdir(parents=True, exist_ok=True))

    versions = {
        "v1": {
            "corpus": pd.DataFrame({
                "customer_open": ["alpha", "beta"],
                "spotify_reply": ["reply_alpha", "reply_beta"],
            }),
            "vecs": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        },
        "v2": {
            "corpus": pd.DataFrame({
                "customer_open": ["gamma", "delta"],
                "spotify_reply": ["reply_gamma", "reply_delta"],
            }),
            "vecs": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        },
    }
    state = {"version": "v1"}

    monkeypatch.setattr(
        retrieve.data_prep, "load_pools",
        lambda: (versions[state["version"]]["corpus"], None),
    )

    def fake_embed(texts):
        if len(texts) == 2:  # corpus build call (2 rows in both versions)
            return versions[state["version"]]["vecs"]
        return np.array([[1.0, 0.0]], dtype=np.float32)  # query embedding

    monkeypatch.setattr(retrieve.llm_client, "embed", fake_embed)

    # Build v1 and retrieve() once, so _load_index's memo gets primed with
    # the v1 index (mirroring "retrieve() ran earlier in the same process").
    retrieve.build_index()
    res_v1 = retrieve.retrieve("anything", k=1)
    assert res_v1[0]["spotify_reply"] == "reply_alpha"

    # Simulate a real rebuild from fresh data: remove the on-disk index so
    # build_index() takes the real-build path (not the cache-hit skip).
    (kb_dir / "kb_vectors.npy").unlink()
    (kb_dir / "kb_meta.parquet").unlink()
    state["version"] = "v2"
    retrieve.build_index()

    # Without cache_clear() in build_index()'s real-build path, this would
    # still return the v1 result served from the stale in-memory memo.
    res_v2 = retrieve.retrieve("anything", k=1)
    assert res_v2[0]["spotify_reply"] == "reply_delta"
