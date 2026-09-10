import numpy as np
from support_agent import retrieve

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
