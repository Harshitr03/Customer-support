"""Parse twcs.csv, reconstruct threads, filter to Spotify, sample + split."""
import json
import pandas as pd
from . import config

def _load_raw() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_CSV, dtype={"tweet_id": "Int64"})
    df["inbound"] = df["inbound"].astype(str).str.lower().eq("true")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce",
                                      format="%a %b %d %H:%M:%S %z %Y")
    return df

def reconstruct_threads(df: pd.DataFrame) -> list[dict]:
    by_id = {int(r.tweet_id): r for r in df.itertuples() if pd.notna(r.tweet_id)}
    # find roots: tweets with no in_response_to, or whose parent is absent
    def parent(r):
        p = r.in_response_to_tweet_id
        if pd.isna(p) or p in ("", None):
            return None
        try:
            return int(float(p))
        except (ValueError, TypeError):
            return None
    children: dict[int, list[int]] = {}
    for tid, r in by_id.items():
        p = parent(r)
        if p is not None:
            children.setdefault(p, []).append(tid)
    roots = [tid for tid, r in by_id.items() if parent(r) not in by_id]
    threads = []
    for root in roots:
        # BFS collect the connected component
        seen, stack = set(), [root]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(children.get(n, []))
            pr = parent(by_id[n]) if n in by_id else None
            if pr in by_id and pr not in seen:
                stack.append(pr)
        turns = [by_id[t] for t in seen if t in by_id]
        turns.sort(key=lambda r: (r.created_at is pd.NaT, r.created_at))
        threads.append({
            "root_id": int(root),
            "turns": [{"tweet_id": int(t.tweet_id), "author_id": str(t.author_id),
                       "inbound": bool(t.inbound), "text": str(t.text)} for t in turns],
        })
    return threads

def spotify_threads(threads: list[dict]) -> list[dict]:
    out = []
    for th in threads:
        replies = [t["text"] for t in th["turns"]
                   if not t["inbound"] and t["author_id"] == config.BRAND]
        if not replies:
            continue
        opens = [t["text"] for t in th["turns"] if t["inbound"]]
        if not opens:
            continue
        out.append({"root_id": th["root_id"], "customer_open": opens[0],
                    "spotify_replies": replies, "turns": th["turns"]})
    return out

def build_pool() -> None:
    config.ensure_dirs()
    corpus_p = config.INTERIM_DIR / "corpus_pool.parquet"
    eval_p = config.INTERIM_DIR / "eval_pool.parquet"
    if corpus_p.exists() and eval_p.exists():
        return
    df = _load_raw()
    sp = spotify_threads(reconstruct_threads(df))
    rows = [{"root_id": t["root_id"], "customer_open": t["customer_open"],
             "spotify_reply": t["spotify_replies"][0],
             "turns_json": json.dumps(t["turns"])} for t in sp]
    pool = pd.DataFrame(rows)
    if len(pool) > config.POOL_SIZE:
        pool = pool.sample(config.POOL_SIZE, random_state=config.SEED)
    pool = pool.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
    cut = int(len(pool) * config.CORPUS_FRAC)
    pool.iloc[:cut].to_parquet(corpus_p, index=False)
    pool.iloc[cut:].to_parquet(eval_p, index=False)

def load_pools() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_parquet(config.INTERIM_DIR / "corpus_pool.parquet"),
            pd.read_parquet(config.INTERIM_DIR / "eval_pool.parquet"))
