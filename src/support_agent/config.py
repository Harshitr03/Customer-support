"""Central config: paths, seeds, model IDs, constants."""
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SEED = 42
POOL_SIZE = 6000
CORPUS_FRAC = 0.90
GOLDEN_SIZE = 200
GOLDEN_MIN_PER_INTENT = 10
BRAND = "SpotifyCares"

GEN_MODEL = "gemini-3.5-flash-lite"
# 3.5 Flash-Lite rejects types.ThinkingConfig(thinking_budget=0) with a 400
# INVALID_ARGUMENT; "low" is the minimum thinking_level it accepts. Passed
# straight to types.ThinkingConfig(**GEN_THINKING) in llm_client._raw_generate.
GEN_THINKING = {"thinking_level": "low"}
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
EMBED_BATCH = 100
# Free-tier embed quota is 100 requests (each input text counts) per minute
# per user/project/model, so pace batches to at most one per this interval.
EMBED_BATCH_INTERVAL_S = 61.0
# Retrieval index size (corpus rows, in load_pools() order). Capped because
# the free-tier embedding quota only covers a growing prefix of the corpus
# (all 5,400 rows are now embedded). Keep this an int (not None) so
# build_index() never blocks on the full corpus.
KB_SIZE = 5400

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "twcs" / "twcs.csv"
INTERIM_DIR = DATA_DIR / "interim"
CACHE_DIR = DATA_DIR / "cache"
# Committed, read-only replay cache: the exact LLM/embedding responses
# behind the reported headline numbers, so graders can reproduce them
# offline with zero API calls (see llm_client's replay fallback and
# scripts/run_demo.py). Never written to by generate()/embed() -- only by
# llm_client.export_touched_cache(), under `run_demo.py --live --export-cache`.
REPLAY_CACHE_DIR = DATA_DIR / "llm_cache"
KB_DIR = DATA_DIR / "kb"
GOLDEN_DIR = DATA_DIR / "golden"
# Committed eval-harness outputs. NOT included in ensure_dirs()'s
# generated-artifact list -- it's created lazily, only by a real eval run
# (eval/run_eval.py writing results), never as a side effect of import or of
# --estimate (which performs no writes at all).
RESULTS_DIR = ROOT / "results"

def ensure_dirs() -> None:
    for d in (INTERIM_DIR, CACHE_DIR, KB_DIR, GOLDEN_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, with a concise format. Idempotent."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
