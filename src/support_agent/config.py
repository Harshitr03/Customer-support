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

GEN_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
EMBED_BATCH = 100
# Free-tier embed quota is 100 requests (each input text counts) per minute
# per user/project/model, so pace batches to at most one per this interval.
EMBED_BATCH_INTERVAL_S = 61.0
# Retrieval index size (corpus rows, in load_pools() order). Capped because
# the free-tier embedding quota only covers a growing prefix of the corpus
# (2,000/5,400 rows embedded so far); raise as more rows get embedded. Keep
# this an int (not None) so build_index() never blocks on the full corpus.
KB_SIZE = 2000

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "twcs" / "twcs.csv"
INTERIM_DIR = DATA_DIR / "interim"
CACHE_DIR = DATA_DIR / "cache"
KB_DIR = DATA_DIR / "kb"
GOLDEN_DIR = DATA_DIR / "golden"

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
