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

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "twcs" / "twcs.csv"
INTERIM_DIR = DATA_DIR / "interim"
CACHE_DIR = DATA_DIR / "cache"
KB_DIR = INTERIM_DIR / "kb"
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
