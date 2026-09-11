import logging

from support_agent import config

def test_config_constants():
    assert config.SEED == 42
    assert config.POOL_SIZE == 6000
    assert config.GEN_MODEL == "gemini-3.5-flash-lite"
    assert config.GEN_THINKING == {"thinking_level": "low"}
    assert config.EMBED_MODEL == "gemini-embedding-001"
    assert config.RAW_CSV.name == "twcs.csv"

def test_ensure_dirs_creates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTERIM_DIR", tmp_path / "interim")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "KB_DIR", tmp_path / "kb")
    config.ensure_dirs()
    assert (tmp_path / "interim").is_dir()
    assert (tmp_path / "cache").is_dir()

def test_setup_logging_runs():
    config.setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
