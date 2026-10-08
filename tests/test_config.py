import json
from pathlib import Path

import pytest

from paperpen.config import AppConfig, default_config, load_config, save_config


def test_defaults():
    cfg = default_config()
    assert cfg.source.type == "webcam"
    assert cfg.paper.warp_width > 0
    assert cfg.filter.min_cutoff > 0


def test_roundtrip(tmp_path: Path):
    path = tmp_path / "config.json"
    cfg = default_config()
    cfg.source.type = "http"
    cfg.source.url = "http://10.0.0.5:8080/video"
    cfg.tip.bin_threshold = 90
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.source.type == "http"
    assert loaded.source.url == "http://10.0.0.5:8080/video"
    assert loaded.tip.bin_threshold == 90


def test_partial_config(tmp_path: Path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"source": {"type": "http", "url": "http://x/v"}}))
    cfg = load_config(path)
    assert cfg.source.type == "http"
    assert cfg.source.width == AppConfig().source.width


def test_unknown_key_rejected(tmp_path: Path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"sourc": {}}))
    with pytest.raises(KeyError):
        load_config(path)


def test_load_none_is_defaults():
    assert load_config(None).source.type == "webcam"
