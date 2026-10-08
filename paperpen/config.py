from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class SourceConfig:
    type: str = "webcam"  # "webcam" | "http" | "file"
    device: int = 0  # webcam index (webcam type)
    url: str = ""  # e.g. "http://192.168.1.42:8080/video" (http type)
    path: str = ""  # video file for replay (file type)
    width: int = 1280
    height: int = 720
    fps: int = 30
    reconnect_delay_s: float = 1.0


@dataclass
class PaperConfig:
    warp_width: int = 1240  # A4 @ 150 dpi: precision is ample (0.17 mm/px),
    warp_height: int = 1754  # and it keeps a pipeline step ~23 ms instead of ~100
    detect_interval_s: float = 2.0
    min_area_frac: float = 0.05
    track_flow: bool = True
    min_brightness: float = 90.0  # mean gray inside quad; rejects covered/dark lenses
    width_mm: float = 210.0
    height_mm: float = 297.0


@dataclass
class PenDownConfig:
    enabled: bool = False  # log shadow-gap / new-ink contact signals (M6 experiments)
    shadow_contact_mm: float = 1.5  # tip-shadow gap below this = contact
    band_lo: int = 140  # gray band that shadows occupy on white paper
    band_hi: int = 215
    shadow_search_mm: float = 15.0
    ink_delta: int = 60  # prev - current gray change that counts as fresh ink
    ink_max_gray: int = 160  # fresh ink must actually be dark
    ink_radius_mm: float = 3.0
    baseline_frames: int = 0  # reserved; ink detection uses frame differencing


@dataclass
class TipConfig:
    mode: str = "cv"  # "cv" | "onnx"
    model_path: str = "models/tip.onnx"
    min_area_px: int = 40
    max_area_frac: float = 0.08
    min_aspect: float = 1.8  # fingers grip close to the tip: keep the gate low
    bin_threshold: int = 110
    skin_reject: bool = True
    min_body_width_px: float = 12.0
    min_confidence: float = 0.35
    detect_scale: float = 0.5  # run detection on a scaled warp (speed vs precision)


@dataclass
class FilterConfig:
    min_cutoff: float = 1.0
    beta: float = 0.02
    d_cutoff: float = 1.0
    max_coast_frames: int = 10


@dataclass
class LogConfig:
    dir: str = "data/logs"
    record_video: bool = False
    video_dir: str = "data/videos"
    flush_every: int = 30


@dataclass
class MouseConfig:
    enabled: bool = False
    screen_margin: int = 40
    hover_only: bool = False


@dataclass
class AppConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    tip: TipConfig = field(default_factory=TipConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    log: LogConfig = field(default_factory=LogConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    pendown: PenDownConfig = field(default_factory=PenDownConfig)


def _nested_dataclass(f: fields) -> type | None:
    if isinstance(f.default_factory, type) and is_dataclass(f.default_factory):
        return f.default_factory
    return None


def _build(cls: type[T], data: dict[str, Any] | None) -> T:
    if data is None:
        return cls()
    valid = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"unknown config key {key!r} for {cls.__name__}")
        nested = _nested_dataclass(valid[key])
        if nested is not None:
            if not isinstance(value, dict):
                raise TypeError(f"config key {key!r} must be an object")
            kwargs[key] = _build(nested, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def default_config() -> AppConfig:
    return AppConfig()


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _build(AppConfig, data)


def save_config(cfg: AppConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
