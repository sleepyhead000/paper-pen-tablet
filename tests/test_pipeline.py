import csv
import time
from pathlib import Path

import cv2
import numpy as np

from paperpen.capture.source import Frame
from paperpen.config import AppConfig, LogConfig, PaperConfig
from paperpen.output import CSV_HEADER, SessionLogger
from paperpen.pipeline import Pipeline, draw_overlay

# Portrait quad (~A4 aspect) so the warp stays near-isotropic like a real
# top-down camera setup; a landscape quad would be stretched 2x+ anisotropically.
QUAD = np.array([[175, 30], [475, 45], [495, 455], [155, 440]], np.float32)


def scene(tip=(250, 200), angle=30, with_pen=True, with_paper=True):
    img = np.full((480, 640, 3), 40, np.uint8)
    if with_paper:
        cv2.fillConvexPoly(img, QUAD.astype(np.int32), (238, 238, 238))
        cv2.line(img, (185, 385), (470, 405), (30, 30, 30), 4)  # old ink
    if with_pen:
        a = np.radians(angle)
        axis = np.array([np.cos(a), np.sin(a)])
        perp = np.array([-axis[1], axis[0]])
        t = np.asarray(tip, np.float64)
        base = t + axis * 24
        back = t + axis * 190
        poly = np.array(
            [t, base + perp * 11, back + perp * 11, back - perp * 11, base - perp * 11],
            np.int32,
        )
        cv2.fillConvexPoly(img, poly, (30, 30, 36))
    return img


def make_cfg(tmp_path: Path | None = None) -> AppConfig:
    cfg = AppConfig()
    cfg.paper = PaperConfig(warp_width=600, warp_height=800, detect_interval_s=10.0)
    cfg.log = LogConfig(dir=str(tmp_path or Path("data/logs")))
    return cfg


def test_pipeline_tip_roundtrip():
    cfg = make_cfg()
    pipe = Pipeline(cfg)
    frame = Frame(scene(tip=(250, 200)), time.monotonic(), 0)
    r = pipe.step(frame)
    assert r.paper is not None
    assert r.warped is not None
    assert r.tip is not None
    assert r.paper_xy is not None
    assert r.tip_frame is not None
    err = np.hypot(r.tip_frame[0] - 250, r.tip_frame[1] - 200)
    assert err < 15.0, f"frame-space tip off by {err:.1f}px"
    assert 0.0 <= r.paper_xy[0] <= 1.0 and 0.0 <= r.paper_xy[1] <= 1.0


def test_pipeline_no_paper():
    cfg = make_cfg()
    pipe = Pipeline(cfg)
    r = pipe.step(Frame(np.full((480, 640, 3), 40, np.uint8), time.monotonic(), 0))
    assert r.paper is None
    assert r.tip is None
    assert r.tip_frame is None
    assert r.paper_xy is None


def test_pipeline_logs_csv(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    logger = SessionLogger(cfg.log, session_name="t")
    pipe = Pipeline(cfg, logger)
    pipe.step(Frame(scene(), time.monotonic(), 0))
    pipe.step(Frame(scene(with_pen=False), time.monotonic(), 1))
    pipe.step(Frame(np.full((480, 640, 3), 40, np.uint8), time.monotonic(), 2))
    logger.close()

    with logger.csv_path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == CSV_HEADER
    assert len(rows) == 4  # header + 3
    with_pen = rows[1]
    assert with_pen[1] == "0"
    assert with_pen[2] and with_pen[3]  # tip px present
    assert with_pen[6] in ("0", "1")
    assert float(with_pen[7]) > 0.0
    no_pen = rows[2]
    assert no_pen[2] == "" and no_pen[3] == ""  # no tip
    no_paper = rows[3]
    assert no_paper[2] == "" and no_paper[4] == ""
    assert no_paper[6] == "0"  # pen_down default

    homography = logger.csv_path.with_suffix(".homography.csv")
    assert homography.exists()
    with homography.open(newline="") as fh:
        h_rows = list(csv.reader(fh))
    assert len(h_rows) == 3  # header + 2 frames with paper


def test_logger_video_record(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    cfg.log.record_video = True
    logger = SessionLogger(cfg.log, session_name="vid")
    img = scene()
    pipe = Pipeline(cfg, logger)
    for i in range(3):
        pipe.step(Frame(img, time.monotonic(), i))
    logger.close()
    assert logger.csv_path.exists()
    mp4 = logger.csv_path.with_suffix(".mp4")
    assert mp4.exists() and mp4.stat().st_size > 0


def test_draw_overlay():
    cfg = make_cfg()
    pipe = Pipeline(cfg)
    img = scene()
    r = pipe.step(Frame(img, time.monotonic(), 0))
    out = draw_overlay(img.copy(), r)
    assert out.shape == img.shape
    # overlay must not touch the input image
    assert not np.array_equal(out, img)


def test_pipeline_with_video_source(tmp_path: Path):
    path = tmp_path / "in.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    for _ in range(5):
        vw.write(scene())
    vw.release()

    from paperpen.capture import VideoFileSource

    cfg = make_cfg(tmp_path)
    logger = SessionLogger(cfg.log, session_name="v")
    pipe = Pipeline(cfg, logger)
    n = 0
    with VideoFileSource(str(path)) as src:
        while True:
            f = src.next_frame()
            if f is None:
                break
            r = pipe.step(f)
            if r.tip is not None:
                n += 1
    logger.close()
    assert n >= 4
    assert logger.rows == 5
