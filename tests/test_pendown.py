import csv
from pathlib import Path

from paperpen.evaluate.pendown import score_pen_down

PD_HEADER = [
    "timestamp_ms",
    "frame_id",
    "key_down",
    "tip_conf",
    "shadow_gap_mm",
    "contact_shadow",
    "new_ink",
    "contact_ink",
]


def write_pd(path: Path, rows):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(PD_HEADER)
        w.writerows(rows)


def test_perfect_signals(tmp_path: Path):
    rows = []
    for i in range(10):
        down = i % 2
        rows.append([i * 33.0, i, down, "0.900", "1.000", down, 1 if down else 0, 1 if down else 0])
    p = tmp_path / "a.pendown.csv"
    write_pd(p, rows)
    r = score_pen_down(p)
    assert r["frames"] == 10
    assert r["contact_shadow"]["f1"] == 1.0
    assert r["contact_ink"]["f1"] == 1.0


def test_missed_and_false_positives(tmp_path: Path):
    rows = [
        # truth down, shadow says up -> miss
        [0.0, 0, 1, "0.9", "5.000", 0, "", ""],
        # truth up, shadow says down -> false positive
        [33.0, 1, 0, "0.9", "0.500", 1, "", ""],
        # correct down, ink unavailable (no fresh ink -> dry pen)
        [66.0, 2, 1, "0.9", "1.000", 1, 0, 0],
    ]
    p = tmp_path / "b.pendown.csv"
    write_pd(p, rows)
    r = score_pen_down(p)
    s = r["contact_shadow"]
    assert s["tp"] == 1 and s["fp"] == 1 and s["fn"] == 1
    assert s["precision"] == 0.5 and s["recall"] == 0.5
    # ink values only exist on row 2 (truth=1, pred=0 -> miss)
    i = r["contact_ink"]
    assert i["tp"] == 0 and i["fn"] == 1 and i["fp"] == 0


def test_unavailable_signals_excluded(tmp_path: Path):
    rows = [
        [0.0, 0, 1, "0.0", "", "", "", ""],  # no tip -> both signals empty
        [33.0, 1, 1, "0.9", "1.000", 1, 1, 1],
    ]
    p = tmp_path / "c.pendown.csv"
    write_pd(p, rows)
    r = score_pen_down(p)
    assert r["contact_shadow"]["tp"] == 1
    assert r["contact_shadow"]["fn"] == 0  # the no-tip frame is not scored


# --- detector unit tests -------------------------------------------------


def make_warped():
    import numpy as np

    return np.full((800, 600, 3), 240, np.uint8)


def make_tip(x=300.0, y=400.0):
    from paperpen.track.tip_cv import TipDetection

    return TipDetection(x=x, y=y, confidence=0.9, angle_deg=0.0, body_end=(0.0, 0.0))


def test_shadow_gap_far_and_near():
    import cv2

    from paperpen.config import PenDownConfig
    from paperpen.track.pendown import PenDownDetector

    det = PenDownDetector(PenDownConfig(), 600, 800, 210.0, 297.0)
    px_mm = 600 / 210.0

    # shadow 6 mm away from the tip (account for the blob's own radius)
    warped = make_warped()
    cx = int(round(300 + 6 * px_mm + 3))
    cv2.circle(warped, (cx, 400), 3, (180, 180, 180), -1)
    sig = det.update(warped, make_tip())
    assert sig.shadow_gap_mm is not None
    assert 5.0 <= sig.shadow_gap_mm <= 7.5, sig.shadow_gap_mm
    assert sig.contact_shadow is False

    # shadow merged with the tip
    warped2 = make_warped()
    cv2.circle(warped2, (302, 400), 4, (180, 180, 180), -1)
    sig2 = det.update(warped2, make_tip())
    assert sig2.contact_shadow is True
    assert sig2.shadow_gap_mm is not None and sig2.shadow_gap_mm <= 1.5


def test_no_shadow_means_no_contact():
    from paperpen.config import PenDownConfig
    from paperpen.track.pendown import PenDownDetector

    det = PenDownDetector(PenDownConfig(), 600, 800, 210.0, 297.0)
    sig = det.update(make_warped(), make_tip())
    assert sig.shadow_gap_mm is None
    assert sig.contact_shadow is False


def test_fresh_ink_signal():
    import cv2

    from paperpen.config import PenDownConfig
    from paperpen.track.pendown import PenDownDetector

    det = PenDownDetector(PenDownConfig(), 600, 800, 210.0, 297.0)
    det.update(make_warped(), make_tip())  # establishes prev frame (blank)
    # ink appears right under the tip
    warped = make_warped()
    cv2.line(warped, (295, 400), (320, 403), (40, 40, 40), 3)
    sig = det.update(warped, make_tip())
    assert sig.new_ink is True and sig.contact_ink is True
    # same image next frame -> no *new* ink
    sig2 = det.update(warped.copy(), make_tip())
    assert sig2.new_ink is False
    # fresh ink far from the tip does not count (locality check)
    warped3 = make_warped()
    cv2.line(warped3, (50, 50), (90, 55), (40, 40, 40), 3)
    sig3 = det.update(warped3, make_tip())
    assert sig3.new_ink is False


def test_pipeline_writes_pendown_log(tmp_path: Path):
    import time

    from paperpen.capture.source import Frame
    from paperpen.config import PenDownConfig
    from paperpen.output import SessionLogger
    from paperpen.pipeline import Pipeline

    from test_pipeline import make_cfg, scene

    cfg = make_cfg(tmp_path)
    cfg.pendown = PenDownConfig(enabled=True)
    logger = SessionLogger(cfg.log, session_name="pd")
    pipe = Pipeline(cfg, logger)
    pipe.step(Frame(scene(), time.monotonic(), 0), pen_down=False)
    pipe.step(Frame(scene(), time.monotonic(), 1), pen_down=True)
    logger.close()
    pd_csv = logger.csv_path.with_suffix(".pendown.csv")
    assert pd_csv.exists()
    lines = pd_csv.read_text().strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("timestamp_ms,frame_id,key_down")
