import csv
import math
from pathlib import Path

import cv2

from paperpen.evaluate import (
    generate_shapes,
    geometry_distance_mm,
    load_truth,
    score_trace,
    shape_truth,
)
from paperpen.evaluate.shapes import PX_H, PX_W, PAPER_H_MM, PAPER_W_MM
from paperpen.output import CSV_HEADER


def write_csv(path: Path, rows):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        w.writerows(rows)


def row(t_ms, x_norm, y_norm, down=1, conf=0.9):
    return [
        f"{t_ms:.1f}",
        0,
        "",
        "",
        f"{x_norm:.5f}" if x_norm is not None else "",
        f"{y_norm:.5f}" if y_norm is not None else "",
        down,
        f"{conf:.3f}",
    ]


def test_generate_shapes(tmp_path: Path):
    truth = generate_shapes(tmp_path)
    for name in ("grid", "circle", "lines"):
        p = tmp_path / f"{name}.png"
        assert p.exists()
        img = cv2.imread(str(p))
        assert img.shape == (PX_H, PX_W, 3)
        assert (img.mean(axis=2) < 128).sum() > 1000  # has ink
    assert (tmp_path / "shapes.json").exists()
    assert load_truth(tmp_path / "shapes.json").keys() == truth.keys()


def test_geometry_distances():
    truth = shape_truth()
    # point on the horizontal arm of a grid cross
    assert geometry_distance_mm(60.0, 55.0, truth["grid"]) == 0.0
    # 3 mm above the arm
    assert geometry_distance_mm(60.0, 52.0, truth["grid"]) == 3.0
    # on the circle circumference
    c = truth["circle"]
    px = c["center"][0] + c["radius"]
    assert geometry_distance_mm(px, c["center"][1], c) == 0.0
    # at the circle center -> error = radius
    assert geometry_distance_mm(c["center"][0], c["center"][1], c) == c["radius"]


def test_perfect_circle_trace(tmp_path: Path):
    truth = shape_truth()
    c = truth["circle"]
    rows = []
    t = 1000.0
    for i in range(120):
        a = 2 * math.pi * i / 120
        x = c["center"][0] + c["radius"] * math.cos(a)
        y = c["center"][1] + c["radius"] * math.sin(a)
        rows.append(row(t, x / PAPER_W_MM, y / PAPER_H_MM))
        t += 33.0
    csv_path = tmp_path / "perfect.csv"
    write_csv(csv_path, rows)
    r = score_trace(csv_path, truth, "circle")
    assert r["n"] == 120
    assert r["mean_mm"] < 0.01
    assert r["losses"] == 0


def test_noisy_circle_trace(tmp_path: Path):
    truth = shape_truth()
    c = truth["circle"]
    rows = []
    t = 1000.0
    for i in range(60):
        a = 2 * math.pi * i / 60
        r = c["radius"] + 1.5
        x = c["center"][0] + r * math.cos(a)
        y = c["center"][1] + r * math.sin(a)
        rows.append(row(t, x / PAPER_W_MM, y / PAPER_H_MM))
        t += 33.0
    csv_path = tmp_path / "noisy.csv"
    write_csv(csv_path, rows)
    r = score_trace(csv_path, truth, "circle")
    assert 1.0 < r["mean_mm"] < 2.0
    assert r["max_mm"] < 2.5


def test_tracking_losses_counted(tmp_path: Path):
    truth = shape_truth()
    c = truth["circle"]
    rows = []
    t = 1000.0
    for i in range(40):
        a = 2 * math.pi * i / 40
        x = c["center"][0] + c["radius"] * math.cos(a)
        y = c["center"][1] + c["radius"] * math.sin(a)
        rows.append(row(t, x / PAPER_W_MM, y / PAPER_H_MM))
        t += 33.0
        if i == 20:
            t += 1000.0  # 1s tracking loss
    csv_path = tmp_path / "lossy.csv"
    write_csv(csv_path, rows)
    r = score_trace(csv_path, truth, "circle")
    assert r["losses"] == 1
    assert r["lost_ms"] >= 1000
    assert r["track_loss_pct"] > 0


def test_read_trace_skips_hover(tmp_path: Path):
    rows = [
        row(1000.0, 0.5, 0.5, down=0),
        row(1033.0, None, None, down=1),  # pen down but no coords
        row(1066.0, 0.4, 0.3, down=1),
    ]
    csv_path = tmp_path / "hover.csv"
    write_csv(csv_path, rows)
    from paperpen.evaluate import read_trace

    t, xs, ys = read_trace(csv_path)
    assert len(xs) == 1
    assert xs[0] == 0.4 * PAPER_W_MM


def test_score_unknown_shape(tmp_path: Path):
    csv_path = tmp_path / "x.csv"
    write_csv(csv_path, [row(1000.0, 0.5, 0.5)])
    try:
        score_trace(csv_path, shape_truth(), "spiral")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
