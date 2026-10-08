from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

PAPER_W_MM = 210.0
PAPER_H_MM = 297.0
DPI = 150
PX_W = int(round(PAPER_W_MM / 25.4 * DPI))  # 1240
PX_H = int(round(PAPER_H_MM / 25.4 * DPI))  # 1754


def mm_to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
    s = DPI / 25.4
    return (int(round(x_mm * s)), int(round(y_mm * s)))


def shape_truth() -> dict:
    """Ground-truth geometry in mm, y measured down from the top-left."""
    grid_segments = []
    for cy in (55.0, 148.5, 242.0):
        for cx in (55.0, 105.0, 155.0):
            grid_segments.append([cx - 10, cy, cx + 10, cy])
            grid_segments.append([cx, cy - 10, cx, cy + 10])
    return {
        "grid": {"segments": grid_segments},
        "circle": {"center": [105.0, 148.5], "radius": 80.0},
        "lines": {
            "segments": [
                [30.0, 60.0, 180.0, 60.0],
                [105.0, 30.0, 105.0, 264.0],
                [30.0, 30.0, 180.0, 264.0],
            ]
        },
    }


def _draw_shape(img: np.ndarray, name: str, truth: dict) -> None:
    ink = (20, 20, 20)
    thick = max(2, int(round(0.4 * DPI / 25.4)))
    if name in ("grid", "lines"):
        for x1, y1, x2, y2 in truth[name]["segments"]:
            cv2.line(img, mm_to_px(x1, y1), mm_to_px(x2, y2), ink, thick, cv2.LINE_AA)
    elif name == "circle":
        cx, cy = truth[name]["center"]
        r = truth[name]["radius"]
        cv2.circle(img, mm_to_px(cx, cy), int(round(r * DPI / 25.4)), ink, thick, cv2.LINE_AA)


def generate_shapes(out_dir: str | Path) -> dict:
    """Write printable A4 PNGs (grid.png, circle.png, lines.png) and
    shapes.json ground truth. Returns the truth dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth = shape_truth()
    for name in truth:
        img = np.full((PX_H, PX_W, 3), 255, np.uint8)
        _draw_shape(img, name, truth)
        cv2.imwrite(str(out / f"{name}.png"), img)
    (out / "shapes.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def load_truth(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _seg_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    vv = vx * vx + vy * vy
    if vv <= 1e-12:
        return math.hypot(wx, wy)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    return math.hypot(px - (x1 + t * vx), py - (y1 + t * vy))


def geometry_distance_mm(x: float, y: float, shape: dict) -> float:
    """Shortest distance in mm from (x, y) to the shape's ink."""
    if "segments" in shape:
        return min(
            _seg_dist(x, y, x1, y1, x2, y2) for x1, y1, x2, y2 in shape["segments"]
        )
    if "center" in shape:
        cx, cy = shape["center"]
        return abs(math.hypot(x - cx, y - cy) - shape["radius"])
    raise ValueError(f"unknown shape spec: {shape.keys()}")


def read_trace(csv_path: str | Path) -> tuple[list[float], list[float], list[float]]:
    """Returns (t_ms, x_mm, y_mm) for pen-down samples with valid coords."""
    t_ms: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    with Path(csv_path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("pen_down") != "1":
                continue
            if not row.get("paper_x") or not row.get("paper_y"):
                continue
            t_ms.append(float(row["timestamp_ms"]))
            xs.append(float(row["paper_x"]) * PAPER_W_MM)
            ys.append(float(row["paper_y"]) * PAPER_H_MM)
    return t_ms, xs, ys


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(sorted(values))
    return float(arr[min(int(round(p / 100 * (len(arr) - 1))), len(arr) - 1)])


def score_trace(
    csv_path: str | Path,
    truth: dict,
    shape_name: str,
    loss_gap_ms: float = 250.0,
) -> dict:
    """Accuracy of a traced shape: mm error vs ground truth + loss stats."""
    if shape_name not in truth:
        raise KeyError(f"unknown shape {shape_name!r}, have {list(truth)}")
    t_ms, xs, ys = read_trace(csv_path)
    if not xs:
        return {"n": 0}
    errs = [geometry_distance_mm(x, y, truth[shape_name]) for x, y in zip(xs, ys)]
    losses = 0
    lost_ms = 0.0
    for i in range(1, len(t_ms)):
        gap = t_ms[i] - t_ms[i - 1]
        if gap > loss_gap_ms:
            losses += 1
            lost_ms += gap
    total_ms = t_ms[-1] - t_ms[0] if len(t_ms) > 1 else 0.0
    return {
        "n": len(errs),
        "mean_mm": round(float(np.mean(errs)), 3),
        "median_mm": round(float(np.median(errs)), 3),
        "p90_mm": round(_pct(errs, 90), 3),
        "max_mm": round(float(np.max(errs)), 3),
        "losses": losses,
        "lost_ms": round(lost_ms, 1),
        "track_loss_pct": round(100.0 * lost_ms / total_ms, 2) if total_ms > 0 else 0.0,
    }
