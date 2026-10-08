from __future__ import annotations

import csv
from pathlib import Path

LABEL_HEADER = ["frame_id", "timestamp_ms", "x", "y", "warp_width", "warp_height"]


class LabelStore:
    """Tip keypoint labels in warped paper coords (M5 dataset collection).

    Rewrites the CSV on every change so a crash never loses labels."""

    def __init__(self, path: str | Path, warp_w: int, warp_h: int):
        self.path = Path(path)
        self.warp_w = warp_w
        self.warp_h = warp_h
        self.rows: list[dict] = []
        if self.path.exists():
            with self.path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    self.rows.append(
                        {
                            "frame_id": int(row["frame_id"]),
                            "timestamp_ms": float(row["timestamp_ms"]),
                            "x": float(row["x"]),
                            "y": float(row["y"]),
                            "warp_width": int(row["warp_width"]),
                            "warp_height": int(row["warp_height"]),
                        }
                    )

    def add(self, frame_id: int, timestamp_ms: float, x: float, y: float) -> None:
        self.rows.append(
            {
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "x": x,
                "y": y,
                "warp_width": self.warp_w,
                "warp_height": self.warp_h,
            }
        )
        self._flush()

    def undo(self) -> dict | None:
        row = self.rows.pop() if self.rows else None
        self._flush()
        return row

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LABEL_HEADER)
            w.writeheader()
            w.writerows(self.rows)

    def __len__(self) -> int:
        return len(self.rows)
