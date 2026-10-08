from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import LogConfig

CSV_HEADER = [
    "timestamp_ms",
    "frame_id",
    "tip_x_px",
    "tip_y_px",
    "paper_x",
    "paper_y",
    "pen_down",
    "confidence",
]


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


class SessionLogger:
    """CSV log per PRD §6, optional raw video and homography sidecar."""

    def __init__(self, cfg: LogConfig, session_name: str | None = None):
        self.cfg = cfg
        name = session_name or time.strftime("session_%Y%m%d_%H%M%S")
        self.dir = Path(cfg.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / f"{name}.csv"
        self._fh = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(CSV_HEADER)
        self._rows = 0
        self._video: cv2.VideoWriter | None = None
        self._video_path: Path | None = None

    def log(
        self,
        timestamp_ms: float,
        frame_id: int,
        tip_px: tuple[float, float] | None,
        paper_xy: tuple[float, float] | None,
        pen_down: bool,
        confidence: float,
        homography: np.ndarray | None = None,
        frame: np.ndarray | None = None,
    ) -> None:
        self._writer.writerow(
            [
                f"{timestamp_ms:.1f}",
                frame_id,
                _fmt(tip_px[0] if tip_px else None),
                _fmt(tip_px[1] if tip_px else None),
                _fmt(paper_xy[0] if paper_xy else None),
                _fmt(paper_xy[1] if paper_xy else None),
                int(pen_down),
                f"{confidence:.3f}",
            ]
        )
        self._rows += 1
        if homography is not None:
            self._write_homography(timestamp_ms, frame_id, homography)
        if frame is not None:
            self._write_frame(frame)
        if self._rows % self.cfg.flush_every == 0:
            self._fh.flush()

    def _write_homography(self, timestamp_ms: float, frame_id: int, h: np.ndarray) -> None:
        if not hasattr(self, "_h_fh"):
            self._h_fh = (self.csv_path.with_suffix(".homography.csv")).open(
                "w", newline="", encoding="utf-8"
            )
            self._h_writer = csv.writer(self._h_fh)
            self._h_writer.writerow(["timestamp_ms", "frame_id"] + [f"h{i}" for i in range(9)])
        self._h_writer.writerow(
            [f"{timestamp_ms:.1f}", frame_id] + [f"{v:.6f}" for v in h.reshape(-1)]
        )

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._video is None:
            h, w = frame.shape[:2]
            self._video_path = self.dir / f"{self.csv_path.stem}.mp4"
            self._video = cv2.VideoWriter(
                str(self._video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
            )
        self._video.write(frame)

    def log_pen_down(
        self,
        timestamp_ms: float,
        frame_id: int,
        signals,
        key_down: bool,
        tip_conf: float,
    ) -> None:
        """Sidecar log for M6 pen-down experiments (signals vs key ground truth)."""
        if not hasattr(self, "_pd_fh"):
            self._pd_fh = (self.csv_path.with_suffix(".pendown.csv")).open(
                "w", newline="", encoding="utf-8"
            )
            self._pd_writer = csv.writer(self._pd_fh)
            self._pd_writer.writerow(
                [
                    "timestamp_ms",
                    "frame_id",
                    "key_down",
                    "tip_conf",
                    "shadow_gap_mm",
                    "contact_shadow",
                    "new_ink",
                    "contact_ink",
                ]
            )
        self._pd_writer.writerow(
            [
                f"{timestamp_ms:.1f}",
                frame_id,
                int(key_down),
                f"{tip_conf:.3f}",
                "" if signals.shadow_gap_mm is None else f"{signals.shadow_gap_mm:.3f}",
                "" if signals.contact_shadow is None else int(signals.contact_shadow),
                "" if signals.new_ink is None else int(signals.new_ink),
                "" if signals.contact_ink is None else int(signals.contact_ink),
            ]
        )
        self._pd_rows = getattr(self, "_pd_rows", 0) + 1
        if self._pd_rows % self.cfg.flush_every == 0:
            self._pd_fh.flush()

    @property
    def rows(self) -> int:
        return self._rows

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()
        if getattr(self, "_h_fh", None) is not None:
            self._h_fh.flush()
            self._h_fh.close()
        if getattr(self, "_pd_fh", None) is not None:
            self._pd_fh.flush()
            self._pd_fh.close()
        if self._video is not None:
            self._video.release()
            self._video = None
