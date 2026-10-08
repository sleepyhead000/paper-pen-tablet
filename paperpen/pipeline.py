from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibrate import PaperState, PaperTracker, warp_paper
from .capture.source import Frame
from .config import AppConfig
from .output.logger import SessionLogger
from .track.pendown import PenDownDetector
from .track.tip_cv import TipDetection, detect_tip


@dataclass
class StepResult:
    paper: PaperState | None
    warped: np.ndarray | None
    tip: TipDetection | None  # paper/warped pixel coords
    tip_frame: tuple[float, float] | None  # raw frame pixel coords
    paper_xy: tuple[float, float] | None  # normalized [0,1] paper coords
    confidence: float = 0.0


class Pipeline:
    """paper detection -> warp -> tip detection -> coordinate mapping -> log."""

    def __init__(self, cfg: AppConfig, logger: SessionLogger | None = None):
        self.cfg = cfg
        self.tracker = PaperTracker(cfg.paper)
        self.logger = logger
        self.pendown = (
            PenDownDetector(
                cfg.pendown,
                cfg.paper.warp_width,
                cfg.paper.warp_height,
                cfg.paper.width_mm,
                cfg.paper.height_mm,
            )
            if cfg.pendown.enabled
            else None
        )

    def step(self, frame: Frame, pen_down: bool = False) -> StepResult:
        paper = self.tracker.update(frame.image, frame.t_recv)
        warped = None
        tip = None
        tip_frame = None
        paper_xy = None
        confidence = 0.0
        if paper is not None:
            warped = warp_paper(
                frame.image,
                paper.homography,
                self.cfg.paper.warp_width,
                self.cfg.paper.warp_height,
            )
            tip = detect_tip(warped, self.cfg.tip)
            if tip is not None:
                confidence = tip.confidence
                inv = np.linalg.inv(paper.homography)
                pt = cv2.perspectiveTransform(
                    np.array([[[tip.x, tip.y]]], np.float32), inv
                )[0, 0]
                tip_frame = (float(pt[0]), float(pt[1]))
                paper_xy = (
                    tip.x / self.cfg.paper.warp_width,
                    tip.y / self.cfg.paper.warp_height,
                )
        signals = None
        if self.pendown is not None and warped is not None:
            signals = self.pendown.update(warped, tip)
        if self.logger is not None:
            self.logger.log(
                timestamp_ms=frame.t_wall * 1000.0,
                frame_id=frame.frame_id,
                tip_px=tip_frame,
                paper_xy=paper_xy,
                pen_down=pen_down,
                confidence=confidence,
                homography=paper.homography if paper is not None else None,
                frame=frame.image if self.logger.cfg.record_video else None,
            )
            if signals is not None:
                self.logger.log_pen_down(
                    frame.t_wall * 1000.0,
                    frame.frame_id,
                    signals,
                    pen_down,
                    confidence,
                )
        return StepResult(
            paper=paper,
            warped=warped,
            tip=tip,
            tip_frame=tip_frame,
            paper_xy=paper_xy,
            confidence=confidence,
        )


def draw_overlay(image: np.ndarray, result: StepResult) -> np.ndarray:
    """Debug view: paper quad, warped thumbnail, tip marker."""
    out = image
    if result.paper is not None:
        pts = result.paper.corners.astype(np.int32)
        for i in range(4):
            a, b = tuple(pts[i]), tuple(pts[(i + 1) % 4])
            cv2.line(out, a, b, (0, 200, 255), 2)
        cv2.putText(
            out,
            result.paper.method,
            tuple(pts[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
    if result.tip_frame is not None:
        p = tuple(int(v) for v in result.tip_frame)
        cv2.circle(out, p, 9, (0, 0, 255), 2)
        cv2.drawMarker(out, p, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(
            out,
            f"{result.confidence:.2f}",
            (p[0] + 12, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        if result.warped is not None:
            thumb = cv2.resize(result.warped, (192, 272))
            oh, ow = out.shape[:2]
            y0 = oh - 272 - 8
            if y0 > 0:
                x0 = ow - 192 - 8
                thumb_color = thumb if thumb.ndim == 3 else cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
                out[y0 : y0 + 272, x0 : x0 + 192] = thumb_color
    return out
