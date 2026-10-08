from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import PaperConfig


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points to TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    peri = cv2.arcLength(contour, True)
    for eps_frac in (0.02, 0.015, 0.03, 0.01, 0.04, 0.05, 0.005):
        approx = cv2.approxPolyDP(contour, eps_frac * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx.reshape(4, 2))
    return None


def detect_paper(
    image: np.ndarray, min_area_frac: float = 0.05, min_brightness: float = 90.0
) -> np.ndarray | None:
    """Find the white paper quad in a frame. Returns 4x2 TL,TR,BR,BL or None."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu may select either polarity; paper is the large bright-ish region,
    # so test both polarities and keep the first plausible quad.
    h, w = gray.shape[:2]
    frame_area = float(h * w)
    for mask in (binary, 255 - binary):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours[:5]:
            area = cv2.contourArea(c)
            if area < min_area_frac * frame_area or area > 0.98 * frame_area:
                continue
            quad = _quad_from_contour(c)
            if quad is None:
                continue
            if _quad_brightness(gray, quad) < min_brightness:
                continue  # covered lens / dark scene, not paper
            return quad
    return None


def _quad_brightness(gray: np.ndarray, quad: np.ndarray) -> float:
    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    mask = cv2.erode(mask, np.ones((15, 15), np.uint8))
    if not np.any(mask):
        return 0.0
    return float(gray[mask > 0].mean())


def compute_homography(corners: np.ndarray, warp_w: int, warp_h: int) -> np.ndarray:
    src = order_corners(corners)
    dst = np.array(
        [[0, 0], [warp_w - 1, 0], [warp_w - 1, warp_h - 1], [0, warp_h - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def warp_paper(
    image: np.ndarray, homography: np.ndarray, warp_w: int, warp_h: int
) -> np.ndarray:
    return cv2.warpPerspective(image, homography, (warp_w, warp_h))


@dataclass
class PaperState:
    corners: np.ndarray  # 4x2 float32, frame coords, TL,TR,BR,BL
    homography: np.ndarray  # 3x3 float32
    method: str  # "detect" | "track"
    t: float


class PaperTracker:
    """Full corner detection on an interval, optical flow in between."""

    def __init__(self, cfg: PaperConfig | None = None):
        self.cfg = cfg or PaperConfig()
        self.state: PaperState | None = None
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._last_detect_t = -1e9
        self.detect_count = 0
        self.track_count = 0
        self.lost_count = 0

    def update(self, image: np.ndarray, t: float | None = None) -> PaperState | None:
        if t is None:
            t = time.monotonic()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        tracked: np.ndarray | None = None
        if self.state is not None and self.cfg.track_flow and self._prev_gray is not None:
            tracked = self._track(self._prev_gray, gray)

        need_detect = (
            self.state is None
            or tracked is None
            or (t - self._last_detect_t) >= self.cfg.detect_interval_s
        )
        if need_detect:
            quad = detect_paper(image, self.cfg.min_area_frac, self.cfg.min_brightness)
            if quad is not None:
                self.state = PaperState(
                    corners=quad,
                    homography=compute_homography(quad, self.cfg.warp_width, self.cfg.warp_height),
                    method="detect",
                    t=t,
                )
                self._last_detect_t = t
                self.detect_count += 1
            elif tracked is not None and self.state is not None:
                # keep the tracked position until redetection succeeds
                self.state = PaperState(
                    corners=tracked,
                    homography=compute_homography(tracked, self.cfg.warp_width, self.cfg.warp_height),
                    method="track",
                    t=t,
                )
                self.track_count += 1
            elif self.state is not None and tracked is None:
                # tracking failed and detection failed -> lost
                self.state = None
                self.lost_count += 1

        if self.state is not None and tracked is not None and not need_detect:
            self.state = PaperState(
                corners=tracked,
                homography=compute_homography(tracked, self.cfg.warp_width, self.cfg.warp_height),
                method="track",
                t=t,
            )
            self.track_count += 1

        self._prev_gray = gray
        self._prev_pts = (
            self.state.corners.reshape(-1, 1, 2).astype(np.float32)
            if self.state is not None
            else None
        )
        return self.state

    def _track(self, prev_gray: np.ndarray, cur_gray: np.ndarray) -> np.ndarray | None:
        if self._prev_pts is None:
            return None
        pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            cur_gray,
            self._prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if pts is None or status is None:
            return None
        if not bool(status.all()):
            return None
        out = pts.reshape(4, 2)
        moved = np.linalg.norm(out - self._prev_pts.reshape(4, 2), axis=1)
        if float(moved.max()) > 200.0:  # implausible jump -> distrust flow
            return None
        if abs(cv2.contourArea(order_corners(out))) < 0.2 * self._frame_area():
            return None
        return order_corners(out)

    def _frame_area(self) -> float:
        if self._prev_gray is None:
            return 1.0
        h, w = self._prev_gray.shape[:2]
        return float(h * w)
