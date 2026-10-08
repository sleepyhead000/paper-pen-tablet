from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import PenDownConfig
from .tip_cv import TipDetection


@dataclass
class PenDownSignals:
    """Per-frame contact evidence (PRD §4.2). None = signal unavailable."""

    shadow_gap_mm: float | None = None
    contact_shadow: bool | None = None
    new_ink: bool | None = None
    contact_ink: bool | None = None


class PenDownDetector:
    """Two candidate automatic contact signals, logged for offline scoring
    against a key-held ground truth (M6):

    - shadow gap: distance from tip to the pen's shadow; the tip and its
      shadow merge on contact (depends on lighting).
    - fresh ink: frame-differenced dark pixels appearing under the tip;
      only fires if the pen actually writes.
    """

    def __init__(
        self,
        cfg: PenDownConfig,
        warp_w: int,
        warp_h: int,
        width_mm: float,
        height_mm: float,
    ):
        self.cfg = cfg
        self.px_per_mm_x = warp_w / width_mm
        self.px_per_mm_y = warp_h / height_mm
        self.prev_gray: np.ndarray | None = None

    def update(self, warped: np.ndarray, tip: TipDetection | None) -> PenDownSignals:
        sig = PenDownSignals()
        if warped.ndim == 3:
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            gray = warped
        prev = self.prev_gray
        self.prev_gray = gray
        if tip is None:
            return sig
        cfg = self.cfg
        tx, ty = int(round(tip.x)), int(round(tip.y))
        h, w = gray.shape[:2]
        if not (0 <= tx < w and 0 <= ty < h):
            return sig

        # --- shadow gap ---
        r_search = int(round(cfg.shadow_search_mm * self.px_per_mm_x))
        band = ((gray >= cfg.band_lo) & (gray <= cfg.band_hi)).astype(np.uint8) * 255
        if r_search > 2:
            search = np.zeros_like(band)
            cv2.circle(search, (tx, ty), r_search, 255, -1)
            band = cv2.bitwise_and(band, search)
        if np.any(band):
            # distance from every pixel to the nearest band pixel
            dt = cv2.distanceTransform(255 - band, cv2.DIST_L2, 3)
            gap_px = float(dt[ty, tx])
            gap_mm = gap_px / self.px_per_mm_x
            sig.shadow_gap_mm = round(gap_mm, 3)
            sig.contact_shadow = gap_mm <= cfg.shadow_contact_mm
        elif cfg.shadow_search_mm > 0:
            sig.shadow_gap_mm = None
            sig.contact_shadow = False  # no shadow in range -> clearly not touching

        # --- fresh ink ---
        if prev is not None and prev.shape == gray.shape:
            r_ink = max(2, int(round(cfg.ink_radius_mm * self.px_per_mm_x)))
            disk = np.zeros(gray.shape, np.uint8)
            cv2.circle(disk, (tx, ty), r_ink, 255, -1)
            fresh = (
                ((prev.astype(np.int16) - gray.astype(np.int16)) >= cfg.ink_delta)
                & (gray <= cfg.ink_max_gray)
                & (disk > 0)
            )
            sig.new_ink = bool(np.any(fresh))
            sig.contact_ink = sig.new_ink
        return sig
