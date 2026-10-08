from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from ..config import TipConfig


@dataclass
class TipDetection:
    x: float  # warped/paper pixel coords
    y: float
    confidence: float
    angle_deg: float  # pen axis orientation
    body_end: tuple[float, float]  # far (blunt/hand) end of the pen body
    contour: np.ndarray | None = field(default=None, repr=False)


def looks_like_skin(bgr_mean: np.ndarray) -> bool:
    b, g, r = (float(v) for v in bgr_mean)
    if r < 50 or g < 40:
        return False
    if not (r > g > b):
        return False
    drb = r - b
    return 15.0 <= drb <= 130.0 and (g - b) < 0.8 * drb + 10.0


def skin_fraction(color: np.ndarray) -> float:
    """Vectorized fraction of skin-colored pixels in an HxWx3 region."""
    b = color[..., 0].astype(np.float64)
    g = color[..., 1].astype(np.float64)
    r = color[..., 2].astype(np.float64)
    drb = r - b
    cond = (
        (r >= 50)
        & (g >= 40)
        & (r > g)
        & (g > b)
        & (drb >= 15.0)
        & (drb <= 130.0)
        & ((g - b) < 0.8 * drb + 10.0)
    )
    return float(cond.mean()) if cond.size else 0.0


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _score_contour(
    contour: np.ndarray, gray: np.ndarray, cfg: TipConfig
) -> tuple[float, TipDetection | None]:
    """Score one contour; returns (confidence, detection) or (0, None)."""
    area = float(cv2.contourArea(contour))
    total = gray.shape[0] * gray.shape[1]
    if area < cfg.min_area_px or area > cfg.max_area_frac * total:
        return 0.0, None

    pts = contour.reshape(-1, 2).astype(np.float64)
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    if cov.ndim != 2 or not np.all(np.isfinite(cov)):
        return 0.0, None
    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals[0] <= 1e-9:
        return 0.0, None
    axis = eigvecs[:, 1]  # largest eigenvalue last
    perp_v = eigvecs[:, 0]
    aspect = float(np.sqrt(eigvals[1] / eigvals[0]))
    if aspect < cfg.min_aspect:
        return 0.0, None

    proj = (pts - mean) @ axis
    perp = (pts - mean) @ perp_v
    width = float(np.percentile(perp, 95) - np.percentile(perp, 5))
    if width < cfg.min_body_width_px:
        return 0.0, None  # too thin: ink strokes, hairs, noise

    p_lo, p_hi = float(proj.min()), float(proj.max())
    span = max(p_hi - p_lo, 1e-6)
    # Sparse vs solid: a solid pen barrel fills most of its own bounding slab,
    # while clusters of handwriting only fill the strokes (~0.2-0.4).
    solidity = area / (span * max(width, 1e-6))
    if solidity < cfg.min_solidity:
        return 0.0, None
    # Narrow slice at each end: a wide slice washes out the taper that
    # distinguishes the pointed tip from the blunt back end.
    end_slice = max(0.06 * span, 4.0)

    def end_width(lo: bool) -> float:
        m = (proj <= p_lo + end_slice) if lo else (proj >= p_hi - end_slice)
        if int(m.sum()) < 3:
            return float("inf")
        w = np.percentile(perp[m], 90) - np.percentile(perp[m], 10)
        return float(w)

    w_lo, w_hi = end_width(True), end_width(False)
    lo_idx = int(np.argmin(proj))
    hi_idx = int(np.argmax(proj))
    if w_lo <= w_hi:
        tip_idx, body_idx = lo_idx, hi_idx
        width_sharp = (w_hi / w_lo) if np.isfinite(w_lo) and w_lo > 0.5 else 1.0
    else:
        tip_idx, body_idx = hi_idx, lo_idx
        width_sharp = (w_lo / w_hi) if np.isfinite(w_hi) and w_hi > 0.5 else 1.0
    if width_sharp < cfg.min_taper_ratio:
        return 0.0, None  # uniform bar: margin rules, ruled lines, barrel segments

    a_score = _clip01((aspect - cfg.min_aspect) / cfg.min_aspect)
    s_score = _clip01(width_sharp - 1.0)
    w_score = _clip01((width - cfg.min_body_width_px) / max(cfg.min_body_width_px, 1.0))
    region = np.zeros_like(gray)
    cv2.drawContours(region, [contour.astype(np.int32)], -1, 255, -1)
    region = region > 0
    if not region.any():
        return 0.0, None
    mean_gray = float(gray[region].mean())
    d_score = _clip01((cfg.bin_threshold - mean_gray) / max(cfg.bin_threshold, 1.0))
    o_score = _clip01((solidity - cfg.min_solidity) / max(1.0 - cfg.min_solidity, 1e-6))
    confidence = _clip01(
        0.25 * a_score + 0.30 * s_score + 0.15 * w_score + 0.15 * d_score + 0.15 * o_score
    )

    tip_pt = pts[tip_idx]
    body_pt = pts[body_idx]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    det = TipDetection(
        x=float(tip_pt[0]),
        y=float(tip_pt[1]),
        confidence=confidence,
        angle_deg=angle,
        body_end=(float(body_pt[0]), float(body_pt[1])),
        contour=contour.astype(np.int32),
    )
    return confidence, det


def detect_tip(
    warped: np.ndarray, cfg: TipConfig | None = None
) -> TipDetection | None:
    """Detect the pen tip in a top-down (warped) paper view.

    Strategy: the pen is the dark elongated blob whose narrow end is the
    tip; skin-colored blobs (fingers) are rejected; shadows usually stay
    above the threshold and ink lines are too thin to pass the width gate.
    """
    cfg = cfg or TipConfig()
    scale = float(cfg.detect_scale)
    if scale != 1.0:
        warped = cv2.resize(warped, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # thresholds are expressed in full-resolution warp pixels
        cfg = replace(
            cfg,
            min_area_px=max(1, int(cfg.min_area_px * scale * scale)),
            min_body_width_px=cfg.min_body_width_px * scale,
        )
    det = _detect_tip_scaled(warped, cfg)
    if det is None:
        return None
    if scale == 1.0:
        return det
    inv = 1.0 / scale
    contour = det.contour.astype(np.float32) * inv if det.contour is not None else None
    return TipDetection(
        x=det.x * inv,
        y=det.y * inv,
        confidence=det.confidence,
        angle_deg=det.angle_deg,
        body_end=(det.body_end[0] * inv, det.body_end[1] * inv),
        contour=contour.astype(np.int32) if contour is not None else None,
    )


def _detect_tip_scaled(warped: np.ndarray, cfg: TipConfig) -> TipDetection | None:
    cfg = cfg or TipConfig()
    if warped.ndim == 3:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        color = warped
    else:
        gray = warped
        color = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Non-paper mask: darker than the paper (ink, nib, dark barrels) OR
    # saturated (colored barrels like orange — bright, so the dark gate
    # alone would miss them). Shadows and white paper stay out (low sat).
    dark = blur < cfg.bin_threshold
    sat = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[..., 1]
    mask = (dark | (sat > cfg.sat_threshold)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # bridge specular glints on the metal nib: without this the nib's
    # connectivity (and thus the reported tip) flickers frame to frame
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best: TipDetection | None = None
    best_score = 0.0
    for c in contours:
        conf, det = _score_contour(c, gray, cfg)
        if det is None or conf <= 0.0:
            continue
        if cfg.skin_reject:
            filled = np.zeros(gray.shape[:2], np.uint8)
            cv2.drawContours(filled, [c], -1, 255, -1)
            region = filled > 0
            # fraction, not mean color: fingers gripping the barrel must not
            # veto the pen when the barrel still dominates the blob.
            if region.any() and skin_fraction(color[region]) > cfg.skin_frac_max:
                continue
        if conf > best_score:
            best_score = conf
            best = det
    if best is None or best_score < cfg.min_confidence:
        return None
    return best
