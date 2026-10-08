import cv2
import numpy as np

from paperpen.calibrate import (
    PaperTracker,
    compute_homography,
    detect_paper,
    order_corners,
    warp_paper,
)
from paperpen.config import PaperConfig

GT = np.array([[100, 80], [540, 60], [580, 400], [80, 420]], np.float32)


def scene(quad=GT, size=(480, 640), ink=True, seed=0):
    """Dark desk with a white paper quad (+ optional ink marks and noise)."""
    h, w = size
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 40, np.uint8)
    img = (img + rng.integers(0, 12, img.shape)).astype(np.uint8)
    cv2.fillConvexPoly(img, quad.astype(np.int32), (242, 242, 242))
    if ink:
        pts = quad.astype(np.int32)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.rectangle(img, (cx - 60, cy - 20), (cx + 40, cy - 10), (20, 20, 20), -1)
        cv2.line(img, (cx - 50, cy + 30), (cx + 60, cy + 45), (15, 15, 15), 3)
    return img


def translated(quad, dx, dy):
    return (quad + np.array([dx, dy], np.float32)).astype(np.float32)


def test_order_corners_shuffle():
    shuffled = GT[[2, 0, 3, 1]]
    assert np.allclose(order_corners(shuffled), GT, atol=1e-5)


def test_detect_paper():
    quad = detect_paper(scene())
    assert quad is not None
    # tolerance includes the deliberate ~3 px inward bias of the bright mask
    assert np.allclose(quad, GT, atol=5.0)


def test_detect_paper_no_paper():
    assert detect_paper(np.full((480, 640, 3), 40, np.uint8)) is None


def test_detect_paper_dark_scene_rejected():
    # Otsu finds the "brightest" region even in an all-dark frame; the
    # brightness gate must reject it as paper.
    dark = np.full((480, 640, 3), 45, np.uint8)
    cv2.fillConvexPoly(dark, GT.astype(np.int32), (75, 75, 75))
    assert detect_paper(dark) is None


def test_detect_paper_min_brightness_override():
    dark = np.full((480, 640, 3), 45, np.uint8)
    cv2.fillConvexPoly(dark, GT.astype(np.int32), (75, 75, 75))
    assert detect_paper(dark, min_brightness=50.0) is not None


def test_homography_maps_corners():
    H = compute_homography(GT, 1654, 2339)
    mapped = cv2.perspectiveTransform(GT.reshape(-1, 1, 2), H).reshape(4, 2)
    expect = np.array([[0, 0], [1653, 0], [1653, 2338], [0, 2338]], np.float32)
    assert np.allclose(mapped, expect, atol=1.0)


def test_warp_paper_shape_and_content():
    H = compute_homography(GT, 400, 566)
    warped = warp_paper(scene(), H, 400, 566)
    assert warped.shape == (566, 400, 3)
    assert float(warped[40, 200].mean()) > 200  # inside paper
    dark_frac = float((warped.mean(axis=2) < 60).mean())
    assert dark_frac > 0.005  # ink marks survived the warp


def test_tracker_detect_then_track():
    tracker = PaperTracker(PaperConfig(detect_interval_s=10.0))
    st0 = tracker.update(scene(), t=0.0)
    assert st0 is not None and st0.method == "detect"
    gt1 = translated(GT, 12, 7)
    st1 = tracker.update(scene(quad=gt1), t=0.05)
    assert st1 is not None
    assert st1.method == "track"
    # flow carries detect's deliberate ~3 px inward bias along with it
    assert np.allclose(st1.corners, gt1, atol=6.0)
    assert tracker.detect_count == 1


def test_tracker_lost_and_reacquire():
    tracker = PaperTracker(PaperConfig(detect_interval_s=10.0))
    assert tracker.update(scene(), t=0.0) is not None
    empty = np.full((480, 640, 3), 40, np.uint8)
    lost = tracker.update(empty, t=0.05)
    assert lost is None
    assert tracker.lost_count >= 1
    back = tracker.update(scene(), t=0.10)
    assert back is not None and back.method == "detect"


def test_tracker_periodic_redetect():
    tracker = PaperTracker(PaperConfig(detect_interval_s=0.1))
    tracker.update(scene(), t=0.0)
    quad = translated(GT, 5, 5)
    st = tracker.update(scene(quad=quad), t=0.2)
    assert st is not None and st.method == "detect"
    assert np.allclose(st.corners, quad, atol=5.0)
    assert tracker.detect_count == 2


def test_tracker_survives_missed_redetect(monkeypatch):
    import paperpen.calibrate.paper as paper_mod

    tracker = PaperTracker(PaperConfig(detect_interval_s=0.0))
    tracker.update(scene(), t=0.0)
    quad = translated(GT, 6, 4)
    # periodic detection fails (e.g. edges briefly occluded) but flow still tracks
    monkeypatch.setattr(paper_mod, "detect_paper", lambda *a, **k: None)
    st = tracker.update(scene(quad=quad), t=1.0)
    assert st is not None and st.method == "track"
    assert np.allclose(st.corners, quad, atol=4.0)


def test_far_dim_redetect_rejected_while_tracking(monkeypatch):
    """A bright-desk/pad quad far from the tracked paper must not hijack it."""
    import paperpen.calibrate.paper as paper_mod

    tracker = PaperTracker(PaperConfig(detect_interval_s=0.0))
    assert tracker.update(scene(), t=0.0) is not None
    far = translated(GT, 400, 0)  # lands on the dark desk in scene()
    monkeypatch.setattr(paper_mod, "detect_paper", lambda *a, **k: far)
    st = tracker.update(scene(), t=0.1)
    assert st is not None and st.method == "track"
    assert np.allclose(st.corners, GT, atol=4.0)
    assert tracker.detect_count == 1  # the far quad was not accepted


def test_accept_detect_rules():
    from paperpen.calibrate.paper import PaperState, compute_homography

    tracker = PaperTracker(PaperConfig())
    tracker.state = PaperState(
        corners=GT.astype(np.float32),
        homography=compute_homography(GT, 400, 566),
        method="detect",
        t=0.0,
    )
    near = (GT + np.array([100, 0], np.float32)).astype(np.float32)
    far = (GT + np.array([400, 0], np.float32)).astype(np.float32)
    gray_dim = np.full((480, 640), 60, np.uint8)
    gray_paper = np.full((480, 640), 235, np.uint8)
    assert tracker._accept_detect(near, gray_dim)  # near the track: trusted
    assert not tracker._accept_detect(far, gray_dim)  # far and not paper-white
    assert tracker._accept_detect(far, gray_paper)  # far but unmistakably paper
    tracker.state = None
    assert tracker._accept_detect(far, gray_dim)  # no state: lock anything
