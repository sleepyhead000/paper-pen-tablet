import cv2
import numpy as np
import pytest

from paperpen.config import TipConfig
from paperpen.track.tip_cv import detect_tip, looks_like_skin

W, H = 800, 1000


def paper() -> np.ndarray:
    return np.full((H, W, 3), 235, np.uint8)


def draw_pen(
    img,
    tip,
    angle_deg,
    length=420,
    width=44,
    taper=44,
    color=(28, 28, 32),
):
    """Elongated blob with a pointed tip. angle = direction tip->body."""
    a = np.radians(angle_deg)
    axis = np.array([np.cos(a), np.sin(a)])
    perp = np.array([-axis[1], axis[0]])
    tip = np.asarray(tip, np.float64)
    base = tip + axis * taper
    back = tip + axis * length
    pts = [
        tip,
        base + perp * (width / 2),
        back + perp * (width / 2),
        back - perp * (width / 2),
        base - perp * (width / 2),
    ]
    poly = np.array(pts, np.int32)
    cv2.fillConvexPoly(img, poly, color)
    return tip


def draw_ink(img, p0, p1, thickness=5, color=(25, 25, 25)):
    cv2.line(img, tuple(map(int, p0)), tuple(map(int, p1)), color, thickness)


def test_detects_pen_tip():
    img = paper()
    expected = draw_pen(img, (250, 300), angle_deg=35)
    det = detect_tip(img, TipConfig())
    assert det is not None
    err = np.hypot(det.x - expected[0], det.y - expected[1])
    assert err < 8.0, f"tip off by {err:.1f}px"
    assert det.confidence > 0.5


def test_detects_tip_any_angle():
    for angle in (0, 90, 180, 270, 200):
        img = paper()
        expected = draw_pen(img, (400, 500), angle_deg=angle)
        det = detect_tip(img, TipConfig())
        assert det is not None, f"missed at angle {angle}"
        err = np.hypot(det.x - expected[0], det.y - expected[1])
        assert err < 8.0, f"angle {angle}: {err:.1f}px"


def test_ink_lines_rejected():
    img = paper()
    draw_ink(img, (100, 200), (700, 260))
    draw_ink(img, (120, 700), (160, 300), thickness=7)
    assert detect_tip(img, TipConfig()) is None


def test_shadow_rejected():
    img = paper()
    # bright shadow: above bin_threshold, so outside the dark mask
    draw_pen(img, (300, 400), 45, color=(150, 150, 155))
    assert detect_tip(img, TipConfig()) is None


def test_pen_beats_shadow():
    img = paper()
    draw_pen(img, (300, 410), 45, color=(145, 145, 150))  # shadow, offset from pen
    expected = draw_pen(img, (300, 400), 45, color=(25, 25, 30))  # the pen
    det = detect_tip(img, TipConfig())
    assert det is not None
    err = np.hypot(det.x - expected[0], det.y - expected[1])
    assert err < 8.0


def test_skin_blob_rejected():
    img = paper()
    # dark-skinned elongated finger in the mask range
    draw_pen(img, (400, 400), 60, length=380, width=90, taper=120, color=(70, 95, 135))
    assert detect_tip(img, TipConfig(skin_reject=True)) is None
    # without the skin gate it would be a candidate
    assert detect_tip(img, TipConfig(skin_reject=False)) is not None


def test_scene_with_everything():
    img = paper()
    draw_ink(img, (80, 120), (720, 190))
    draw_ink(img, (100, 850), (650, 900), thickness=6)
    draw_pen(img, (350, 460), 20, color=(150, 150, 155))  # shadow
    expected = draw_pen(img, (350, 450), 20, color=(30, 30, 35))  # pen
    det = detect_tip(img, TipConfig())
    assert det is not None
    err = np.hypot(det.x - expected[0], det.y - expected[1])
    assert err < 10.0


def test_confidence_gate():
    img = paper()
    draw_pen(img, (300, 300), 45)
    assert detect_tip(img, TipConfig(min_confidence=1.1)) is None


@pytest.mark.parametrize(
    "length,width,taper",
    [(420, 44, 44), (190, 22, 24), (120, 30, 40), (300, 60, 55), (150, 20, 30)],
)
@pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, 225, 270, 315])
def test_tip_direction_across_shapes(length, width, taper, angle):
    img = paper()
    expected = draw_pen(img, (400, 500), angle, length=length, width=width, taper=taper)
    det = detect_tip(img, TipConfig())
    assert det is not None, f"missed len={length} ang={angle}"
    err = np.hypot(det.x - expected[0], det.y - expected[1])
    # half-res detection: tolerance covers apex quantization, not direction
    assert err < 15.0, f"len={length} w={width} ang={angle}: {err:.1f}px"


def test_grayscale_input():
    img = paper()
    expected = draw_pen(img, (250, 300), 35)
    det = detect_tip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), TipConfig())
    assert det is not None
    assert np.hypot(det.x - expected[0], det.y - expected[1]) < 8.0


def test_looks_like_skin():
    assert looks_like_skin(np.array([90, 120, 160]))  # typical caucasian-ish
    assert not looks_like_skin(np.array([200, 200, 200]))  # gray
    assert not looks_like_skin(np.array([40, 40, 200]))  # blue-ish red? BGR blue-ish
    assert not looks_like_skin(np.array([30, 30, 30]))  # pen color
