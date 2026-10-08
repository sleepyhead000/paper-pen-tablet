import time

import cv2
import numpy as np

from paperpen.capture.benchmark import (
    _QR_PREFIX,
    decode_latency,
    make_qr_image,
    measure_fps,
    summarize,
)
from paperpen.capture.source import Frame, FrameSource


class FakeSource(FrameSource):
    def __init__(self, images):
        self._images = list(images)
        self._i = 0

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def next_frame(self, timeout: float = 1.0):
        if self._i >= len(self._images):
            return None
        img = self._images[self._i]
        f = Frame(img, time.monotonic(), self._i)
        self._i += 1
        return f


def test_decode_latency_roundtrip():
    t_gen = time.monotonic() - 0.123  # 123 ms ago
    img = make_qr_image(f"{_QR_PREFIX}{t_gen:.6f}", 500)
    frame = Frame(img, time.monotonic(), 0)
    lat = decode_latency(frame, cv2.QRCodeDetector())
    assert lat is not None
    assert 100.0 < lat < 200.0


def test_decode_latency_rejects_garbage():
    frame = Frame(np.zeros((200, 200, 3), np.uint8), time.monotonic(), 0)
    assert decode_latency(frame, cv2.QRCodeDetector()) is None


def test_decode_latency_rejects_bad_payload():
    img = make_qr_image("not-a-number", 400)
    frame = Frame(img, time.monotonic(), 0)
    assert decode_latency(frame, cv2.QRCodeDetector()) is None


def test_measure_fps_on_frames():
    imgs = [np.full((64, 64, 3), 10, np.uint8) for _ in range(5)]
    fps = measure_fps(FakeSource(imgs), duration_s=0.5)
    assert fps["frames"] == 5


def test_summarize():
    s = summarize([10.0, 20.0, 30.0, 40.0])
    assert s["n"] == 4
    assert s["min_ms"] == 10.0
    assert s["max_ms"] == 40.0
    assert summarize([]) == {"n": 0}
