from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import SourceConfig


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    t_recv: float  # monotonic seconds, when the frame was received
    frame_id: int
    t_wall: float = field(default_factory=time.time)  # wall clock, for logs


class FrameSource(ABC):
    """Produces timestamped frames.

    `next_frame` returns a frame newer than the last one it returned,
    waiting up to `timeout` seconds; None on timeout/EOF.
    """

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def next_frame(self, timeout: float = 1.0) -> Frame | None: ...

    @property
    def connected(self) -> bool:
        return True

    @property
    def live(self) -> bool:
        """True for sources that keep producing (reconnect on failure);
        False once a file source hits EOF."""
        return True

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VideoFileSource(FrameSource):
    """Sequential reader for recorded videos (deterministic replay)."""

    live = False

    def __init__(self, path: str):
        self.path = path
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video file: {self.path}")
        self._frame_id = 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def next_frame(self, timeout: float = 1.0) -> Frame | None:
        if self._cap is None:
            return None
        ok, img = self._cap.read()
        if not ok:
            return None
        frame = Frame(img, time.monotonic(), self._frame_id)
        self._frame_id += 1
        return frame

    def rewind(self) -> None:
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_id = 0


class LiveVideoSource(FrameSource):
    """Reader thread over a cv2.VideoCapture (webcam or HTTP stream).

    Keeps only the newest frame; stale frames are dropped so consumers
    always see near-live data even when processing is slower than capture.
    """

    def __init__(self, open_capture, reconnect_delay_s: float = 1.0):
        self._open_capture = open_capture
        self._reconnect_delay_s = reconnect_delay_s
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._seq = 0
        self._delivered = -1
        self._connected = False
        self._frames_seen = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    def open(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="frames")
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def next_frame(self, timeout: float = 1.0) -> Frame | None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                if self._latest is not None and self._seq > self._delivered:
                    self._delivered = self._seq
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def latest(self) -> Frame | None:
        with self._cond:
            return self._latest

    def _run(self) -> None:
        cap: cv2.VideoCapture | None = None
        while not self._stop.is_set():
            if cap is None:
                cap = self._open_capture()
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = None
                    self._connected = False
                    self._stop.wait(self._reconnect_delay_s)
                    continue
            ok, img = cap.read()
            if not ok or img is None:
                cap.release()
                cap = None
                self._connected = False
                continue
            self._connected = True
            self._frames_seen += 1
            with self._cond:
                self._latest = Frame(img, time.monotonic(), self._frames_seen)
                self._seq += 1
                self._cond.notify_all()
        if cap is not None:
            cap.release()


def _open_webcam(cfg: SourceConfig) -> cv2.VideoCapture:
    # On Windows the MSMF backend can take tens of seconds to open;
    # DirectShow opens in about a second.
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cfg.device, backend)
    if not cap.isOpened():
        return cap
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    return cap


def _open_http(cfg: SourceConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg.url, cv2.CAP_FFMPEG)
    return cap


def make_source(cfg: SourceConfig) -> FrameSource:
    if cfg.type == "file":
        if not cfg.path:
            raise ValueError("source.path required for file source")
        return VideoFileSource(cfg.path)
    if cfg.type == "webcam":
        return LiveVideoSource(lambda: _open_webcam(cfg), cfg.reconnect_delay_s)
    if cfg.type == "http":
        if not cfg.url:
            raise ValueError("source.url required for http source")
        return LiveVideoSource(lambda: _open_http(cfg), cfg.reconnect_delay_s)
    raise ValueError(f"unknown source type: {cfg.type!r}")


def parse_source_spec(spec: str, base: SourceConfig) -> SourceConfig:
    """Build a SourceConfig from a CLI spec: 'webcam', an int index,
    an http(s) URL, or a video file path."""
    import copy

    cfg = copy.deepcopy(base)
    s = spec.strip()
    if s == "webcam":
        cfg.type = "webcam"
    elif s.isdigit():
        cfg.type = "webcam"
        cfg.device = int(s)
    elif s.startswith(("http://", "https://")):
        cfg.type = "http"
        cfg.url = s
    elif os.path.exists(s):
        cfg.type = "file"
        cfg.path = s
    else:
        raise ValueError(f"cannot interpret source spec: {spec!r}")
    return cfg
