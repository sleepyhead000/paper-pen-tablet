from __future__ import annotations

import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from ..capture import make_source
from ..config import AppConfig
from ..output.logger import SessionLogger
from ..pipeline import Pipeline, draw_overlay
from ..track.filter import Smoother


class PipelineWorker(QThread):
    """Capture -> pipeline -> smoothing loop, off the Qt main thread.

    Emits one combined payload per processed frame:
      frame_ready(QImage overlay, dict sample)
    """

    frame_ready = Signal(object, object)
    error = Signal(str)
    finished_ok = Signal(dict)

    def __init__(self, cfg: AppConfig, pen_down_fn, log: bool = True, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.pen_down_fn = pen_down_fn
        self.log = log
        self._quit = False
        self.mouse = None

    def stop(self) -> None:
        self._quit = True

    def run(self) -> None:
        try:
            source = make_source(self.cfg.source)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
            return
        logger = SessionLogger(self.cfg.log) if self.log else None
        pipeline = Pipeline(self.cfg, logger)
        smoother = Smoother(self.cfg.filter)
        if self.cfg.mouse.enabled:
            try:
                from ..output.mouse import MouseController

                self.mouse = MouseController(self.cfg.mouse)
            except Exception as exc:  # noqa: BLE001
                self.error.emit(f"mouse init failed: {exc}")
                self.mouse = None
        frames = 0
        tips = 0
        papers = 0
        last_report = time.monotonic()
        fps = 0.0
        processed_since = 0
        try:
            source.open()
            while not self._quit:
                frame = source.next_frame(timeout=0.5)
                if frame is None:
                    if not source.live:
                        break  # recorded video hit EOF
                    continue
                pen_down = bool(self.pen_down_fn())
                result = pipeline.step(frame, pen_down=pen_down)
                sm = smoother.update(result.paper_xy, frame.t_recv)
                if self.mouse is not None:
                    self.mouse.update(sm.x, sm.y, pen_down, sm.valid)
                frames += 1
                processed_since += 1
                papers += result.paper is not None
                tips += result.tip is not None
                now = time.monotonic()
                if now - last_report >= 1.0:
                    fps = processed_since / (now - last_report)
                    processed_since = 0
                    last_report = now
                overlay = draw_overlay(frame.image.copy(), result)
                sample = {
                    "x": sm.x,
                    "y": sm.y,
                    "valid": sm.valid,
                    "coasting": sm.coasting,
                    "pen_down": pen_down,
                    "confidence": result.confidence,
                    "paper": result.paper is not None,
                    "tip": result.tip is not None,
                    "fps": fps,
                    "frames": frames,
                    "paper_rate": papers / max(frames, 1),
                    "tip_rate": tips / max(frames, 1),
                }
                self.frame_ready.emit(_to_qimage(overlay), sample)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            source.close()
            if self.mouse is not None:
                self.mouse.stop()
            if logger is not None:
                logger.close()
            self.finished_ok.emit(
                {"frames": frames, "tips": tips, "log": str(logger.csv_path) if logger else None}
            )


def _to_qimage(img: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
