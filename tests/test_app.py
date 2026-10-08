import time

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEventLoop

from paperpen.config import AppConfig, LogConfig, PaperConfig
from paperpen.ui.main_window import MainWindow


def _write_video(path, n=8, size=(320, 240)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    assert vw.isOpened()
    for i in range(n):
        img = np.full((size[1], size[0], 3), 40, np.uint8)
        quad = np.array([[30, 20], [290, 25], [295, 220], [25, 215]], np.int32)
        cv2.fillConvexPoly(img, quad, (236, 236, 236))
        # a dark elongated blob as fake pen
        cv2.fillConvexPoly(
            img, np.array([[90, 60], [200, 70], [200, 85], [90, 78]], np.int32), (30, 30, 36)
        )
        vw.write(img)
    vw.release()
    return path


@pytest.fixture(scope="session")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_app_runs_to_video_eof(qapp, tmp_path):
    video = _write_video(tmp_path / "in.mp4")
    cfg = AppConfig()
    cfg.paper = PaperConfig(warp_width=400, warp_height=560, detect_interval_s=10)
    cfg.source.type = "file"
    cfg.source.path = str(video)
    cfg.log = LogConfig(dir=str(tmp_path / "logs"))

    win = MainWindow(cfg, log=True)
    finished = []
    errors = []
    win.worker.finished_ok.connect(finished.append)
    win.worker.error.connect(errors.append)

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not finished:
        qapp.processEvents(QEventLoop.AllEvents, 50)

    assert not errors, f"worker errors: {errors}"
    assert finished, "worker never finished"
    assert finished[0]["frames"] >= 6
    assert finished[0]["log"]

    win.close()
    qapp.processEvents()
