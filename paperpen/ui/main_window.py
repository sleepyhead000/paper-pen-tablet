from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig
from ..output.canvas import CanvasWidget
from .worker import PipelineWorker


class PenKeyState:
    """Global pen-down fallback state (PRD 4.2): hold Space to draw."""

    def __init__(self) -> None:
        self._down = False
        self._listener = None
        try:
            from pynput import keyboard

            def on_press(key):
                if key == keyboard.Key.space:
                    self._down = True

            def on_release(key):
                if key == keyboard.Key.space:
                    self._down = False

            self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.daemon = True
            self._listener.start()
        except Exception:  # noqa: BLE001  # pynput unavailable -> Qt key events
            self._listener = None

    @property
    def down(self) -> bool:
        return self._down

    def set_qt(self, pressed: bool) -> None:
        self._down = pressed

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig, log: bool = True, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.pen_key = PenKeyState()
        self.setWindowTitle("Paper-Pen")
        self.resize(1200, 760)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # left: live camera overlay
        left = QVBoxLayout()
        self.video = QLabel("starting camera...")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(480, 360)
        self.video.setStyleSheet("background:#202226; color:#9aa0a8;")
        left.addWidget(self.video, stretch=1)
        self.info = QLabel("")
        self.info.setStyleSheet("color:#c7cbd1; font-family:monospace;")
        left.addWidget(self.info)
        root.addLayout(left, stretch=3)

        # right: canvas + controls
        right = QVBoxLayout()
        self.canvas = CanvasWidget(
            paper_aspect=self.cfg.paper.warp_width / self.cfg.paper.warp_height
        )
        right.addWidget(self.canvas, stretch=1)

        buttons = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.canvas.clear)
        save_btn = QPushButton("Save PNG")
        save_btn.clicked.connect(self._save_canvas)
        self.draw_check = QCheckBox("Draw (hold Space)")
        self.draw_check.setChecked(True)
        buttons.addWidget(clear_btn)
        buttons.addWidget(save_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.draw_check)
        right.addLayout(buttons)
        root.addLayout(right, stretch=2)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("starting...")

        self.worker = PipelineWorker(cfg, self._pen_down, log=log)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.error.connect(self._on_error)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.start()

    def _pen_down(self) -> bool:
        return self.pen_key.down and self.draw_check.isChecked()

    def _on_frame(self, image: QImage, sample: dict) -> None:
        pix = QPixmap.fromImage(image)
        self.video.setPixmap(
            pix.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        if sample["valid"]:
            self.canvas.add_point(
                sample["x"], sample["y"], sample["pen_down"], sample["coasting"]
            )
        else:
            self.canvas.track_lost()
        state = "ok" if sample["paper"] else "no paper"
        tip_s = f"{sample['confidence']:.2f}" if sample["tip"] else "-"
        coast = " coasting" if sample["coasting"] else ""
        lost = "" if sample["valid"] else " LOST"
        self.info.setText(
            f"fps {sample['fps']:.1f} | paper {state} | tip {tip_s} | "
            f"pen {'DOWN' if sample['pen_down'] else 'up'}{coast}{lost} | "
            f"strokes {len(self.canvas.builder.strokes)} pts {self.canvas.builder.point_count}"
        )

    def _on_error(self, message: str) -> None:
        self.statusBar().showMessage(f"error: {message}")
        self.video.setText(f"error: {message}")

    def _on_finished(self, stats: dict) -> None:
        msg = f"stopped after {stats['frames']} frames"
        if stats.get("log"):
            msg += f" (log: {stats['log']})"
        self.statusBar().showMessage(msg)

    def _save_canvas(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save canvas", "canvas.png", "PNG (*.png)"
        )
        if path:
            self.canvas.grab().save(path)
            self.statusBar().showMessage(f"saved {path}")

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Space:
            self.pen_key.set_qt(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Space:
            self.pen_key.set_qt(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.worker.stop()
        self.worker.wait(5000)
        self.pen_key.stop()
        super().closeEvent(event)


def run_app(cfg: AppConfig, log: bool = True) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(cfg, log=log)
    win.show()
    return app.exec()
