from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


@dataclass
class StrokeBuilder:
    """Headless stroke bookkeeping: turns a stream of (x, y, pen_down)
    samples into polylines in normalized paper coords [0,1]."""

    strokes: list[list[tuple[float, float]]] = field(default_factory=list)
    min_move: float = 0.0005
    _current: list[tuple[float, float]] | None = field(default=None, repr=False)
    _was_down: bool = field(default=False, repr=False)

    def add_point(self, x: float, y: float, pen_down: bool, coasting: bool = False) -> None:
        if not pen_down:
            self._current = None
            self._was_down = False
            return
        if coasting:
            # predicted point: continue an open stroke through the occlusion,
            # but never start ink from a prediction
            if self._current is None:
                return
            self._current.append((x, y))
            return
        if not self._was_down or self._current is None:
            self._current = []
            self.strokes.append(self._current)
        if self._current:
            lx, ly = self._current[-1]
            if abs(x - lx) < self.min_move and abs(y - ly) < self.min_move:
                return
        self._current.append((x, y))
        self._was_down = True

    def track_lost(self) -> None:
        """Tracking went invalid: close the stroke so a long wrong bridge
        cannot appear when tracking recovers."""
        self._current = None
        self._was_down = False

    def clear(self) -> None:
        self.strokes.clear()
        self._current = None
        self._was_down = False

    @property
    def point_count(self) -> int:
        return sum(len(s) for s in self.strokes)


class CanvasWidget(QWidget):
    """Drawing surface. Points arrive in normalized paper coords; the paper
    aspect ratio is preserved and letterboxed inside the widget."""

    def __init__(self, paper_aspect: float = 1240 / 1754, parent=None):
        super().__init__(parent)
        self.paper_aspect = paper_aspect  # w/h
        self.builder = StrokeBuilder()
        self.cursor: tuple[float, float] | None = None
        self.cursor_visible = False
        self.pen_down = False
        self._stroke_color = QColor(25, 35, 60)
        self.setMinimumSize(320, 400)

    def add_point(self, x: float, y: float, pen_down: bool, coasting: bool = False) -> None:
        self.builder.add_point(x, y, pen_down, coasting)
        self.cursor = (x, y)
        self.cursor_visible = True
        self.pen_down = pen_down and not coasting
        self.update()

    def clear(self) -> None:
        self.builder.clear()
        self.update()

    def track_lost(self) -> None:
        self.builder.track_lost()
        self.update()

    def _paper_rect(self) -> QRectF:
        w = float(self.width())
        h = float(self.height())
        if w / h > self.paper_aspect:
            ph = h
            pw = h * self.paper_aspect
        else:
            pw = w
            ph = w / self.paper_aspect
        return QRectF((w - pw) / 2, (h - ph) / 2, pw, ph)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(52, 56, 64))
        rect = self._paper_rect()
        p.fillRect(rect, QColor(252, 251, 247))
        p.setPen(QPen(QColor(90, 95, 105), 1))
        p.drawRect(rect)

        pen = QPen(self._stroke_color, 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        for stroke in self.builder.strokes:
            if not stroke:
                continue
            prev = None
            for nx, ny in stroke:
                pt = QPointF(rect.left() + nx * rect.width(), rect.top() + ny * rect.height())
                if prev is not None:
                    p.drawLine(prev, pt)
                prev = pt

        if self.cursor_visible and self.cursor is not None:
            cx = rect.left() + self.cursor[0] * rect.width()
            cy = rect.top() + self.cursor[1] * rect.height()
            color = QColor(220, 60, 50) if self.pen_down else QColor(60, 130, 220)
            p.setPen(QPen(color, 2))
            p.drawEllipse(QPointF(cx, cy), 7, 7)
            p.drawLine(QPointF(cx - 11, cy), QPointF(cx + 11, cy))
            p.drawLine(QPointF(cx, cy - 11), QPointF(cx, cy + 11))
        p.end()
