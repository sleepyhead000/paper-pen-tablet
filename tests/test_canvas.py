import os

import pytest

from paperpen.output.canvas import StrokeBuilder


def test_stroke_lifecycle():
    b = StrokeBuilder()
    b.add_point(0.1, 0.1, pen_down=True)
    b.add_point(0.2, 0.2, pen_down=True)
    b.add_point(0.3, 0.3, pen_down=True)
    assert len(b.strokes) == 1
    assert len(b.strokes[0]) == 3
    b.add_point(0.4, 0.4, pen_down=False)  # lift
    b.add_point(0.5, 0.5, pen_down=False)  # hover move: no ink
    assert len(b.strokes) == 1
    b.add_point(0.6, 0.6, pen_down=True)  # new stroke
    assert len(b.strokes) == 2
    assert b.point_count == 4


def test_coasting_continues_open_stroke():
    b = StrokeBuilder()
    b.add_point(0.1, 0.1, pen_down=True)
    b.add_point(0.2, 0.2, pen_down=True)
    b.add_point(0.3, 0.3, pen_down=True, coasting=True)  # occlusion: predicted
    b.add_point(0.4, 0.4, pen_down=True, coasting=True)
    assert len(b.strokes) == 1
    assert b.point_count == 4


def test_coasting_never_starts_a_stroke():
    b = StrokeBuilder()
    b.add_point(0.1, 0.1, pen_down=False)  # lifted
    b.add_point(0.2, 0.2, pen_down=True, coasting=True)  # prediction while down
    assert b.strokes == []


def test_track_lost_closes_stroke():
    b = StrokeBuilder()
    b.add_point(0.1, 0.1, pen_down=True)
    b.add_point(0.2, 0.2, pen_down=True)
    b.track_lost()
    b.add_point(0.7, 0.7, pen_down=True)  # recovery after loss -> new stroke
    assert len(b.strokes) == 2
    assert b.strokes[0][-1][0] == pytest.approx(0.2)


def test_min_move_filter():
    b = StrokeBuilder(min_move=0.01)
    b.add_point(0.5, 0.5, pen_down=True)
    b.add_point(0.501, 0.501, pen_down=True)  # below threshold
    b.add_point(0.502, 0.502, pen_down=True)
    assert len(b.strokes[0]) == 1
    b.add_point(0.55, 0.5, pen_down=True)
    assert len(b.strokes[0]) == 2


def test_hover_never_starts_stroke():
    b = StrokeBuilder()
    b.add_point(0.1, 0.1, pen_down=False)
    b.add_point(0.2, 0.2, pen_down=False)
    assert b.strokes == []
    assert b.point_count == 0


def test_clear():
    b = StrokeBuilder()
    b.add_point(0.1, 0.1, pen_down=True)
    b.clear()
    assert b.strokes == []
    assert b.point_count == 0


@pytest.fixture(scope="session")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_canvas_widget_paints(qapp):
    from paperpen.output.canvas import CanvasWidget

    w = CanvasWidget(paper_aspect=0.7)
    w.resize(400, 560)
    w.add_point(0.2, 0.2, pen_down=True)
    w.add_point(0.6, 0.6, pen_down=True)
    w.add_point(0.8, 0.3, pen_down=False)
    img = w.grab().toImage()
    assert img.width() == 400 and img.height() == 560
    assert not img.isNull()
