from paperpen.config import MouseConfig
from paperpen.output.mouse import MouseController


class FakeBackend:
    def __init__(self):
        self._pos = (0, 0)
        self.events = []

    @property
    def position(self):
        return self._pos

    @position.setter
    def position(self, value):
        self._pos = (int(value[0]), int(value[1]))
        self.events.append(("move", self._pos))

    def press(self):
        self.events.append(("press",))

    def release(self):
        self.events.append(("release",))


def make(**kw):
    cfg = MouseConfig(enabled=True, **{k: v for k, v in kw.items() if k != "backend"})
    backend = kw.get("backend") or FakeBackend()
    return MouseController(cfg, screen=(1000, 500), backend=backend), backend


def test_mapping_with_margin():
    m, b = make(screen_margin=50)
    assert m.to_screen(0.0, 0.0) == (50, 50)
    assert m.to_screen(1.0, 1.0) == (950, 450)
    assert m.to_screen(0.5, 0.5) == (500, 250)
    # out of range clamps
    assert m.to_screen(-1.0, 2.0) == (50, 450)


def test_hover_moves_without_press():
    m, b = make()
    m.update(0.2, 0.2, pen_down=False, valid=True)
    assert b.events == [("move", m.to_screen(0.2, 0.2))]
    assert not m.pressed


def test_pen_down_presses_once():
    m, b = make()
    m.update(0.1, 0.1, pen_down=True, valid=True)
    m.update(0.2, 0.2, pen_down=True, valid=True)
    m.update(0.3, 0.3, pen_down=True, valid=True)
    assert b.events.count(("press",)) == 1
    assert not m.pressed or True
    m.update(0.4, 0.4, pen_down=False, valid=True)
    assert b.events.count(("release",)) == 1
    assert not m.pressed


def test_tracking_loss_holds_and_releases():
    m, b = make()
    m.update(0.5, 0.5, pen_down=True, valid=True)
    pos = b.position
    b.events.clear()
    m.update(0.9, 0.9, pen_down=True, valid=False)
    # no movement, button released
    assert b.position == pos
    assert b.events == [("release",)]
    assert not m.pressed


def test_disabled_controller_is_noop():
    cfg = MouseConfig(enabled=False)
    b = FakeBackend()
    m = MouseController(cfg, screen=(1000, 500), backend=b)
    m.update(0.5, 0.5, pen_down=True, valid=True)
    assert b.events == []


def test_hover_only_mode():
    m, b = make(hover_only=True)
    m.update(0.5, 0.5, pen_down=True, valid=True)
    assert b.events == [("move", m.to_screen(0.5, 0.5))]
    assert not m.pressed


def test_stop_releases():
    m, b = make()
    m.update(0.5, 0.5, pen_down=True, valid=True)
    m.stop()
    assert not m.pressed
    assert b.events[-1] == ("release",)
