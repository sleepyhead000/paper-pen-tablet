from __future__ import annotations

from ..config import MouseConfig


class _PynputBackend:
    def __init__(self):
        from pynput.mouse import Button, Controller

        self._mouse = Controller()
        self._Button = Button

    @property
    def position(self) -> tuple[int, int]:
        p = self._mouse.position
        return (int(p[0]), int(p[1]))

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        self._mouse.position = (int(value[0]), int(value[1]))

    def press(self) -> None:
        self._mouse.press(self._Button.left)

    def release(self) -> None:
        self._mouse.release(self._Button.left)


def _screen_size() -> tuple[int, int]:
    try:
        import ctypes

        w = int(ctypes.windll.user32.GetSystemMetrics(0))
        h = int(ctypes.windll.user32.GetSystemMetrics(1))
        if w > 0 and h > 0:
            return w, h
    except Exception:  # noqa: BLE001
        pass
    import pyautogui

    size = pyautogui.size()
    return int(size[0]), int(size[1])


class MouseController:
    """Normalized paper coords [0,1] -> screen pointer.

    Safety rules from PRD §5: on tracking loss hold the last position and
    release the button, so a lost tip never drags a line across the screen.
    """

    def __init__(
        self,
        cfg: MouseConfig | None = None,
        screen: tuple[int, int] | None = None,
        backend=None,
    ):
        self.cfg = cfg or MouseConfig()
        self.screen = screen or _screen_size()
        self.backend = backend if backend is not None else _PynputBackend()
        self.pressed = False
        self.moves = 0
        self.presses = 0

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        w, h = self.screen
        m = self.cfg.screen_margin
        sx = m + min(max(x, 0.0), 1.0) * max(w - 2 * m, 1)
        sy = m + min(max(y, 0.0), 1.0) * max(h - 2 * m, 1)
        return (int(min(max(sx, 0), w - 1)), int(min(max(sy, 0), h - 1)))

    def update(self, x: float, y: float, pen_down: bool, valid: bool) -> None:
        if not self.cfg.enabled:
            return
        if not valid:
            # tracking lost: hold position, never keep the button held
            if self.pressed:
                self.backend.release()
                self.pressed = False
            return
        self.backend.position = self.to_screen(x, y)
        self.moves += 1
        if self.cfg.hover_only:
            if self.pressed:
                self.backend.release()
                self.pressed = False
            return
        if pen_down and not self.pressed:
            self.backend.press()
            self.pressed = True
            self.presses += 1
        elif not pen_down and self.pressed:
            self.backend.release()
            self.pressed = False

    def stop(self) -> None:
        if self.pressed:
            self.backend.release()
            self.pressed = False
