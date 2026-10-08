from pathlib import Path

from paperpen.labeling import LabelStore


def test_add_and_reload(tmp_path: Path):
    p = tmp_path / "tips.csv"
    s = LabelStore(p, 1240, 1754)
    s.add(0, 1000.0, 123.4, 567.8)
    s.add(5, 1200.0, 10.0, 20.0)
    assert len(s) == 2
    again = LabelStore(p, 1240, 1754)  # crash-safe re-read
    assert len(again) == 2
    assert again.rows[0]["x"] == 123.4
    assert again.rows[0]["warp_width"] == 1240
    assert again.rows[1]["frame_id"] == 5


def test_undo(tmp_path: Path):
    p = tmp_path / "tips.csv"
    s = LabelStore(p, 100, 100)
    s.add(1, 1.0, 5.0, 6.0)
    s.add(2, 2.0, 7.0, 8.0)
    removed = s.undo()
    assert removed["frame_id"] == 2
    assert len(s) == 1
    assert LabelStore(p, 100, 100).rows[0]["frame_id"] == 1
    assert s.undo() is not None
    assert s.undo() is None
    assert len(s) == 0
