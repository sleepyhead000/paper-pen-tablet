from pathlib import Path

import cv2
import numpy as np
import pytest

from paperpen.capture import VideoFileSource, make_source, parse_source_spec
from paperpen.config import SourceConfig


def _write_video(path: Path, n: int = 10, size: int = 64) -> Path:
    vw = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (size, size)
    )
    assert vw.isOpened()
    for i in range(n):
        img = np.full((size, size, 3), i * 20, np.uint8)
        vw.write(img)
    vw.release()
    return path


def test_video_file_sequence(tmp_path: Path):
    path = _write_video(tmp_path / "t.mp4")
    with VideoFileSource(str(path)) as src:
        ids, times = [], []
        while True:
            f = src.next_frame(timeout=0.5)
            if f is None:
                break
            ids.append(f.frame_id)
            times.append(f.t_recv)
        assert len(ids) >= 8
        assert ids == sorted(ids)
        assert all(t2 >= t1 for t1, t2 in zip(times, times[1:]))
        assert src.next_frame(timeout=0.1) is None  # EOF stays EOF


def test_video_file_rewind(tmp_path: Path):
    path = _write_video(tmp_path / "t.mp4")
    with VideoFileSource(str(path)) as src:
        first = src.next_frame()
        assert first is not None
        src.rewind()
        again = src.next_frame()
        assert again is not None
        assert again.frame_id == 0


def test_make_source_file(tmp_path: Path):
    path = _write_video(tmp_path / "t.mp4")
    src = make_source(SourceConfig(type="file", path=str(path)))
    assert isinstance(src, VideoFileSource)


def test_make_source_validation():
    with pytest.raises(ValueError):
        make_source(SourceConfig(type="http", url=""))
    with pytest.raises(ValueError):
        make_source(SourceConfig(type="bogus"))


def test_parse_source_spec(tmp_path: Path):
    base = SourceConfig()
    assert parse_source_spec("webcam", base).type == "webcam"
    assert parse_source_spec("2", base).device == 2
    http = parse_source_spec("http://1.2.3.4:8080/video", base)
    assert http.type == "http" and http.url == "http://1.2.3.4:8080/video"
    vid = _write_video(tmp_path / "v.mp4")
    f = parse_source_spec(str(vid), base)
    assert f.type == "file" and f.path == str(vid)
    with pytest.raises(ValueError):
        parse_source_spec("nonsense", base)


def test_context_manager_closes(tmp_path: Path):
    path = _write_video(tmp_path / "t.mp4")
    src = VideoFileSource(str(path))
    with src:
        assert src.next_frame() is not None
    assert src._cap is None
