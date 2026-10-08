import pytest

import paperpen
from paperpen.cli import main


def test_version():
    assert paperpen.__version__


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_write_config(tmp_path):
    path = tmp_path / "config.json"
    assert main(["write-config", str(path)]) == 0
    assert path.exists()


def test_qr_roundtrip():
    import cv2

    from paperpen.capture.benchmark import _QR_PREFIX, make_qr_image

    img = make_qr_image(f"{_QR_PREFIX}123.456", size=400)
    data, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    assert data == f"{_QR_PREFIX}123.456"
