from __future__ import annotations

import json
import time

import cv2
import numpy as np

from .source import FrameSource

_QR_PREFIX = "PP1:"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(sorted(values), dtype=np.float64)
    idx = min(int(round(p / 100.0 * (len(arr) - 1))), len(arr) - 1)
    return float(arr[idx])


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "median_ms": round(_percentile(values, 50), 2),
        "p90_ms": round(_percentile(values, 90), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


def measure_fps(source: FrameSource, duration_s: float = 10.0) -> dict:
    start = time.monotonic()
    deadline = start + duration_s
    frames = 0
    gaps: list[float] = []
    last_t: float | None = None
    while time.monotonic() < deadline:
        frame = source.next_frame(timeout=1.0)
        if frame is None:
            continue
        frames += 1
        if last_t is not None:
            gaps.append((frame.t_recv - last_t) * 1000.0)
        last_t = frame.t_recv
    elapsed = time.monotonic() - start
    return {
        "duration_s": round(elapsed, 2),
        "frames": frames,
        "fps": round(frames / elapsed, 2) if elapsed > 0 else 0.0,
        "frame_interval_ms": summarize(gaps),
        "connected": source.connected,
    }


def make_qr_image(payload: str, size: int = 512) -> np.ndarray:
    enc = cv2.QRCodeEncoder_create()
    qr = enc.encode(payload)  # uint8, 0 = dark module, 255 = light
    resized = cv2.resize(qr, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def decode_latency(frame, detector) -> float | None:
    """Return latency in ms for a frame carrying the QR payload, else None."""
    data, _, _ = detector.detectAndDecode(frame.image)
    if not data or not data.startswith(_QR_PREFIX):
        return None
    try:
        t_shown = float(data[len(_QR_PREFIX) :])
    except ValueError:
        return None
    return (frame.t_recv - t_shown) * 1000.0


def measure_latency(
    source: FrameSource,
    duration_s: float = 15.0,
    qr_size_frac: float = 0.4,
    window_name: str = "paperpen latency bench",
) -> dict:
    """Glass-to-glass-ish latency: display a QR of the current monotonic
    time fullscreen, read it back through the camera.

    latency = frame receive time - time the displayed QR was generated.
    Includes: QR encode+paint, display scanout, camera exposure/readout,
    stream transport, decode-on-arrival of the frame. Decode time after
    arrival is excluded (receive time is stamped at capture).
    """
    size = int(512 * qr_size_frac / 0.4)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    detector = cv2.QRCodeDetector()
    latencies: list[float] = []
    decode_fail = 0
    reads = 0
    start = time.monotonic()
    deadline = start + duration_s
    try:
        while time.monotonic() < deadline:
            t_gen = time.monotonic()
            cv2.imshow(window_name, make_qr_image(f"{_QR_PREFIX}{t_gen:.6f}", size))
            cv2.waitKey(1)
            frame = source.next_frame(timeout=1.0)
            if frame is None:
                continue
            reads += 1
            lat_ms = decode_latency(frame, detector)
            if lat_ms is None:
                decode_fail += 1
                continue
            latencies.append(lat_ms)
    finally:
        cv2.destroyWindow(window_name)

    result = {
        "duration_s": round(time.monotonic() - start, 2),
        "reads": reads,
        "decode_fail": decode_fail,
        "latency": summarize(latencies),
        "includes": "qr encode+paint, display scanout, camera, transport (excludes post-receive decode)",
    }
    return result


def format_report(fps: dict | None, latency: dict | None) -> str:
    lines = []
    if fps is not None:
        lines.append(f"FPS: {fps['fps']} over {fps['duration_s']}s ({fps['frames']} frames)")
        iv = fps.get("frame_interval_ms", {})
        if iv.get("n"):
            lines.append(
                f"  frame interval: median {iv['median_ms']} ms, p90 {iv['p90_ms']} ms"
            )
    if latency is not None:
        lat = latency.get("latency", {})
        if lat.get("n"):
            lines.append(
                f"Latency: median {lat['median_ms']} ms, p90 {lat['p90_ms']} ms, "
                f"min {lat['min_ms']} ms, max {lat['max_ms']} ms ({lat['n']} samples)"
            )
        else:
            lines.append("Latency: no samples decoded")
        lines.append(
            f"  QR reads: {latency['reads']}, decode failures: {latency['decode_fail']}"
        )
        lines.append(f"  {latency['includes']}")
    return "\n".join(lines)


def to_json(fps: dict | None, latency: dict | None) -> str:
    return json.dumps({"fps": fps, "latency": latency}, indent=2)
