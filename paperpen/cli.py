from __future__ import annotations

import argparse
import json
import sys
import time

from .capture import make_source, parse_source_spec
from .capture.benchmark import format_report, measure_fps, measure_latency, to_json
from .config import default_config, load_config, save_config


def _resolve_source(args) -> object:
    cfg = load_config(args.config)
    if getattr(args, "source", None):
        cfg.source = parse_source_spec(args.source, cfg.source)
    if cfg.source.type == "file" and not cfg.source.path:
        raise SystemExit("no source given: pass --source or set source in config")
    if cfg.source.type == "http" and not cfg.source.url:
        raise SystemExit("http source needs a URL: pass --source http://IP:8080/video")
    return make_source(cfg.source)


def cmd_write_config(args) -> int:
    save_config(default_config(), args.path)
    print(f"wrote default config to {args.path}")
    return 0


def cmd_view(args) -> int:
    import cv2

    source = _resolve_source(args)
    frames = 0
    start = time.monotonic()
    fps = 0.0
    with source:
        print("showing live view; press q in the window to quit")
        while True:
            frame = source.next_frame(timeout=1.0)
            if frame is None:
                if not source.connected:
                    print("waiting for stream...", end="\r")
                continue
            frames += 1
            now = time.monotonic()
            if now - start >= 1.0:
                fps = frames / (now - start)
                frames, start = 0, now
            img = frame.image.copy()
            cv2.putText(
                img,
                f"{fps:.1f} fps  id={frame.frame_id}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("paperpen view", img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    return 0


def cmd_bench(args) -> int:
    source = _resolve_source(args)
    with source:
        # let the stream warm up so reconnection/rebuffering is not counted
        warm = time.monotonic() + 2.0
        while time.monotonic() < warm and source.next_frame(timeout=1.0) is not None:
            pass
        fps = measure_fps(source, args.seconds)
        latency = None
        if not args.no_qr:
            print("point the camera at the fullscreen QR window now...")
            latency = measure_latency(source, args.seconds)
    if args.json:
        print(to_json(fps, latency))
    else:
        print(format_report(fps, latency))
    return 0


def cmd_run(args) -> int:
    import cv2

    from .output import SessionLogger
    from .pipeline import Pipeline, draw_overlay

    cfg = load_config(args.config)
    if args.source:
        cfg.source = parse_source_spec(args.source, cfg.source)
    if args.no_log:
        cfg.log.dir = ""
    logger = SessionLogger(cfg.log) if not args.no_log else None
    source = make_source(cfg.source)
    pipeline = Pipeline(cfg, logger)
    stats = {"frames": 0, "paper": 0, "tip": 0}
    pen_down = False
    start = last = time.monotonic()
    processed = 0
    try:
        with source:
            while True:
                frame = source.next_frame(timeout=1.0)
                if frame is None:
                    if not source.live:
                        break  # recorded video hit EOF
                    if not source.connected and not args.show:
                        break
                    continue
                result = pipeline.step(frame, pen_down=pen_down)
                stats["frames"] += 1
                stats["paper"] += result.paper is not None
                stats["tip"] += result.tip is not None
                processed += 1
                now = time.monotonic()
                if now - last >= 2.0:
                    rate = processed / (now - last)
                    print(
                        f"pipeline {rate:.1f} fps | paper {stats['paper']}/{stats['frames']} "
                        f"tip {stats['tip']}/{stats['frames']}"
                    )
                    processed, last = 0, now
                if args.show:
                    cv2.imshow("paperpen run", draw_overlay(frame.image.copy(), result))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord(" "):
                        pen_down = not pen_down
                        print(f"pen_down -> {pen_down}")
                elif args.seconds and (now - start) > args.seconds:
                    break
    finally:
        if args.show:
            cv2.destroyAllWindows()
        if logger is not None:
            logger.close()
            print(f"logged {logger.rows} rows -> {logger.csv_path}")
    return 0


def cmd_app(args) -> int:
    from .ui import run_app

    cfg = load_config(args.config)
    if args.source:
        cfg.source = parse_source_spec(args.source, cfg.source)
    if args.mouse:
        cfg.mouse.enabled = True
    return run_app(cfg, log=not args.no_log)


def cmd_shapes(args) -> int:
    from .evaluate import generate_shapes

    truth = generate_shapes(args.out)
    print(f"wrote printable shapes for {list(truth)} -> {args.out}")
    print("print the PNG at 100% scale on A4 (no page scaling)")
    return 0


def cmd_score(args) -> int:
    from .evaluate import load_truth, score_trace

    truth = load_truth(args.truth)
    result = score_trace(args.csv, truth, args.shape)
    if not result.get("n"):
        print("no pen-down samples with coordinates found in the CSV")
        return 1
    print(json.dumps({**result, "shape": args.shape}, indent=2))
    return 0


def cmd_score_pendown(args) -> int:
    from .evaluate.pendown import score_pen_down

    result = score_pen_down(args.csv)
    print(json.dumps(result, indent=2))
    best = max(
        ("contact_shadow", "contact_ink"),
        key=lambda m: result[m]["f1"],
    )
    print(
        f"winner by F1: {best} "
        f"(f1={result[best]['f1']}, "
        f"shadow f1={result['contact_shadow']['f1']}, "
        f"ink f1={result['contact_ink']['f1']})"
    )
    return 0


def cmd_label(args) -> int:
    """Interactive tip-label collection for M5: show the warped paper view,
    click the tip, save, advance. Labels land in warped pixel coords."""
    import cv2

    from .labeling import LabelStore
    from .pipeline import Pipeline

    cfg = load_config(args.config)
    if args.source:
        cfg.source = parse_source_spec(args.source, cfg.source)
    source = make_source(cfg.source)
    # labels only need the warp, not the main session CSV
    pipeline = Pipeline(cfg, logger=None)
    store = LabelStore(args.out, cfg.paper.warp_width, cfg.paper.warp_height)
    pending: list[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.clear()
            pending.append((float(x), float(y)))

    win = "paperpen label"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("click = mark tip | s = save+next | n = next (unlabeled) | u = undo | q = quit")
    try:
        with source:
            while True:
                frame = source.next_frame(timeout=1.0)
                if frame is None:
                    if not source.live:
                        print("end of video")
                        break
                    continue
                result = pipeline.step(frame)
                view = result.warped
                if view is None:
                    continue
                if pending:
                    cx, cy = int(pending[0][0]), int(pending[0][1])
                    cv2.drawMarker(view, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.putText(
                    view,
                    f"labels={len(store)} frame={frame.frame_id} "
                    f"conf={result.confidence:.2f}",
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 120, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(win, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s") and pending:
                    x, y = pending.pop()
                    store.add(frame.frame_id, frame.t_wall * 1000.0, x, y)
                    print(f"saved label #{len(store)} at ({x:.0f},{y:.0f})")
                elif key == ord("n"):
                    pending.clear()
                elif key == ord("u"):
                    removed = store.undo()
                    print(f"undid {removed}" if removed else "nothing to undo")
    finally:
        cv2.destroyAllWindows()
    print(f"{len(store)} labels -> {store.path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paperpen", description="Paper-Pen Tablet")
    p.add_argument("--config", default=None, help="path to config JSON")
    sub = p.add_subparsers(dest="cmd", required=True)

    wc = sub.add_parser("write-config", help="write default config JSON")
    wc.add_argument("path")
    wc.set_defaults(func=cmd_write_config)

    v = sub.add_parser("view", help="live camera view with FPS readout")
    v.add_argument("--source", help="webcam | N | http://IP:PORT/path | video file")
    v.set_defaults(func=cmd_view)

    b = sub.add_parser("bench", help="measure capture FPS and end-to-end latency")
    b.add_argument("--source", help="webcam | N | http://IP:PORT/path | video file")
    b.add_argument("--seconds", type=float, default=15.0)
    b.add_argument("--no-qr", action="store_true", help="FPS only, skip latency")
    b.add_argument("--json", action="store_true", help="machine-readable output")
    b.set_defaults(func=cmd_bench)

    r = sub.add_parser("run", help="run the full pipeline with overlay and logging")
    r.add_argument("--source", help="webcam | N | http://IP:PORT/path | video file")
    r.add_argument("--show", action="store_true", help="live debug overlay window")
    r.add_argument("--no-log", action="store_true", help="skip CSV logging")
    r.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = forever)")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("app", help="launch the drawing app (PySide6)")
    a.add_argument("--source", help="webcam | N | http://IP:PORT/path | video file")
    a.add_argument("--no-log", action="store_true", help="skip CSV logging")
    a.add_argument("--mouse", action="store_true", help="control the system pointer")
    a.set_defaults(func=cmd_app)

    g = sub.add_parser("shapes", help="generate printable test shapes + ground truth")
    g.add_argument("out", nargs="?", default="data/shapes")
    g.set_defaults(func=cmd_shapes)

    sc = sub.add_parser("score", help="score a traced session against ground truth")
    sc.add_argument("csv", help="session CSV from a trace run")
    sc.add_argument("--shape", required=True, choices=["grid", "circle", "lines"])
    sc.add_argument("--truth", default="data/shapes/shapes.json")
    sc.set_defaults(func=cmd_score)

    sp = sub.add_parser(
        "score-pendown", help="score shadow/ink contact signals vs key ground truth"
    )
    sp.add_argument("csv", help="session .pendown.csv from an experiment run")
    sp.set_defaults(func=cmd_score_pendown)

    lb = sub.add_parser("label", help="collect tip keypoint labels (M5 dataset)")
    lb.add_argument("--source", help="webcam | N | http://IP:PORT/path | video file")
    lb.add_argument("--out", default="data/dataset/tips.csv")
    lb.set_defaults(func=cmd_label)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
