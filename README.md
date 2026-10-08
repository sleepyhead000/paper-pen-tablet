# Paper-Pen Tablet

Point a camera at a sheet of white paper, hold a normal pen — and use it as a
graphics tablet. No digitizer, no modified pen, no extra hardware beyond a
camera (your Android phone works fine).

**Outputs:**

1. A drawing canvas inside the app
2. Mouse/pointer control for other apps
3. A coordinate log for testing and evaluation

Vision code is decoupled from the UI so it can be ported later
(Python + OpenCV first; ONNX/native later if needed). See [PRD.md](PRD.md) for
the full project plan.

## How it works

```
camera frame
   └─> paper detection ──> homography warp (top-down A4 view)
         └─> pen tip detection (classic CV: contours + width-taper)
               └─> pen-down signals (shadow gap / fresh ink, experimental)
                     └─> One Euro filter + coasting through occlusions
                           └─> canvas strokes / mouse events / CSV log
```

## Quickstart (Windows, Python 3.12+)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m paperpen write-config config.json   # defaults
```

> The venv is not part of the repo. If `.venv` already exists locally, skip
> creating it.

### Commands

| Command | What it does |
|---|---|
| `paperpen view` | live camera view with FPS readout |
| `paperpen bench` | capture FPS + end-to-end (glass-to-glass) latency |
| `paperpen run --show` | full pipeline with debug overlay + CSV logging |
| `paperpen app` | the drawing app (PySide6 canvas, Space = pen key) |
| `paperpen app --mouse` | same, but drives the system mouse |
| `paperpen shapes` | generate printable test shapes + ground truth |
| `paperpen score <csv> --shape grid` | score a traced session (mm error) |
| `paperpen score-pendown <csv>` | score contact signals vs key ground truth |
| `paperpen label` | collect tip keypoint labels (dataset for M5) |
| `paperpen bench --source http://IP:8080/video` | bench a phone stream |

All commands accept `--source`:

```
--source webcam          # default local webcam
--source 1               # camera index
--source http://IP:8080/video    # IP Webcam (Android)
--source path\to\clip.mp4        # video file (offline re-runs)
```

### Phone as the camera (recommended)

1. Install **IP Webcam** from the Play Store on the phone.
2. Start the server (defaults: port `8080`) and note the IPv4 address.
3. Mount the phone looking straight down at the paper (stable, no glare).
4. Stream URL: `http://<phone-ip>:8080/video`

```powershell
.\.venv\Scripts\python -m paperpen view --source http://192.168.1.42:8080/video
```

## Configuration

`config.json` holds every tunable (written by `write-config`, and re-read each
run). Sections: `source`, `paper`, `tip`, `filter`, `log`, `mouse`, `pendown`.
Defaults target A4 paper at ~150 DPI (`warp_width: 1240`, `warp_height: 1754`).
Use `paperpen --config path\to\other.json <cmd>` for alternate setups.

Logs land in `data/logs/*.csv` (format from PRD §6, one row per frame). The
whole `data/` directory is gitignored.

## Repo layout

```
paperpen/
  capture/      FrameSource (webcam / phone HTTP / video file) + FPS/latency bench
  calibrate/    paper corner detection, homography, PaperTracker (flow between redetects)
  track/        tip detector (classic CV), pen-down signals, smoothing filter
  output/       CSV logger, stroke canvas widget, mouse controller
  evaluate/     printable ground-truth shapes + scorers (trace, pen-down)
  ui/           PySide6 main window + worker thread
  cli.py        command line entry point
  config.py     typed config dataclasses, JSON (de)serialization
  pipeline.py   frame -> paper -> tip -> normalized coords
tests/          125 unit/integration tests (pytest)
```

## Status

Milestones M0–M8 code-complete (see PRD §7); everything is exercised by
synthetic scenes/video in CI-style tests. Still **pending on hardware**:
phone latency measurement, real-paper tracing accuracy, and the pen-down
method shootout.

**Testing guide: [TESTING.md](TESTING.md)** — run the automated suite, then
walk through the live experiments in order.
