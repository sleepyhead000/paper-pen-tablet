# Testing Guide

Walk through this top to bottom. Sections 1–3 need no hardware; 4–7 use the
phone, paper, and pen. Every command runs from the repo root in PowerShell.

## 1. Automated suite (no hardware)

```powershell
.\.venv\Scripts\python -m pytest -q
```

Expected: `125 passed` in a few seconds. Anything failing blocks progress —
fix first.

Optional lint (install once with
`.\.venv\Scripts\python -m pip install pyflakes`):

```powershell
.\.venv\Scripts\python -m pyflakes paperpen tests
```

Expected: no output.

## 2. Headless end-to-end self-test (no hardware)

Generates a synthetic clip (paper quad + moving pen + ink line) and runs the
full pipeline over it:

```powershell
.\.venv\Scripts\python -c @'
import cv2, numpy as np, os
os.makedirs('data/e2e', exist_ok=True)
vw = cv2.VideoWriter('data/e2e/synth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640,480))
quad = np.array([[175,30],[475,45],[495,455],[155,440]], np.int32)
for i in range(30):
    img = np.full((480,640,3), 45, np.uint8)
    cv2.fillConvexPoly(img, quad, (236,236,236))
    cv2.line(img,(185,385),(470,405),(30,30,30),4)
    tx, ty = 250 + int(40*np.sin(i/5)), 200
    a, L, W, T = np.radians(30), 190, 22, 24
    ax = np.array([np.cos(a), np.sin(a)]); pe = np.array([-ax[1], ax[0]])
    t = np.array([tx, ty], float); base = t + ax*T; back = t + ax*L
    cv2.fillConvexPoly(img, np.array([t, base+pe*W, back+pe*W, back-pe*W, base-pe*W], np.int32), (30,30,36))
    vw.write(img)
vw.release()
print('video written')
'@
.\.venv\Scripts\python -m paperpen run --source data/e2e/synth.mp4 --seconds 30
```

**Pass:** prints `logged 30 rows -> data\logs\session_*.csv`, then check the
tip actually tracked the pen (x sweeps ~250→287 across frames, confidence
~0.8, y ≈ 200):

```powershell
Get-Content (Get-ChildItem data\logs\session_*.csv | Sort-Object LastWriteTime | Select-Object -Last 1) | Select-Object -First 6
```

Fail patterns: empty `tip_x_px` (detection failing) or constant x at a line
position (detector locked onto the printed ink line instead of the pen).

## 3. Camera bench (webcam sanity)

```powershell
.\.venv\Scripts\python -m paperpen view          # q in the window to quit
.\.venv\Scripts\python -m paperpen bench --seconds 10 --no-qr
```

**Pass:** FPS ≥ ~10 for a webcam in a lit room. A dark/covered lens gives
0–7 fps — clean the lens, add light. (On this machine MSMF device opening is
very slow; if `view` hangs >10 s on a webcam, that's the known driver issue —
phone streaming avoids it entirely.)

## 4. Phone stream (M1 acceptance)

1. Play Store → **IP Webcam** → start server → note `http://<ip>:8080`.
2. Mount the phone looking down at the paper, ~25–40 cm away, no glare.
3. Then:

```powershell
.\.venv\Scripts\python -m paperpen view --source http://<phone-ip>:8080/video
.\.venv\Scripts\python -m paperpen bench --source http://<phone-ip>:8080/video --seconds 15
```

`bench` opens a **QR code window on your monitor — point the phone at the
monitor for the latency phase** (it measures glass-to-glass: display →
camera → stream → decode).

**Record:** FPS and latency ms (the PRD targets are still `[UNKNOWN]` until
this run — write the numbers into PRD §8). Also note `decode failures`.

## 5. Tip tracking live (M3)

```powershell
.\.venv\Scripts\python -m paperpen run --source http://<phone-ip>:8080/video --show
```

A debug window shows the warped paper view with the detected tip crosshair.
Move the pen over the paper (different angles, speeds, a second pen if you
have one). Press `q` in the window when done — note that `--seconds` only
applies without `--show`.

**Pass:** crosshair sticks to the real tip most of the time (≥ ~90% of frames
with a tip visible in the CSV). **Check known failure modes:**
- Locks onto writing already on the page → test on a fresh sheet; raise
  `tip.bin_threshold` if needed.
- Locks onto your finger → `tip.skin_reject` should be on (default).
- Confuses shadow with pen → lighting more even / camera closer to vertical.

Optionally repeat with `--show` under 3 lighting conditions and 2–3 pens
(PRD §8).

## 6. Shapes accuracy (M8)

```powershell
.\.venv\Scripts\python -m paperpen shapes            # -> data/shapes/*.png + shapes.json
# print data/shapes/grid.png (or circle/lines) on a full A4 sheet, put it under the camera
.\.venv\Scripts\python -m paperpen app --source http://<phone-ip>:8080/video
# trace the printed shape on the paper: hold Space to draw (the "Draw (hold
# Space)" checkbox is on by default); "Save PNG" exports the canvas image;
# close the window to end the session, then score:
.\.venv\Scripts\python -m paperpen score data\logs\session_XXXX.csv --shape grid
```

`app` draws into the canvas window; the matching session CSV is the newest
file in `data/logs/`.

**Pass (suggested):** median error under ~2 mm, 95th percentile under ~5 mm,
tracking-loss low (reported in the score output). Report the actual numbers —
these thresholds are your call to tighten once you see real data.

## 7. Pen-down shootout (M6)

Enable signal logging in `config.json`:

```json
"pendown": { "enabled": true }
```

Then run a session where you **toggle Space in sync with touching / lifting
the pen** (Space state is ground truth: toggle in `run --show`, hold-to-draw
in `app`):

```powershell
.\.venv\Scripts\python -m paperpen run --source http://<phone-ip>:8080/video --show
```

Each run writes `data/logs/session_*.pendown.csv`. Score it:

```powershell
.\.venv\Scripts\python -m paperpen score-pendown data\logs\session_XXXX.pendown.csv
```

Output: precision/recall/F1 for `contact_shadow`, `contact_ink`, and the
combined signals.

**Decision rule:** pick the method with F1 ≥ ~0.9 across a few sessions and
lighting setups; if neither gets there, fall back to the Space-key method and
note it (PRD §4.2 fallback). Repeat under 2–3 lighting conditions — the
shadow method is lighting-sensitive by design.

## 8. Drawing app + mouse safety (M4/M7)

```powershell
.\.venv\Scripts\python -m paperpen app --source http://<phone-ip>:8080/video
.\.venv\Scripts\python -m paperpen app --source http://<phone-ip>:8080/video --mouse
```

**Pass:**
- Without `--mouse`: strokes appear in the canvas while you hold Space.
- With `--mouse`: pointer maps to the paper, left button follows pen-down,
  and — safety check — when tracking is lost (cover the lens or lift the pen
  out of frame) the pointer **stops and releases**, it does not jump.

## 9. Labeling smoke test (M5, only if needed)

```powershell
.\.venv\Scripts\python -m paperpen label --source http://<phone-ip>:8080/video --out data/dataset/tips.csv
```

Click = mark tip, `s` = save+advance, `n` = skip, `u` = undo, `q` = quit.
Only collect this dataset if sections 5–7 show classic CV is not good enough
(the model path is conditional).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `view`/`bench` hangs on webcam | known MSMF slowness — use the phone stream |
| FPS very low | more light, wipe lens, close other camera apps |
| Paper not detected | white A4 filling ≥ ~30% of frame, `paper.min_brightness` (90) assumes decent light |
| Coordinates wrong after moving paper | normal — full redetect happens every `paper.detect_interval_s` (2 s) |
| Phantom mouse movement | keep `mouse.enabled` off unless actively using `--mouse` |
| Phone stream stutters | same Wi-Fi, disable phone sleep, lower IP Webcam resolution |

When all sections pass, the remaining PRD §8 metrics (latency, mm error,
pen-down F1, loss rate, lighting/pen matrix) are exactly what was measured —
paste them into the PRD and the project is evaluation-complete.
