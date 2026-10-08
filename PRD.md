# Paper-Pen Tablet: Project Plan

Status: draft v1. Items marked **[UNKNOWN]** are things I do not know yet and that need a test, not an assumption.

## 1. Goal
Use a camera (webcam or phone) pointed at white paper to track the tip of a normal pen and act as a graphics tablet, with no extra hardware.

Outputs:
1. Drawing canvas inside the app
2. Mouse/pointer control for other apps
3. Log of exact coordinates to a file (for testing)

Constraints: normal pen only (no markers, no modified pen). White paper.

## 2. Tech recommendation

| | Stack | Why |
|---|---|---|
| **Fastest prototype** | Python + OpenCV + NumPy, PyQt or pygame for canvas, pynput/pyautogui for mouse | Quickest to iterate on vision; huge CV ecosystem |
| **Good for the future** | Train the tip detector in PyTorch, export to ONNX; later move to a native app (C++/Rust + ONNX Runtime, or Qt) or a web app (onnxruntime-web) | Keeps the model portable; Python is too slow/awkward to ship as a low-latency product |

My judgment: start in Python, keep vision code separate from UI so it can be ported. I have not benchmarked this on your hardware, so latency is **[UNKNOWN]** until tested.

## 3. Pipeline
1. **Capture**: camera frames, fixed resolution, fixed exposure/focus if the camera allows it.
2. **Paper detection and calibration**: find the 4 paper corners, compute a perspective transform (homography) to a flat paper coordinate system. Re-check periodically for paper/camera movement.
3. **Pen tip detection**: find the tip position in each frame.
4. **Pen-down detection**: decide if the tip touches the paper or hovers.
5. **Filtering**: smooth jitter (e.g. One Euro filter or Kalman filter), predict through short occlusions.
6. **Mapping**: paper coordinates → canvas / screen coordinates.
7. **Output**: canvas strokes, mouse events, log file.

## 4. Hardest problems and options

### 4.1 Tip detection with a normal pen
No marker means classic color tracking will not work reliably. Options:
- **A. Train a small keypoint/detection model** on labeled frames of your own pen + hand + paper. Most robust option I expect, but needs data collection and labeling. How much data is enough is **[UNKNOWN]**.
- **B. Classic CV** (edges/contours, pen is the elongated object entering the frame, tip = furthest point from the hand). Fast to try, probably fragile with different pens/grips.
- Suggested path: try B for a quick baseline, move to A.

### 4.2 Pen-down vs hover (biggest risk)
One camera is 2D, so contact is hard to see. Options, all **[UNKNOWN]** in reliability until tested:
- **Shadow gap**: the tip and its shadow merge on contact. Depends on lighting.
- **Ink detection**: new ink appearing under the tip proves contact. Only works if the pen writes, and has a delay.
- **Second camera** (e.g. phone + webcam at different angles): gives depth, but adds hardware/setup, which you want to avoid.
- **Learned classifier**: train a model to classify touching vs hovering from a crop around the tip. Needs labeled data.
- **Fallback**: always-draw mode with a keyboard key or on-screen button to lift. Not ideal but works.

### 4.3 Occlusion
Hand and fingers hide the tip, especially with a top-down camera. Camera angle matters (see questions).

## 5. Things to be careful about
- **Latency**: aim for low end-to-end delay; target numbers are **[UNKNOWN]** until measured. Test the full camera→screen delay, not just the code.
- **Camera frame rate**: low FPS in dim light, auto-exposure changes, rolling shutter blur on fast strokes.
- **Motion blur** on the tip during fast writing.
- **Jitter vs lag trade-off**: more smoothing = less jitter but more lag.
- **Lighting**: shadows from your hand, glare on paper, flicker from lamps.
- **Paper movement**: if the paper shifts, all coordinates are wrong. Re-detect corners or track paper continuously.
- **Perspective accuracy**: small tilt errors become large position errors at paper edges.
- **Tip is tiny**: at low resolution the tip may be a few pixels; precision is limited by camera resolution.
- **Pen variety**: different pens, colors, and caps. Test several.
- **Writing already on paper**: ink marks can confuse paper and tip detection.
- **Hand rejection**: palm or other fingers must not be mistaken for the pen.
- **Phone camera**: needs a stable mount; connection method (USB / Wi-Fi stream / virtual webcam) adds latency. **[UNKNOWN]** which is best for you.
- **Mouse control**: OS permissions (macOS accessibility, Wayland restrictions on Linux). Mouse jumps when tracking is lost, so hold the last position or disable output.
- **Privacy**: camera feed is sensitive, process locally.
- **No pressure data**: a camera cannot easily give pen pressure. **Idea**: estimate it from tilt, stroke speed, or ink line width, but treat as experimental.

## 6. Log format (for testing)
CSV or JSONL, one row per frame:
`timestamp_ms, frame_id, tip_x_px, tip_y_px, paper_x, paper_y, pen_down, confidence`
Also save the raw video optionally, so tracking can be re-run and compared offline.

## 7. Milestones
1. **Setup + capture**: camera feed, FPS and latency measurement.
2. **Paper calibration**: corner detection, top-down warped view.
3. **Tip detection v1** (classic CV) with live overlay and logging.
4. **Dataset + model** for tip detection (if v1 is not good enough).
5. **Pen-down experiments**: compare methods from 4.2 and pick one.
6. **Smoothing + canvas drawing**.
7. **Mouse mode**.
8. **Evaluation**: accuracy tests (draw known shapes: lines, circles, a grid) and measure error against the ground truth.
9. **Polish / port** to the long-term stack.

## 8. How we measure success
- Position error (mm on paper) tracing printed shapes
- End-to-end latency (ms)
- Pen-down accuracy (false touches / missed touches)
- Tracking loss rate over a writing session
- Works under at least 3 lighting conditions and 3 different pens

## 9. Open questions
1. Operating system (Windows / macOS / Linux)?
2. Camera: laptop webcam, phone, or both? Camera position: looking straight down at the paper, or at an angle from the front?
3. Does your pen actually write ink during use, or might it be capped/dry? (affects ink-based pen-down detection)
4. Is a keyboard key / button as a fallback for pen-down acceptable?
5. Is this for personal use, or do you plan to distribute it?
