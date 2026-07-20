# DissTraqt System Architecture

## 🏗️ High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     DissTraqt Monitoring System                 │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│   Video Input    │
│ • File / Webcam  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│             detector_web.py / detector_core.py           │
│                                                          │
│  1. YOLO Tracking (Person + Phone proximity)             │
│  2. MediaPipe FaceMesh + Person Head-Crop Fallback       │
│  3. 3D solvePnP Head Pose (with Perspective Compensation)│
│  4. Iris Gaze Offset Estimation                          │
│  5. Dynamic Config Sync with Flask API                    │
└────────┬─────────────────────────────────────────────────┘
         │ HTTP POST /api/test/add-violation (Violations)
         │ HTTP POST /api/update-state (Live Frame & Heartbeat)
         │ HTTP GET /api/config (Live Config Poll)
         ▼
┌──────────────────────────────────────────────────────────┐
│              app.py (Flask Web Server)                   │
│                                                          │
│  • Manages thread-safe MonitoringState                   │
│  • Serves REST API & saves screenshots / JSON reports    │
│  • Handles live dynamic configuration (/api/config)     │
└────────┬─────────────────────────────────────────────────┘
         │ HTTP GET /api/summary & SSE / Polling
         ▼
┌──────────────────────────────────────────────────────────┐
│          dashboard.html (Web Dashboard Interface)        │
│                                                          │
│  • Real-time Statistics Cards & Person Tracking List     │
│  • Live Sensitivity Sliders (Syncs to Detector)          │
│  • Color-Coded Violation Feed & Screenshot Lightbox      │
└──────────────────────────────────────────────────────────┘
```

---

## 📊 Processing & Pipeline Flow

```
Input Frame
    │
    ├─→ YOLO Detection & ByteTrack
    │   ├─ Person bounding boxes (class 0)
    │   └─ Phone bounding boxes (class 67) ──→ Proximity Matcher
    │
    ├─→ MediaPipe Face Mesh Analyzer
    │   ├─ Full-Frame Pass
    │   └─ Person Head Crop Fallback (for small/distant faces)
    │
    ├─→ 3D solvePnP Head Pose & Gaze Analyzer
    │   ├─ Pitch, Yaw, Roll via 3D Facial Model
    │   ├─ Camera Perspective Compensation:
    │   │  pitch_adjusted = pitch_raw - atan2(cy - cy_center, f)
    │   └─ Iris Gaze Offset (Horizontal / Vertical)
    │
    └─→ Attention State Evaluator (Per Person)
        ├─ Timers with Grace Period (PHONE, GAZE_AWAY, HEAD_DOWN)
        ├─ Threshold Verification vs Dynamic Config
        └─ Dispatch Violation & Heartbeat to Flask API
```

---

## 🔧 Component Architecture

### 1. Central Configuration (`config.py`)
- Defines `DistractionConfig` dataclass with preset defaults optimized for short clips.
- Manages thread-safe JSON serialization/deserialization to `dashboard_data/config.json`.

### 2. Core Detection Package (`detector/`)
- `detector/face_mesh.py`:
  - `MediaPipeFaceAnalyzer`: Performs full-frame face mesh and head-crop fallback upscaling.
  - Computes physical head pose using OpenCV `solvePnP` with vertical/horizontal camera optical center perspective angle compensation.
  - Measures iris displacement relative to eye contours.
- `detector/yolo_tracker.py`:
  - `YOLOTrackerEngine`: Wraps Ultralytics YOLO with ByteTrack.
  - Spatial bounding box expansion matching for phone proximity.
- `detector/detector_core.py`:
  - `DistractionDetectorCore`: Integrates YOLO + MediaPipe + Config.
  - Tracks per-person `PersonAttentionState` timers and grace period decays.

### 3. Detector Runner (`detector_web.py`)
- Runs main video frame loop.
- Periodically polls `/api/config` to dynamically update detection thresholds without script restarts.
- Sends violation screenshot frames and live stream JPEG frames to Flask backend.

### 4. Flask API Server (`app.py`)
- `MonitoringState`: In-memory thread-safe state tracking active subjects, reports, and screenshots.
- Endpoints:
  - `GET /api/summary`: Overview statistics and live view path.
  - `GET /api/config` & `POST /api/config`: Get or update system configuration live.
  - `GET /api/violations`: List violation logs sorted by time.
  - `POST /api/test/add-violation`: Log new violation and save screenshot image.
  - `POST /api/report/export`: Generate JSON summary report.
  - `POST /api/reset`: Clear session metrics.

### 5. Web Interface (`templates/dashboard.html`)
- Glassmorphic UI featuring live stats cards, person tracking feed, screenshot lightbox viewer, and dynamic sensitivity control sliders syncing directly to `/api/config`.

---

## 🎯 Distraction Evaluation Rules

1. **Phone Violation**: Triggered when a cell phone is within a person's expanded bounding box for $\ge \text{PHONE\_THRESHOLD\_SEC}$ (default `1.0s`).
2. **Gaze Away Violation**: Triggered when iris displacement $|\text{gaze\_deg}| \ge \text{GAZE\_AWAY\_THRESHOLD\_DEG}$ (default `20.0°`) for $\ge \text{GAZE\_AWAY\_THRESHOLD\_SEC}$ (default `1.0s`).
3. **Head Down / Away Violation**: Triggered when perspective-compensated head pitch $\ge \text{HEAD\_DOWN\_THRESHOLD\_DEG}$ (default `22.0°`) or yaw $|\text{yaw}| \ge \text{HEAD\_YAW\_THRESHOLD\_DEG}$ (default `25.0°`) for $\ge \text{HEAD\_DOWN\_THRESHOLD\_SEC}$ (default `1.2s`).
4. **Grace Period**: Temporary detection dropouts are forgiven for up to $\text{GRACE\_PERIOD\_SEC}$ (default `1.0s`) before resetting active distraction timers.
