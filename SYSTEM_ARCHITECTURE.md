# DissTraqt System Architecture

## 🏗️ System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     DissTraqt Monitoring System                 │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│   Video Input    │
│                  │
│ • Video file     │
│ • Webcam (0)     │
│ • Stream         │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│          detector_web.py (Video Processor)               │
│                                                          │
│  1. Load YOLO Model                                     │
│  2. Load MediaPipe Face Detection                       │
│  3. For each frame:                                     │
│     • Detect persons & phones (YOLO)                    │
│     • Detect face landmarks (MediaPipe)                 │
│     • Estimate gaze direction                           │
│     • Estimate head pose (pitch/yaw/roll)               │
│     • Track violations                                  │
│     • Send data to Flask API                            │
│  4. Display live video with annotations                 │
└────────┬──────────────────────────────────────────────────┘
         │ HTTP POST /api/test/add-violation
         │ (Violation events)
         ▼
┌──────────────────────────────────────────────────────────┐
│            app.py (Flask Web Server)                     │
│                                                          │
│  • Receives violation events from detector               │
│  • Stores in MonitoringState (in-memory)               │
│  • Calculates statistics                               │
│  • Generates reports                                    │
│  • Serves API endpoints                                │
│  • Saves screenshots & reports to disk                 │
└────────┬──────────────────────────────────────────────────┘
         │ HTTP GET /api/summary
         │ HTTP GET /api/statistics
         │ HTTP GET /api/violations
         ▼
┌──────────────────────────────────────────────────────────┐
│        dashboard.html (Web Interface)                    │
│                                                          │
│  • Real-time statistics cards                          │
│  • Person tracking list                                │
│  • Violation feed (live updates)                        │
│  • Filter tabs (phone/gaze/head)                        │
│  • Risk scoring visualization                          │
│  • Export button                                       │
│  • Auto-refresh every 3 seconds                        │
└──────────────────────────────────────────────────────────┘
         │
         ▼ (User opens browser)
    http://localhost:5000
```

---

## 📊 Data Flow

```
Video Frame
    │
    ▼
┌─────────────────────┐
│  YOLO Detection     │
│ (person + phone)    │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────┐
│  MediaPipe Face Mesh     │
│ (468 face landmarks)     │
└──────────┬───────────────┘
           │
           ├─→ Gaze Estimation (iris position)
           │   └─→ Check if gaze_away
           │
           ├─→ Head Pose (nose-chin, eye-nose vectors)
           │   ├─→ Pitch (up/down)
           │   ├─→ Yaw (left/right)
           │   └─→ Roll (tilt)
           │
           └─→ Phone Detection (proximity)
               └─→ Check if phone_near_person
    
    Violation Events (if threshold exceeded)
    │
    ├─→ Violation Type:
    │   ├─→ 'phone' (duration > 10s)
    │   ├─→ 'gaze_away' (duration > 5s)
    │   └─→ 'head_down' (duration > 8s)
    │
    └─→ Send to API
        └─→ Store in PersonReport
            └─→ Update Dashboard in real-time
```

---

## 🔧 Component Details

### 1. Detector (detector_web.py)
```
Inputs:
├─ Video file/webcam
├─ YOLO model (yolov8n.pt)
└─ MediaPipe Face Mesh

Processing:
├─ YOLO: Detect person & phone bounding boxes
├─ MediaPipe: Detect 468 face landmarks
├─ Gaze: Calculate iris position relative to eye
├─ Head Pose: Compute euler angles from landmarks
└─ Tracking: Match detections across frames

Outputs:
├─ Violation events (sent to API)
├─ Live video display with annotations
└─ Person state tracking
```

### 2. Web Server (app.py)
```
Endpoints:
├─ GET  /api/summary          → Current state
├─ GET  /api/statistics       → Aggregated stats
├─ GET  /api/violations       → All violation events
├─ GET  /api/violations/type/ → Filter by type
├─ GET  /api/person/<id>      → Individual report
├─ POST /api/test/add-violation → Test endpoint
├─ POST /api/report/export    → Export JSON
└─ POST /api/reset            → Clear data

Storage:
├─ MonitoringState (RAM) → Active session data
├─ Screenshots (disk) → dashboard_data/screenshots/
└─ Reports (disk) → dashboard_data/reports/
```

### 3. Dashboard (dashboard.html)
```
Components:
├─ Statistics Panel (6 cards)
├─ People List (with risk scores)
├─ Violation Feed (color-coded)
├─ Filter Tabs (phone/gaze/head)
└─ Control Buttons (refresh/export/reset)

Auto-refresh:
└─ Every 3 seconds via JavaScript
   └─ Fetches /api/summary
   └─ Fetches /api/violations
   └─ Updates DOM with new data
```

---

## 🎯 Violation Detection Logic

### Phone Detection
```
1. YOLO detects phone object
2. Check if phone is near person (within expanded bbox)
3. If detected continuously:
   → Increment phone_duration
   → If phone_duration >= PHONE_THRESHOLD_SEC (10s)
     → Report violation
     → Mark phone_violation_reported = True
4. When phone disappears:
   → Reset phone_duration
   → Reset phone_violation_reported for next event
```

### Gaze Detection
```
1. MediaPipe detects face landmarks
2. Get iris position (landmarks 468, 473)
3. Get eye bounds (landmarks 33, 133, 159, 158, etc.)
4. Calculate iris position relative to eye bounds (0-1)
5. Convert to degrees:
   → gaze_angle_h = (gaze_h - 0.5) * 90  [degrees]
   → gaze_angle_v = (gaze_v - 0.5) * 60  [degrees]
6. If abs(gaze_angle) > GAZE_AWAY_THRESHOLD_DEG (30°):
   → Increment gaze_away_duration
   → If gaze_away_duration >= GAZE_AWAY_THRESHOLD_SEC (5s)
     → Report violation
7. When looking back:
   → Reset gaze_away_duration
```

### Head Pose Detection
```
1. Get key face landmarks:
   - Nose tip (landmark 1)
   - Chin (landmark 152)
   - Left eye (landmark 33)
   - Right eye (landmark 263)
   - Mouth corners (landmarks 61, 291)

2. Calculate vectors:
   - vertical_vec = chin - nose
     → pitch = arctan2(vertical_vec.y, -vertical_vec.z)
   
   - nose_to_eye = eye_center - nose
     → yaw = arctan2(nose_to_eye.x, -nose_to_eye.z)
   
   - mouth_to_nose = nose - mouth_center
     → roll = arctan2(mouth_to_nose.x, mouth_to_nose.y)

3. If pitch > HEAD_DOWN_THRESHOLD_DEG (25°):
   → Increment head_down_duration
   → If head_down_duration >= HEAD_DOWN_THRESHOLD_SEC (8s)
     → Report violation
4. When head up:
   → Reset head_down_duration
```

---

## 📈 Risk Scoring Algorithm

```
Distraction Score (0-100):

score = (phone_events * 10) + (gaze_events * 5) + (head_events * 5)
score = min(100, score)  # Cap at 100

Risk Levels:
├─ 🟢 0-30: Low Risk
├─ 🟠 30-60: Medium Risk
└─ 🔴 60-100: High Risk

Example:
Person has:
  • 3 phone violations = 3 * 10 = 30
  • 4 gaze violations = 4 * 5 = 20
  • 2 head violations = 2 * 5 = 10
  Total = 60 (Medium Risk - Orange)
```

---

## 💾 Data Models

### PersonState (In Detector)
```python
@dataclass
class PersonState:
    track_id: int
    first_seen: float
    last_seen: float
    
    # Phone tracking
    phone_duration: float
    phone_interaction_start: Optional[float]
    phone_violation_reported: bool
    
    # Gaze tracking
    gaze_away_duration: float
    gaze_away_start: Optional[float]
    gaze_violation_reported: bool
    
    # Head tracking
    head_down_duration: float
    head_down_start: Optional[float]
    head_violation_reported: bool
```

### PersonReport (In API)
```python
@dataclass
class PersonReport:
    person_id: int
    total_phone_events: int
    total_phone_duration: float
    total_gaze_events: int
    total_gaze_duration: float
    total_head_events: int
    total_head_duration: float
    violations: List[ViolationEvent]
    
    @property
    def distraction_score(self) -> int:
        return min(100, 
            phone_events * 10 + 
            gaze_events * 5 + 
            head_events * 5)
```

### ViolationEvent (API)
```python
@dataclass
class ViolationEvent:
    person_id: int
    event_type: str  # 'phone' | 'gaze_away' | 'head_down'
    timestamp: float
    duration: float
    screenshot_path: Optional[str]
```

---

## 🔄 Real-Time Update Cycle

```
Timer: Every 3 seconds

Browser/Dashboard
    │
    ├─→ fetch('/api/summary')
    │   └─→ Get current people & violations
    │
    ├─→ fetch('/api/violations')
    │   └─→ Get all violation events
    │
    └─→ fetch('/api/statistics')
        └─→ Get aggregated stats

Update DOM
    ├─→ Update stat cards
    ├─→ Refresh person list
    ├─→ Update violation feed
    └─→ Recalculate colors/scores
```

---

## 📊 JSON Export Structure

```json
{
  "generated_at": "2024-07-14T10:30:00",
  "session_duration_minutes": 45.5,
  "summary": {
    "total_people": 5,
    "total_violations": 23,
    "people": [
      {
        "person_id": 1,
        "total_phone_events": 3,
        "total_phone_duration": 35.2,
        "total_gaze_events": 2,
        "total_gaze_duration": 12.1,
        "total_head_events": 1,
        "total_head_duration": 8.5,
        "distraction_score": 55,
        "violations": [...]
      },
      ...
    ]
  },
  "statistics": {
    "phone_violations": 12,
    "gaze_violations": 8,
    "head_violations": 3,
    "avg_distraction_score": 42.3,
    "most_distracted_person": {
      "id": 1,
      "score": 65
    }
  }
}
```

---

## 🎯 System Flow Example

```
1. User runs: python run_dashboard.py
   └─→ Flask server starts on :5000
   └─→ Browser opens dashboard

2. User runs: python detector_web.py
   └─→ Loads YOLO + MediaPipe
   └─→ Opens video file
   └─→ Starts processing frames

3. Frame 1 arrives:
   └─→ YOLO detects Person(ID=1) + Phone
   └─→ MediaPipe detects face landmarks
   └─→ Gaze is center (not away)
   └─→ Head is neutral (not down)
   └─→ No violations yet

4. Frame 50 (2 seconds later):
   └─→ YOLO still detects Person(ID=1) + Phone
   └─→ Phone_duration = 2.0s (< 10s threshold)
   └─→ No violation yet

5. Frame 350 (12 seconds later):
   └─→ YOLO still detects Person(ID=1) + Phone
   └─→ Phone_duration = 12.0s (>= 10s threshold)
   └─→ SEND VIOLATION: person_id=1, event_type='phone', duration=12.0
   └─→ API stores violation in PersonReport
   └─→ Dashboard refreshes (3 sec) and shows:
       - Person 1: 1 phone event
       - Risk score: 10/100 (low)

6. Frame 400:
   └─→ Phone disappears
   └─→ Reset phone_duration
   └─→ Reset phone_violation_reported

7. User clicks "Export Report":
   └─→ API saves JSON to dashboard_data/reports/
   └─→ JSON contains all violations + stats
   └─→ Ready for analysis
```

---

## 🚀 Startup Sequence

```
1. python run_dashboard.py
   ├─→ Check Python packages
   ├─→ Start Flask app.py
   ├─→ Load routes from app.py
   ├─→ Initialize MonitoringState()
   ├─→ Serve static files
   └─→ Wait for requests

2. python detector_web.py
   ├─→ Load YOLO model (30s)
   ├─→ Load MediaPipe Face Mesh
   ├─→ Open video file
   ├─→ Start main loop
   └─→ Connect to Flask API

3. Browser: http://localhost:5000
   ├─→ Download dashboard.html
   ├─→ Load dashboard.js script
   ├─→ JavaScript: setInterval(refreshData, 3000)
   └─→ Auto-fetch from API every 3 sec
```

