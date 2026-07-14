"""
Phone + Head Pose + Periodic Counting
  1. Phone/head-away timers now use dedicated "last seen WITH this signal"
     timestamps instead of the generic `last_seen` field, so the grace
     period actually works and timers reset when the signal genuinely stops.
  2. The periodic window now accumulates SECONDS distracted per person,
     not frame counts — the old version could cross its alert threshold
     in well under a second at normal frame rates.
  3. Head pose uses solvePnP + a pinhole camera model (real degrees,
     robust to frame aspect ratio and face distance) instead of raw
     normalized-coordinate arctan2, which wasn't physically meaningful.
  4. Gaze/head-away state now decays via the same grace-period pattern as
     phone, instead of freezing forever if face-matching misses a frame.
"""

import time
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# ============= SETTINGS (EDIT THESE) =============
VIDEO_SOURCE = './assets/classroom.mp4'
MODEL_WEIGHTS = "yolo26n.pt"

CONF_THRESHOLD = 0.35
PERSON_CLASS = 0
PHONE_CLASS = 67

# --- Duration thresholds (seconds) ---
PHONE_THRESHOLD_SEC = 5
GAZE_AWAY_THRESHOLD_SEC = 5
HEAD_DOWN_THRESHOLD_SEC = 8

# --- Grace periods (seconds) — forgive brief detection flicker before resetting a timer ---
PHONE_GRACE_SEC = 1.5
GAZE_GRACE_SEC = 1.0
HEAD_GRACE_SEC = 1.0

# --- Angle thresholds (degrees) ---
GAZE_AWAY_THRESHOLD_DEG = 30      # iris offset from eye-center, rescaled to a pseudo-degree
HEAD_YAW_THRESHOLD_DEG = 25.0     # head turned sideways beyond this
HEAD_PITCH_DOWN_THRESHOLD_DEG = 25.0  # head tilted down beyond this

# --- Periodic window ---
PERIOD_DURATION_SEC = 600     # 10 minutes (lower for testing, e.g. 30)
ALERT_FRACTION = 0.5          # a person is "flagged" if distracted >= this fraction of the window
ALERT_PERSON_COUNT = 3        # alert fires if >= this many people are flagged
# ==================================================

print("Loading YOLO model...")
yolo_model = YOLO(MODEL_WEIGHTS)

print("Loading MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=15,
    refine_landmarks=True,   # needed for iris landmarks (indices 468-477) used by gaze estimation
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Generic 3D face model for solvePnP (arbitrary units, standard reference points)
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye, left corner
    (225.0, 170.0, -135.0),   # Right eye, right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

POSE_LANDMARK_IDS = {
    "nose_tip": 1, "chin": 152,
    "left_eye_left_corner": 33, "right_eye_right_corner": 263,
    "left_mouth_corner": 61, "right_mouth_corner": 291,
}
LEFT_EYE_RING = [33, 133, 159, 158, 157, 173, 155, 154]
RIGHT_EYE_RING = [263, 362, 386, 385, 384, 398, 382, 381]
LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER = 468, 473  # only valid with refine_landmarks=True


# ============= DATA CLASSES =============
@dataclass
class PersonState:
    track_id: int

    # Phone
    phone_start: Optional[float] = None
    last_seen_with_phone: Optional[float] = None
    phone_duration: float = 0.0

    # Gaze
    gaze_away_start: Optional[float] = None
    last_seen_gaze_away: Optional[float] = None
    gaze_away_duration: float = 0.0

    # Head pose
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_away_start: Optional[float] = None
    last_seen_head_away: Optional[float] = None
    head_away_duration: float = 0.0

    # For windowed aggregation: last timestamp we integrated distracted-time for
    last_window_update: Optional[float] = None


class DistractionAggregator:
    """Accumulates seconds-distracted per person within a rolling window,
    flags anyone over ALERT_FRACTION of the window, and fires an alert if
    enough people are flagged. This replaces the old per-frame counter."""

    def __init__(self, window_sec, alert_fraction, alert_person_count):
        self.window_sec = window_sec
        self.alert_fraction = alert_fraction
        self.alert_person_count = alert_person_count
        self.window_start = time.time()
        self.distracted_seconds: Dict[int, float] = defaultdict(float)

    def update(self, tid: int, is_distracted: bool, now: float, state: PersonState):
        last = state.last_window_update if state.last_window_update is not None else now
        dt = now - last
        if is_distracted:
            self.distracted_seconds[tid] += dt
        state.last_window_update = now

    def maybe_flush(self, now: float):
        elapsed = now - self.window_start
        if elapsed < self.window_sec:
            return None, False
        flagged = [tid for tid, secs in self.distracted_seconds.items()
                   if secs >= self.alert_fraction * self.window_sec]
        report = {
            "window_start": self.window_start, "window_end": now,
            "people_tracked": len(self.distracted_seconds),
            "flagged_person_ids": flagged, "flagged_count": len(flagged),
        }
        alert_fired = len(flagged) >= self.alert_person_count
        self.window_start = now
        self.distracted_seconds.clear()
        return report, alert_fired


# ============= GEOMETRY HELPERS =============
def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_in_expanded_box(point, box, expand_ratio=0.15):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ex, ey = w * expand_ratio, h * expand_ratio
    ex1, ey1, ex2, ey2 = x1 - ex, y1 - ey, x2 + ex, y2 + ey
    px, py = point
    return ex1 <= px <= ex2 and ey1 <= py <= ey2


def match_point_to_nearest_person(point, people):
    best_tid, best_dist = None, None
    for tid, box, _ in people:
        if not point_in_expanded_box(point, box):
            continue
        bcx, bcy = box_center(box)
        d = (point[0] - bcx) ** 2 + (point[1] - bcy) ** 2
        if best_dist is None or d < best_dist:
            best_dist, best_tid = d, tid
    return best_tid


# ============= HEAD POSE (solvePnP — real degrees) =============
def estimate_head_pose(face_landmarks, frame_w, frame_h):
    """Proper geometric head pose: maps 6 known 3D face points to their
    detected 2D pixel positions via solvePnP. Robust to frame aspect ratio
    and face distance, unlike raw-coordinate arctan2 heuristics."""
    pts_px = {}
    for name, idx in POSE_LANDMARK_IDS.items():
        lm = face_landmarks.landmark[idx]
        pts_px[name] = (lm.x * frame_w, lm.y * frame_h)

    image_points = np.array([
        pts_px["nose_tip"], pts_px["chin"],
        pts_px["left_eye_left_corner"], pts_px["right_eye_right_corner"],
        pts_px["left_mouth_corner"], pts_px["right_mouth_corner"],
    ], dtype=np.float64)

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rotation_vec, _ = cv2.solvePnP(
        FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    sy = math.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
    if sy >= 1e-6:
        pitch = math.atan2(rotation_mat[2, 1], rotation_mat[2, 2])
        yaw = math.atan2(-rotation_mat[2, 0], sy)
    else:
        pitch = math.atan2(-rotation_mat[1, 2], rotation_mat[1, 1])
        yaw = math.atan2(-rotation_mat[2, 0], sy)

    return math.degrees(pitch), math.degrees(yaw)


def estimate_gaze_offset_deg(face_landmarks):
    """Secondary, softer signal: iris position relative to eye-ring bounds,
    rescaled to a pseudo-degree range. Noisier than head pose — use as a
    supplement, not the primary 'looking away' signal."""
    try:
        left_iris = face_landmarks.landmark[LEFT_IRIS_CENTER]
        right_iris = face_landmarks.landmark[RIGHT_IRIS_CENTER]
        lx = [face_landmarks.landmark[i].x for i in LEFT_EYE_RING]
        rx = [face_landmarks.landmark[i].x for i in RIGHT_EYE_RING]

        left_ratio = np.clip((left_iris.x - min(lx)) / max(max(lx) - min(lx), 1e-6), 0, 1)
        right_ratio = np.clip((right_iris.x - min(rx)) / max(max(rx) - min(rx), 1e-6), 0, 1)
        gaze_h = (left_ratio + right_ratio) / 2
        return (gaze_h - 0.5) * 90  # pseudo-degrees, horizontal only
    except (IndexError, ZeroDivisionError):
        return 0.0


def get_faces_px(frame_rgb, frame_w, frame_h):
    results = face_mesh.process(frame_rgb)
    faces = []
    if not results.multi_face_landmarks:
        return faces
    for fl in results.multi_face_landmarks:
        xs = [lm.x for lm in fl.landmark]
        ys = [lm.y for lm in fl.landmark]
        centroid = (sum(xs) / len(xs) * frame_w, sum(ys) / len(ys) * frame_h)
        faces.append((centroid, fl))
    return faces


# ============= MAIN LOOP =============
person_states: Dict[int, PersonState] = {}
aggregator = DistractionAggregator(PERIOD_DURATION_SEC, ALERT_FRACTION, ALERT_PERSON_COUNT)

print(f"Opening video source: {VIDEO_SOURCE!r}...")
print("Press 'q' to quit.\n")

frame_count = 0
for result in yolo_model.track(
    source=VIDEO_SOURCE,
    classes=[PERSON_CLASS, PHONE_CLASS],
    conf=CONF_THRESHOLD,
    persist=True,
    tracker="bytetrack.yaml",
    stream=True,
    verbose=False,
):
    frame_count += 1
    frame = result.orig_img
    frame_h, frame_w = frame.shape[:2]
    now = time.time()

    # ---- YOLO detections ----
    people, phones = [], []
    if result.boxes is not None and result.boxes.id is not None:
        ids = result.boxes.id.int().tolist()
        clss = result.boxes.cls.int().tolist()
        confs = result.boxes.conf.tolist()
        xyxys = result.boxes.xyxy.tolist()
        for tid, cls, conf, box in zip(ids, clss, confs, xyxys):
            (people if cls == PERSON_CLASS else phones).append(
                (tid, box, conf) if cls == PERSON_CLASS else (box, conf)
            )

    people_with_phone = set()
    for pbox, _ in phones:
        tid = match_point_to_nearest_person(box_center(pbox), people)
        if tid is not None:
            people_with_phone.add(tid)

    # ---- Face landmarks + matching to person tracks ----
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = get_faces_px(frame_rgb, frame_w, frame_h)
    person_face_landmarks = {}
    for centroid, fl in faces:
        tid = match_point_to_nearest_person(centroid, people)
        if tid is not None:
            person_face_landmarks[tid] = fl

    # ---- Update per-person state ----
    for tid, box, conf in people:
        if tid not in person_states:
            person_states[tid] = PersonState(track_id=tid)
        state = person_states[tid]

        # --- Phone timer ---
        phone_present = tid in people_with_phone
        if phone_present:
            if state.phone_start is None:
                state.phone_start = now
            state.last_seen_with_phone = now
        elif state.last_seen_with_phone is not None and (now - state.last_seen_with_phone) > PHONE_GRACE_SEC:
            state.phone_start = None
            state.last_seen_with_phone = None
        state.phone_duration = (now - state.phone_start) if state.phone_start else 0.0
        phone_distracted = state.phone_duration >= PHONE_THRESHOLD_SEC

        # --- Head pose + gaze (only if we have a matched face this frame) ---
        gaze_away_now = False
        head_away_now = False
        if tid in person_face_landmarks:
            fl = person_face_landmarks[tid]
            pose = estimate_head_pose(fl, frame_w, frame_h)
            if pose is not None:
                state.head_pitch, state.head_yaw = pose
                head_away_now = (abs(state.head_yaw) > HEAD_YAW_THRESHOLD_DEG or
                                  state.head_pitch > HEAD_PITCH_DOWN_THRESHOLD_DEG)
            gaze_deg = estimate_gaze_offset_deg(fl)
            gaze_away_now = abs(gaze_deg) > GAZE_AWAY_THRESHOLD_DEG
        # if no face matched this frame, both *_now stay False — timers below
        # will decay via grace period rather than freezing, same as phone.

        # --- Gaze timer ---
        if gaze_away_now:
            if state.gaze_away_start is None:
                state.gaze_away_start = now
            state.last_seen_gaze_away = now
        elif state.last_seen_gaze_away is not None and (now - state.last_seen_gaze_away) > GAZE_GRACE_SEC:
            state.gaze_away_start = None
            state.last_seen_gaze_away = None
        state.gaze_away_duration = (now - state.gaze_away_start) if state.gaze_away_start else 0.0
        gaze_distracted = state.gaze_away_duration >= GAZE_AWAY_THRESHOLD_SEC

        # --- Head-away timer ---
        if head_away_now:
            if state.head_away_start is None:
                state.head_away_start = now
            state.last_seen_head_away = now
        elif state.last_seen_head_away is not None and (now - state.last_seen_head_away) > HEAD_GRACE_SEC:
            state.head_away_start = None
            state.last_seen_head_away = None
        state.head_away_duration = (now - state.head_away_start) if state.head_away_start else 0.0
        head_distracted = state.head_away_duration >= HEAD_DOWN_THRESHOLD_SEC

        # --- Combined + windowed aggregation ---
        is_distracted = phone_distracted or gaze_distracted or head_distracted
        aggregator.update(tid, is_distracted, now, state)

        # --- Draw ---
        color, flag = (0, 255, 0), ""
        if phone_distracted:
            color, flag = (0, 0, 255), " [PHONE]"
        elif gaze_distracted:
            color, flag = (0, 165, 255), " [GAZE_AWAY]"
        elif head_distracted:
            color, flag = (255, 0, 255), " [HEAD_AWAY]"

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tid}"
        if state.phone_duration > 0: label += f" | phone {state.phone_duration:.1f}s"
        if state.gaze_away_duration > 0: label += f" | gaze {state.gaze_away_duration:.1f}s"
        if state.head_away_duration > 0: label += f" | head {state.head_away_duration:.1f}s"
        cv2.putText(frame, label + flag, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for box, conf in phones:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)
        cv2.putText(frame, f"phone {conf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)

    # ---- Periodic window check ----
    report, alert_fired = aggregator.maybe_flush(now)
    if report is not None:
        print(f"\n[WINDOW REPORT] {report['people_tracked']} tracked, "
              f"{report['flagged_count']} flagged (IDs: {report['flagged_person_ids']})")
        if alert_fired:
            print(f"  >>> ALERT: {report['flagged_count']} people distracted "
                  f">= {ALERT_FRACTION*100:.0f}% of the last {PERIOD_DURATION_SEC/60:.1f} min window <<<")

    cv2.putText(frame, f"frame #{frame_count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.imshow("Enhanced Detector v3 (fixed)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
print(f"\nDone. Processed {frame_count} frames.")