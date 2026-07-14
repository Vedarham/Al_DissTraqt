"""
DissTraqt Web-Integrated Detector
Processes video and sends violation data to the Flask web dashboard
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import requests
import threading
from pathlib import Path

# ============= SETTINGS =============
VIDEO_SOURCE = './assets/classroom.mp4'
MODEL_WEIGHTS = "yolov8n.pt"

# Thresholds (in seconds)
PHONE_THRESHOLD_SEC = 10
GAZE_AWAY_THRESHOLD_SEC = 5
HEAD_DOWN_THRESHOLD_SEC = 8

# Period-based counting (10 minutes)
PERIOD_DURATION_SEC = 600
ALERT_THRESHOLD_COUNT = 15

# Detection settings
CONF_THRESHOLD = 0.35
PERSON_CLASS = 0
PHONE_CLASS = 67

# Web API settings
API_BASE = 'http://localhost:5000/api'
WEB_API_ENABLED = True

# Gaze estimation sensitivity (degrees)
GAZE_AWAY_THRESHOLD_DEG = 30
HEAD_DOWN_THRESHOLD_DEG = 25

print("="*60)
print("DissTraqt Web-Integrated Detector")
print("="*60)

# ============= DATA CLASSES =============

@dataclass
class PersonState:
    """Tracks attention state for a single person"""
    track_id: int
    first_seen: float
    last_seen: float
    
    phone_detected: bool = False
    phone_interaction_start: Optional[float] = None
    phone_duration: float = 0.0
    
    gaze_horizontal: float = 0.5
    gaze_vertical: float = 0.5
    gaze_away_start: Optional[float] = None
    gaze_away_duration: float = 0.0
    
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    head_down_start: Optional[float] = None
    head_down_duration: float = 0.0
    
    # Track if we've reported this violation already
    phone_violation_reported: bool = False
    gaze_violation_reported: bool = False
    head_violation_reported: bool = False


# ============= HELPER FUNCTIONS =============

def send_violation_to_api(person_id, event_type, duration, screenshot_path=None):
    """Send violation event to web API"""
    if not WEB_API_ENABLED:
        return
    
    try:
        data = {
            'person_id': person_id,
            'event_type': event_type,
            'duration': duration,
            'screenshot_path': screenshot_path
        }
        response = requests.post(
            f'{API_BASE}/test/add-violation',
            json=data,
            timeout=2
        )
        if response.status_code == 200:
            print(f"✓ SENT to API: Person {person_id} | {event_type.upper()} | Duration: {duration:.1f}s")
        else:
            print(f"✗ API error ({response.status_code}): {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"✗ CANNOT CONNECT to {API_BASE}")
        print(f"   Make sure Flask is running: python run_dashboard.py")
    except Exception as e:
        print(f"✗ Error sending violation: {e}")


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def phone_near_person(person_box, phone_box, expand_ratio=0.15):
    """Check if phone is near person's head/hand area"""
    px1, py1, px2, py2 = person_box
    w, h = px2 - px1, py2 - py1
    ex, ey = w * expand_ratio, h * expand_ratio
    ex1, ey1, ex2, ey2 = px1 - ex, py1 - ey, px2 + ex, py2 + ey
    cx, cy = box_center(phone_box)
    return ex1 <= cx <= ex2 and ey1 <= cy <= ey2


def estimate_gaze_direction(face_landmarks, frame_shape) -> Tuple[float, float]:
    """Estimate gaze direction from iris position"""
    LEFT_EYE = [33, 133, 159, 158, 157, 173, 133, 155, 154]
    RIGHT_EYE = [263, 362, 386, 385, 384, 398, 362, 382, 381]
    
    try:
        left_iris = face_landmarks.landmark[468]
        right_iris = face_landmarks.landmark[473]
        
        left_eye_x = [face_landmarks.landmark[i].x for i in LEFT_EYE]
        left_eye_y = [face_landmarks.landmark[i].y for i in LEFT_EYE]
        right_eye_x = [face_landmarks.landmark[i].x for i in RIGHT_EYE]
        right_eye_y = [face_landmarks.landmark[i].y for i in RIGHT_EYE]
        
        left_gaze_x = np.clip(
            (left_iris.x - min(left_eye_x)) / (max(left_eye_x) - min(left_eye_x)),
            0, 1
        )
        right_gaze_x = np.clip(
            (right_iris.x - min(right_eye_x)) / (max(right_eye_x) - min(right_eye_x)),
            0, 1
        )
        
        left_gaze_y = np.clip(
            (left_iris.y - min(left_eye_y)) / (max(left_eye_y) - min(left_eye_y)),
            0, 1
        )
        right_gaze_y = np.clip(
            (right_iris.y - min(right_eye_y)) / (max(right_eye_y) - min(right_eye_y)),
            0, 1
        )
        
        gaze_h = (left_gaze_x + right_gaze_x) / 2
        gaze_v = (left_gaze_y + right_gaze_y) / 2
        
        return gaze_h, gaze_v
    except:
        return 0.5, 0.5


def estimate_head_pose(face_landmarks) -> Tuple[float, float, float]:
    """Estimate head pose (pitch, yaw, roll)"""
    try:
        nose = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y, face_landmarks.landmark[1].z])
        chin = np.array([face_landmarks.landmark[152].x, face_landmarks.landmark[152].y, face_landmarks.landmark[152].z])
        left_eye = np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y, face_landmarks.landmark[33].z])
        right_eye = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y, face_landmarks.landmark[263].z])
        left_mouth = np.array([face_landmarks.landmark[61].x, face_landmarks.landmark[61].y, face_landmarks.landmark[61].z])
        right_mouth = np.array([face_landmarks.landmark[291].x, face_landmarks.landmark[291].y, face_landmarks.landmark[291].z])
        
        vertical_vec = chin - nose
        pitch = np.degrees(np.arctan2(vertical_vec[1], -vertical_vec[2]))
        
        eye_center = (left_eye + right_eye) / 2
        nose_to_eye = eye_center - nose
        yaw = np.degrees(np.arctan2(nose_to_eye[0], -nose_to_eye[2]))
        
        mouth_center = (left_mouth + right_mouth) / 2
        mouth_to_nose = nose - mouth_center
        roll = np.degrees(np.arctan2(mouth_to_nose[0], mouth_to_nose[1]))
        
        return pitch, yaw, roll
    except:
        return 0.0, 0.0, 0.0


def is_gaze_away(gaze_h: float, gaze_v: float) -> bool:
    """Check if gaze is looking away"""
    gaze_angle_h = (gaze_h - 0.5) * 90
    gaze_angle_v = (gaze_v - 0.5) * 60
    return abs(gaze_angle_h) > GAZE_AWAY_THRESHOLD_DEG or abs(gaze_angle_v) > GAZE_AWAY_THRESHOLD_DEG


def is_head_down(pitch: float) -> bool:
    """Check if head is looking down"""
    return pitch > HEAD_DOWN_THRESHOLD_DEG


# ============= MAIN LOOP =============

print(f"Loading YOLO model...")
try:
    yolo_model = YOLO(MODEL_WEIGHTS)
    print("✓ YOLO model loaded")
except Exception as e:
    print(f"✗ Error loading YOLO: {e}")
    exit(1)

print(f"\nChecking MediaPipe...")
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ MediaPipe loaded")
except Exception as e:
    print(f"✗ MediaPipe error: {e}")
    print(f"Fix: python -m pip install mediapipe")
    exit(1)

print(f"\n{'='*60}")
print(f"Opening video: {VIDEO_SOURCE}")
print(f"Web API: {API_BASE}")
print(f"{'='*60}\n")

person_states: Dict[int, PersonState] = {}
grace_period_sec = 1.5
frame_count = 0
start_time = time.time()

try:
    print(f"Initializing YOLO tracker with video: {VIDEO_SOURCE}")
    tracker = yolo_model.track(
        source=VIDEO_SOURCE,
        classes=[PERSON_CLASS, PHONE_CLASS],
        conf=CONF_THRESHOLD,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
    )
    print(f"✓ Tracker initialized, starting frame processing...")
    
    for result in tracker:
        frame_count += 1
        frame = result.orig_img
        
        # Safety check
        if frame is None:
            print(f"✗ Frame {frame_count} is None, breaking")
            break
        now = time.time()
        
        # ===== DETECT PEOPLE AND PHONES =====
        people, phones = [], []
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            clss = result.boxes.cls.int().tolist()
            confs = result.boxes.conf.tolist()
            xyxys = result.boxes.xyxy.tolist()
            for tid, cls, conf, box in zip(ids, clss, confs, xyxys):
                if cls == PERSON_CLASS:
                    people.append((tid, box, conf))
                elif cls == PHONE_CLASS:
                    phones.append((box, conf))
        
        # Attribute phones to nearest person
        people_with_phone = set()
        for pbox, _ in phones:
            best_tid, best_dist = None, None
            for tid, box, _ in people:
                if not phone_near_person(box, pbox):
                    continue
                pcx, pcy = box_center(pbox)
                bcx, bcy = box_center(box)
                d = (pcx - bcx) ** 2 + (pcy - bcy) ** 2
                if best_dist is None or d < best_dist:
                    best_dist, best_tid = d, tid
            if best_tid is not None:
                people_with_phone.add(best_tid)
        
        # ===== DETECT FACES =====
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(frame_rgb)
        
        person_faces: Dict[int, any] = {}
        faces_detected = 0
        if face_results.multi_face_landmarks:
            faces_detected = len(face_results.multi_face_landmarks)
            for face_landmarks in face_results.multi_face_landmarks:
                face_center_x = np.mean([lm.x for lm in face_landmarks.landmark])
                face_center_y = np.mean([lm.y for lm in face_landmarks.landmark])
                
                best_tid, best_dist = None, float('inf')
                for tid, box, _ in people:
                    px1, py1, px2, py2 = box
                    pcx = (px1 + px2) / 2 / frame.shape[1]
                    pcy = (py1 + py2) / 2 / frame.shape[0]
                    d = (face_center_x - pcx) ** 2 + (face_center_y - pcy) ** 2
                    if d < best_dist:
                        best_dist, best_tid = d, tid
                
                if best_tid is not None and best_dist < 0.15:
                    person_faces[best_tid] = face_landmarks
        
        # ===== UPDATE PERSON STATES =====
        for tid, box, conf in people:
            if tid not in person_states:
                person_states[tid] = PersonState(
                    track_id=tid,
                    first_seen=now,
                    last_seen=now
                )
            
            state = person_states[tid]
            state.last_seen = now
            
            # --- Phone interaction ---
            phone_present = tid in people_with_phone
            if phone_present:
                if state.phone_interaction_start is None:
                    state.phone_interaction_start = now
                    state.phone_violation_reported = False  # Reset flag for new interaction
                state.phone_detected = True
                state.phone_duration = now - state.phone_interaction_start
                
                # SEND VIOLATION IMMEDIATELY WHEN THRESHOLD IS EXCEEDED
                if state.phone_duration >= PHONE_THRESHOLD_SEC and not state.phone_violation_reported:
                    send_violation_to_api(tid, 'phone', state.phone_duration)
                    state.phone_violation_reported = True
            else:
                # Phone is no longer present
                if state.phone_interaction_start is not None:
                    state.phone_interaction_start = None
                    state.phone_duration = 0.0
            
            # --- Gaze and head pose ---
            if tid in person_faces:
                face_landmarks = person_faces[tid]
                
                # Gaze
                gaze_h, gaze_v = estimate_gaze_direction(face_landmarks, frame.shape)
                state.gaze_horizontal = gaze_h
                state.gaze_vertical = gaze_v
                
                looking_away = is_gaze_away(gaze_h, gaze_v)
                if looking_away:
                    if state.gaze_away_start is None:
                        state.gaze_away_start = now
                        state.gaze_violation_reported = False  # Reset flag for new gaze event
                    state.gaze_away_duration = now - state.gaze_away_start
                    
                    # SEND VIOLATION IMMEDIATELY WHEN THRESHOLD IS EXCEEDED
                    if state.gaze_away_duration >= GAZE_AWAY_THRESHOLD_SEC and not state.gaze_violation_reported:
                        send_violation_to_api(tid, 'gaze_away', state.gaze_away_duration)
                        state.gaze_violation_reported = True
                else:
                    # Looking back at screen
                    state.gaze_away_start = None
                    state.gaze_away_duration = 0.0
                
                # Head pose
                pitch, yaw, roll = estimate_head_pose(face_landmarks)
                state.head_pitch = pitch
                state.head_yaw = yaw
                state.head_roll = roll
                
                head_is_down = is_head_down(pitch)
                if head_is_down:
                    if state.head_down_start is None:
                        state.head_down_start = now
                        state.head_violation_reported = False  # Reset flag for new head event
                    state.head_down_duration = now - state.head_down_start
                    
                    # SEND VIOLATION IMMEDIATELY WHEN THRESHOLD IS EXCEEDED
                    if state.head_down_duration >= HEAD_DOWN_THRESHOLD_SEC and not state.head_violation_reported:
                        send_violation_to_api(tid, 'head_down', state.head_down_duration)
                        state.head_violation_reported = True
                else:
                    # Head is back up
                    state.head_down_start = None
                    state.head_down_duration = 0.0
        
        # ===== DRAW ANNOTATIONS & STATUS =====
        debug_info = []
        
        for tid, box, conf in people:
            if tid not in person_states:
                continue
            
            state = person_states[tid]
            x1, y1, x2, y2 = map(int, box)
            
            color = (0, 255, 0)
            label = f"ID {tid}"
            
            if state.phone_duration >= PHONE_THRESHOLD_SEC:
                color = (0, 0, 255)
                label += f" [PHONE {state.phone_duration:.0f}s]"
                debug_info.append(f"ID{tid}: PHONE {state.phone_duration:.1f}s")
            elif state.gaze_away_duration >= GAZE_AWAY_THRESHOLD_SEC:
                color = (0, 165, 255)
                label += f" [GAZE {state.gaze_away_duration:.0f}s]"
                debug_info.append(f"ID{tid}: GAZE {state.gaze_away_duration:.1f}s")
            elif state.head_down_duration >= HEAD_DOWN_THRESHOLD_SEC:
                color = (255, 0, 255)
                label += f" [HEAD {state.head_down_duration:.0f}s]"
                debug_info.append(f"ID{tid}: HEAD {state.head_down_duration:.1f}s")
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        for box, conf in phones:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)
            cv2.putText(frame, f"phone {conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)
        
        # Status
        elapsed = time.time() - start_time
        status = f"Frame {frame_count} | People: {len(people)} | Phones: {len(phones)} | Faces: {faces_detected} | API: {'ON' if WEB_API_ENABLED else 'OFF'}"
        cv2.putText(frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Print violations to console
        if debug_info and frame_count % 30 == 0:  # Print every 30 frames to avoid spam
            print(f"Frame {frame_count}: {' | '.join(debug_info)}")
        elif frame_count % 100 == 0:  # Print heartbeat every 100 frames
            print(f"🔄 Processing frame {frame_count}... {len(people)} people, {len(phones)} phones, {len(person_faces)} faces detected")
        
        cv2.putText(frame, "Press 'q' to quit", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Display
        cv2.imshow("DissTraqt Web Detector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"\n✗ Error during processing: {e}")
    import traceback
    traceback.print_exc()

finally:
    cv2.destroyAllWindows()
    print(f"\n{'='*60}")
    print(f"✓ Processing complete")
    print(f"✓ Processed {frame_count} frames")
    print(f"✓ People tracked: {len(person_states)}")
    print(f"{'='*60}\n")
    
    for tid, state in person_states.items():
        print(f"Person {tid}:")
        print(f"  Phone events: {1 if state.phone_violation_reported else 0}")
        print(f"  Gaze events: {1 if state.gaze_violation_reported else 0}")
        print(f"  Head events: {1 if state.head_violation_reported else 0}")
