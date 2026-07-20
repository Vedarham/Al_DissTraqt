"""
DissTraqt Combined Distraction Detector Engine
Integrates YOLO object tracking and MediaPipe 3D solvePnP face analysis.
Manages per-person distraction timers with grace periods and reports violations dynamically.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
import cv2
import numpy as np

import sys
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DistractionConfig

try:
    from .face_mesh import MediaPipeFaceAnalyzer
    from .yolo_tracker import YOLOTrackerEngine
except ImportError:
    from face_mesh import MediaPipeFaceAnalyzer
    from yolo_tracker import YOLOTrackerEngine


@dataclass
class PersonAttentionState:
    track_id: int
    first_seen: float
    last_seen: float

    # Phone state
    phone_detected: bool = False
    phone_start: Optional[float] = None
    last_seen_with_phone: Optional[float] = None
    phone_duration: float = 0.0
    phone_violation_reported: bool = False

    # Gaze state
    gaze_horizontal_deg: float = 0.0
    gaze_vertical_deg: float = 0.0
    gaze_away_start: Optional[float] = None
    last_seen_gaze_away: Optional[float] = None
    gaze_away_duration: float = 0.0
    gaze_violation_reported: bool = False

    # Head pose state
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    head_down_start: Optional[float] = None
    last_seen_head_away: Optional[float] = None
    head_down_duration: float = 0.0
    head_violation_reported: bool = False


class DistractionDetectorCore:
    def __init__(self, config: DistractionConfig):
        self.config = config
        self.yolo_engine = YOLOTrackerEngine(model_weights=config.MODEL_WEIGHTS)
        self.face_analyzer = MediaPipeFaceAnalyzer()
        self.person_states: Dict[int, PersonAttentionState] = {}

    def update_config(self, new_config: DistractionConfig):
        """Update live detection thresholds dynamically."""
        self.config = new_config

    def process_frame(
        self,
        frame: np.ndarray,
        yolo_result
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[int, PersonAttentionState]]:
        """
        Process a single frame result.
        Returns:
            annotated_frame: Frame with bounding boxes & status drawn
            new_violations: List of newly triggered violation dicts
            person_states: Current map of person states
        """
        now = time.time()
        frame_h, frame_w = frame.shape[:2]

        # 1. Extract YOLO detections
        people, phones = self.yolo_engine.extract_detections(yolo_result)
        people_with_phone = self.yolo_engine.match_phones_to_people(people, phones)

        # 2. Extract MediaPipe Face Landmarks + 3D Pose / Gaze
        face_analysis = self.face_analyzer.analyze_frame_faces(frame, people)

        new_violations: List[Dict[str, Any]] = []

        # 3. Update state for each tracked person
        for tid, box, conf in people:
            if tid not in self.person_states:
                self.person_states[tid] = PersonAttentionState(
                    track_id=tid,
                    first_seen=now,
                    last_seen=now
                )

            state = self.person_states[tid]
            state.last_seen = now

            # --- A. PHONE INTERACTION ---
            has_phone = tid in people_with_phone
            state.phone_detected = has_phone

            if has_phone:
                if state.phone_start is None:
                    state.phone_start = now
                    state.phone_violation_reported = False
                state.last_seen_with_phone = now
            elif state.last_seen_with_phone is not None and (now - state.last_seen_with_phone) > self.config.GRACE_PERIOD_SEC:
                state.phone_start = None
                state.last_seen_with_phone = None

            state.phone_duration = (now - state.phone_start) if state.phone_start is not None else 0.0

            if state.phone_duration >= self.config.PHONE_THRESHOLD_SEC and not state.phone_violation_reported:
                new_violations.append({
                    'person_id': tid,
                    'event_type': 'phone',
                    'duration': state.phone_duration,
                    'timestamp': now
                })
                state.phone_violation_reported = True

            # --- B. GAZE & HEAD POSE ---
            if tid in face_analysis:
                fa = face_analysis[tid]
                state.head_pitch = fa['pitch']
                state.head_yaw = fa['yaw']
                state.head_roll = fa['roll']
                state.gaze_horizontal_deg = fa['gaze_h_deg']
                state.gaze_vertical_deg = fa['gaze_v_deg']

                # Gaze away condition
                gaze_away_now = abs(fa['gaze_h_deg']) > self.config.GAZE_AWAY_THRESHOLD_DEG or abs(fa['gaze_v_deg']) > self.config.GAZE_AWAY_THRESHOLD_DEG
                # Head away / down condition
                head_down_now = fa['pitch'] > self.config.HEAD_DOWN_THRESHOLD_DEG or abs(fa['yaw']) > self.config.HEAD_YAW_THRESHOLD_DEG

                # Update Gaze Timer
                if gaze_away_now:
                    if state.gaze_away_start is None:
                        state.gaze_away_start = now
                        state.gaze_violation_reported = False
                    state.last_seen_gaze_away = now
                elif state.last_seen_gaze_away is not None and (now - state.last_seen_gaze_away) > self.config.GRACE_PERIOD_SEC:
                    state.gaze_away_start = None
                    state.last_seen_gaze_away = None

                # Update Head Down Timer
                if head_down_now:
                    if state.head_down_start is None:
                        state.head_down_start = now
                        state.head_violation_reported = False
                    state.last_seen_head_away = now
                elif state.last_seen_head_away is not None and (now - state.last_seen_head_away) > self.config.GRACE_PERIOD_SEC:
                    state.head_down_start = None
                    state.last_seen_head_away = None
            else:
                # Decay timers if face not detected this frame
                if state.last_seen_gaze_away is not None and (now - state.last_seen_gaze_away) > self.config.GRACE_PERIOD_SEC:
                    state.gaze_away_start = None
                    state.last_seen_gaze_away = None

                if state.last_seen_head_away is not None and (now - state.last_seen_head_away) > self.config.GRACE_PERIOD_SEC:
                    state.head_down_start = None
                    state.last_seen_head_away = None

            state.gaze_away_duration = (now - state.gaze_away_start) if state.gaze_away_start is not None else 0.0
            state.head_down_duration = (now - state.head_down_start) if state.head_down_start is not None else 0.0

            # Trigger gaze violation
            if state.gaze_away_duration >= self.config.GAZE_AWAY_THRESHOLD_SEC and not state.gaze_violation_reported:
                new_violations.append({
                    'person_id': tid,
                    'event_type': 'gaze_away',
                    'duration': state.gaze_away_duration,
                    'timestamp': now
                })
                state.gaze_violation_reported = True

            # Trigger head down violation
            if state.head_down_duration >= self.config.HEAD_DOWN_THRESHOLD_SEC and not state.head_violation_reported:
                new_violations.append({
                    'person_id': tid,
                    'event_type': 'head_down',
                    'duration': state.head_down_duration,
                    'timestamp': now
                })
                state.head_violation_reported = True

        # 4. Draw Annotations on Frame
        annotated_frame = frame.copy()

        for tid, box, conf in people:
            if tid not in self.person_states:
                continue

            state = self.person_states[tid]
            x1, y1, x2, y2 = map(int, box)

            color = (0, 255, 0)
            label = f"ID {tid}"

            if state.phone_duration >= self.config.PHONE_THRESHOLD_SEC:
                color = (0, 0, 255)
                label += f" [PHONE {state.phone_duration:.1f}s]"
            elif state.gaze_away_duration >= self.config.GAZE_AWAY_THRESHOLD_SEC:
                color = (0, 165, 255)
                label += f" [GAZE {state.gaze_away_duration:.1f}s]"
            elif state.head_down_duration >= self.config.HEAD_DOWN_THRESHOLD_SEC:
                color = (255, 0, 255)
                label += f" [HEAD {state.head_down_duration:.1f}s]"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for box, conf in phones:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 140, 0), 2)
            cv2.putText(annotated_frame, f"phone {conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)

        return annotated_frame, new_violations, self.person_states

    def close(self):
        self.face_analyzer.close()


if __name__ == '__main__':
    print("=" * 60)
    print("DissTraqt Detector Core Self-Test")
    print("=" * 60)
    cfg = DistractionConfig()
    core = DistractionDetectorCore(cfg)
    print("✓ DistractionDetectorCore initialized successfully!")
    print(f"✓ Configuration: Phone {cfg.PHONE_THRESHOLD_SEC}s, Gaze {cfg.GAZE_AWAY_THRESHOLD_SEC}s, Head {cfg.HEAD_DOWN_THRESHOLD_SEC}s")
    core.close()
    print("=" * 60)
