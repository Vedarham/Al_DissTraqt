"""
MediaPipe Face Analyzer & Head Pose / Gaze Estimator
Uses 3D solvePnP for physical pitch/yaw/roll angle estimation in degrees.
Supports person bounding-box cropping fallback for small/distant faces in wide-angle video.
"""

import math
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
import mediapipe as mp

# Standard 3D Reference Face Model (in mm) for solvePnP
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (index 1)
    (0.0, -330.0, -65.0),     # Chin (index 152)
    (-225.0, 170.0, -135.0),  # Left eye corner (index 33)
    (225.0, 170.0, -135.0),   # Right eye corner (index 263)
    (-150.0, -150.0, -125.0), # Left mouth corner (index 61)
    (150.0, -150.0, -125.0),  # Right mouth corner (index 291)
], dtype=np.float64)

POSE_LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}

LEFT_EYE_RING = [33, 133, 159, 158, 157, 173, 155, 154]
RIGHT_EYE_RING = [263, 362, 386, 385, 384, 398, 382, 381]
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473


class MediaPipeFaceAnalyzer:
    def __init__(self, max_num_faces: int = 15, min_detection_confidence: float = 0.4):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.4,
        )

    def estimate_head_pose(self, face_landmarks, frame_w: int, frame_h: int) -> Optional[Tuple[float, float, float]]:
        """
        Estimate pitch, yaw, roll in degrees using 3D solvePnP.
        Returns (pitch, yaw, roll) where:
          - pitch > 0: looking down (head tilted down)
          - yaw > 0: turning right, yaw < 0: turning left
          - roll: head tilt sideways
        """
        try:
            image_points = []
            for name, idx in POSE_LANDMARK_IDS.items():
                lm = face_landmarks.landmark[idx]
                image_points.append([lm.x * frame_w, lm.y * frame_h])
            
            image_points = np.array(image_points, dtype=np.float64)

            focal_length = frame_w
            center = (frame_w / 2.0, frame_h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1))

            success, rotation_vec, translation_vec = cv2.solvePnP(
                FACE_3D_MODEL,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None

            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            sy = math.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)

            if sy >= 1e-6:
                raw_pitch = math.atan2(rotation_mat[2, 1], rotation_mat[2, 2])
                raw_yaw = math.atan2(-rotation_mat[2, 0], sy)
                raw_roll = math.atan2(rotation_mat[1, 0], rotation_mat[0, 0])
            else:
                raw_pitch = math.atan2(-rotation_mat[1, 2], rotation_mat[1, 1])
                raw_yaw = math.atan2(-rotation_mat[2, 0], sy)
                raw_roll = 0.0

            raw_pitch_deg = math.degrees(raw_pitch)
            raw_yaw_deg = math.degrees(raw_yaw)
            raw_roll_deg = math.degrees(raw_roll)

            # Camera perspective angle compensation:
            # Adjusts for face position relative to optical center so people sitting
            # upright in front or at frame edges are not falsely biased as head down / turned.
            nose_lm = face_landmarks.landmark[POSE_LANDMARK_IDS["nose_tip"]]
            cx, cy = nose_lm.x * frame_w, nose_lm.y * frame_h
            cam_pitch_offset = math.degrees(math.atan2(cy - center[1], focal_length))
            cam_yaw_offset = math.degrees(math.atan2(cx - center[0], focal_length))

            adjusted_pitch = raw_pitch_deg - cam_pitch_offset
            adjusted_yaw = raw_yaw_deg - cam_yaw_offset

            return adjusted_pitch, adjusted_yaw, raw_roll_deg
        except Exception:
            return None

    def estimate_gaze_offset_deg(self, face_landmarks) -> Tuple[float, float]:
        """
        Estimate horizontal and vertical gaze offset in degrees from iris landmarks.
        Returns (gaze_h_deg, gaze_v_deg).
        """
        try:
            left_iris = face_landmarks.landmark[LEFT_IRIS_CENTER]
            right_iris = face_landmarks.landmark[RIGHT_IRIS_CENTER]

            lx = [face_landmarks.landmark[i].x for i in LEFT_EYE_RING]
            rx = [face_landmarks.landmark[i].x for i in RIGHT_EYE_RING]
            ly = [face_landmarks.landmark[i].y for i in LEFT_EYE_RING]
            ry = [face_landmarks.landmark[i].y for i in RIGHT_EYE_RING]

            lw, rw = max(max(lx) - min(lx), 1e-6), max(max(rx) - min(rx), 1e-6)
            lh, rh = max(max(ly) - min(ly), 1e-6), max(max(ry) - min(ry), 1e-6)

            left_ratio_x = np.clip((left_iris.x - min(lx)) / lw, 0, 1)
            right_ratio_x = np.clip((right_iris.x - min(rx)) / rw, 0, 1)
            left_ratio_y = np.clip((left_iris.y - min(ly)) / lh, 0, 1)
            right_ratio_y = np.clip((right_iris.y - min(ry)) / rh, 0, 1)

            gaze_h = (left_ratio_x + right_ratio_x) / 2.0
            gaze_v = (left_ratio_y + right_ratio_y) / 2.0

            gaze_h_deg = (gaze_h - 0.5) * 80.0  # Horizontal pseudo-degrees
            gaze_v_deg = (gaze_v - 0.5) * 60.0  # Vertical pseudo-degrees

            return gaze_h_deg, gaze_v_deg
        except Exception:
            return 0.0, 0.0

    def analyze_frame_faces(
        self,
        frame: np.ndarray,
        person_boxes: List[Tuple[int, List[float], float]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Processes full frame + per-person crops to match face landmarks to person track IDs.
        Returns dict mapping person track ID -> {
            'landmarks': face_landmarks,
            'pitch': pitch, 'yaw': yaw, 'roll': roll,
            'gaze_h_deg': gaze_h_deg, 'gaze_v_deg': gaze_v_deg
        }
        """
        frame_h, frame_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Full-frame pass
        full_results = self.face_mesh.process(frame_rgb)
        matched_landmarks: Dict[int, Any] = {}

        if full_results and full_results.multi_face_landmarks:
            for fl in full_results.multi_face_landmarks:
                xs = [lm.x for lm in fl.landmark]
                ys = [lm.y for lm in fl.landmark]
                centroid_x = (sum(xs) / len(xs)) * frame_w
                centroid_y = (sum(ys) / len(ys)) * frame_h

                # Match to nearest person box
                best_tid, best_dist = None, float('inf')
                for tid, box, _ in person_boxes:
                    x1, y1, x2, y2 = box
                    # Expand box slightly for matching
                    bw, bh = x2 - x1, y2 - y1
                    if (x1 - 0.15 * bw) <= centroid_x <= (x2 + 0.15 * bw) and (y1 - 0.15 * bh) <= centroid_y <= (y2 + 0.15 * bh):
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        dist = (centroid_x - cx) ** 2 + (centroid_y - cy) ** 2
                        if dist < best_dist:
                            best_dist, best_tid = dist, tid

                if best_tid is not None:
                    matched_landmarks[best_tid] = fl

        # 2. Crop Fallback for people whose faces were NOT detected in full frame
        for tid, box, _ in person_boxes:
            if tid in matched_landmarks:
                continue

            x1, y1, x2, y2 = map(int, box)
            pw, ph = x2 - x1, y2 - y1
            if pw <= 10 or ph <= 10:
                continue

            # Upper 50% head/torso region crop
            hx1 = max(0, int(x1 - 0.1 * pw))
            hy1 = max(0, int(y1 - 0.05 * ph))
            hx2 = min(frame_w, int(x2 + 0.1 * pw))
            hy2 = min(frame_h, int(y1 + 0.5 * ph))

            crop = frame_rgb[hy1:hy2, hx1:hx2]
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            # Upscale crop if too small for MediaPipe
            ch, cw = crop.shape[:2]
            if ch < 128 or cw < 128:
                crop = cv2.resize(crop, (max(160, cw * 2), max(160, ch * 2)), interpolation=cv2.INTER_LINEAR)

            crop_results = self.face_mesh.process(crop)
            if crop_results and crop_results.multi_face_landmarks:
                # Use first face in crop
                matched_landmarks[tid] = crop_results.multi_face_landmarks[0]

        # 3. Compute pose & gaze metrics for all matched faces
        analysis: Dict[int, Dict[str, Any]] = {}
        for tid, fl in matched_landmarks.items():
            pose = self.estimate_head_pose(fl, frame_w, frame_h)
            pitch, yaw, roll = pose if pose is not None else (0.0, 0.0, 0.0)
            gaze_h_deg, gaze_v_deg = self.estimate_gaze_offset_deg(fl)

            analysis[tid] = {
                'landmarks': fl,
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                'gaze_h_deg': gaze_h_deg,
                'gaze_v_deg': gaze_v_deg,
            }

        return analysis

    def close(self):
        self.face_mesh.close()


if __name__ == '__main__':
    print("=" * 60)
    print("DissTraqt MediaPipe Face Analyzer Self-Test")
    print("=" * 60)
    analyzer = MediaPipeFaceAnalyzer()
    print("✓ MediaPipeFaceAnalyzer initialized successfully!")
    analyzer.close()
    print("=" * 60)
