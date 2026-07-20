"""
DissTraqt Web-Integrated Detector
Processes video frames, uses robust MediaPipe 3D solvePnP + YOLO tracking,
syncs settings dynamically with the Flask dashboard API, and streams real-time state.
"""

import sys
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import base64
from typing import Dict, List, Optional
import cv2
import requests
import threading

import config
from detector.detector_core import DistractionDetectorCore, PersonAttentionState


_last_api_error_time = 0.0

def send_violation_to_api(api_base: str, person_id: int, event_type: str, duration: float, frame=None):
    """Send violation event to Flask web API with screenshot frame"""
    global _last_api_error_time
    frame_b64 = None
    if frame is not None:
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"✗ Error encoding violation frame: {e}")

    try:
        data = {
            'person_id': person_id,
            'event_type': event_type,
            'duration': duration,
            'frame': frame_b64
        }
        response = requests.post(
            f'{api_base}/test/add-violation',
            json=data,
            timeout=3
        )
        if response.status_code == 200:
            print(f"✓ SENT TO API: Person {person_id} | {event_type.upper()} | Duration: {duration:.1f}s")
        else:
            print(f"✗ API error ({response.status_code}): {response.text}")
    except requests.exceptions.ConnectionError:
        now = time.time()
        if now - _last_api_error_time > 10.0:  # Warn once every 10 seconds max
            print(f"⚠️ API disconnected at {api_base} — start dashboard in another terminal: python run_dashboard.py")
            _last_api_error_time = now
    except Exception as e:
        print(f"✗ Error sending violation: {e}")


def send_heartbeat_to_api(api_base: str, active_people_count: int, active_person_ids: List[int], frame=None):
    """Send heartbeat state & live view frame to web dashboard"""
    frame_b64 = None
    if frame is not None:
        try:
            # Resize live view frame slightly to optimize network bandwidth
            h, w = frame.shape[:2]
            small_frame = cv2.resize(frame, (min(640, w), min(360, h)), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
        except Exception:
            pass

    try:
        data = {
            'active_people_count': active_people_count,
            'active_person_ids': active_person_ids,
            'frame': frame_b64
        }
        requests.post(
            f'{api_base}/update-state',
            json=data,
            timeout=1
        )
    except Exception:
        pass


def fetch_latest_config(api_base: str) -> Optional[config.DistractionConfig]:
    """Poll latest config from backend API or local JSON file."""
    try:
        response = requests.get(f'{api_base}/config', timeout=1)
        if response.status_code == 200:
            return config.DistractionConfig.from_dict(response.json())
    except Exception:
        pass
    return config.load_config()


def main():
    print("=" * 60)
    print("DissTraqt Web-Integrated Detector Engine")
    print("=" * 60)

    # 1. Load initial configuration
    cfg = config.load_config()
    print(f"📋 Initial Thresholds:")
    print(f"   Phone: {cfg.PHONE_THRESHOLD_SEC}s | Gaze: {cfg.GAZE_AWAY_THRESHOLD_SEC}s | Head: {cfg.HEAD_DOWN_THRESHOLD_SEC}s")
    print(f"   Video Source: {cfg.VIDEO_SOURCE}")
    print(f"   Web API: {cfg.API_BASE}")

    # Check Web API status
    if cfg.WEB_API_ENABLED:
        try:
            r = requests.get(f"{cfg.API_BASE}/summary", timeout=2)
            if r.status_code == 200:
                print(f"✓ Web Dashboard API connected at {cfg.API_BASE}")
            else:
                print(f"⚠️ Web Dashboard API returned status {r.status_code}")
        except Exception:
            print(f"⚠️ Web Dashboard API is disconnected at {cfg.API_BASE}")
            print(f"   👉 To connect live dashboard, run in another terminal: python run_dashboard.py")

    # 2. Initialize Detector Core
    detector = DistractionDetectorCore(cfg)
    print("✓ YOLO + MediaPipe 3D solvePnP detector core initialized.")

    frame_count = 0
    start_time = time.time()

    print(f"\n🚀 Starting video tracking on {cfg.VIDEO_SOURCE}...")
    print("   Press 'q' in the OpenCV window to exit.\n")

    try:
        tracker = detector.yolo_engine.model.track(
            source=cfg.VIDEO_SOURCE,
            classes=[0, 67],  # 0=person, 67=cell phone
            conf=cfg.CONF_THRESHOLD,
            persist=True,
            tracker="bytetrack.yaml",
            stream=True,
            verbose=False,
        )

        for result in tracker:
            frame_count += 1
            frame = result.orig_img

            if frame is None:
                print(f"✗ Frame {frame_count} is None, ending.")
                break

            # Poll & update config dynamically every 20 frames (~0.6s)
            if frame_count % 20 == 0 and cfg.WEB_API_ENABLED:
                latest_cfg = fetch_latest_config(cfg.API_BASE)
                if latest_cfg:
                    detector.update_config(latest_cfg)
                    cfg = latest_cfg

            # Process frame through Detector Core
            annotated_frame, violations, states = detector.process_frame(frame, result)

            # Send triggered violations to API
            for v in violations:
                if cfg.WEB_API_ENABLED:
                    send_violation_to_api(
                        cfg.API_BASE,
                        v['person_id'],
                        v['event_type'],
                        v['duration'],
                        annotated_frame
                    )

            # Send heartbeat & live view to API every 10 frames (~0.3s)
            if frame_count % 10 == 0 and cfg.WEB_API_ENABLED:
                active_ids = [s.track_id for s in states.values() if (time.time() - s.last_seen) < 2.0]
                send_heartbeat_to_api(
                    cfg.API_BASE,
                    len(active_ids),
                    active_ids,
                    annotated_frame
                )

            # Display Status overlay
            status_line = (
                f"Frame {frame_count} | Tracked: {len(states)} | "
                f"Phone Thresh: {cfg.PHONE_THRESHOLD_SEC:.1f}s | "
                f"Gaze Thresh: {cfg.GAZE_AWAY_THRESHOLD_SEC:.1f}s"
            )
            cv2.putText(annotated_frame, status_line, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 2)
            cv2.putText(annotated_frame, "Press 'q' to quit", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            # Display Visualization Window
            cv2.imshow("DissTraqt Web Detector", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n👋 Quit signal received.")
                break

    except Exception as e:
        print(f"\n❌ Error in main loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        detector.close()
        cv2.destroyAllWindows()
        print(f"\n" + "=" * 60)
        print(f"✓ Detector stopped cleanly. Total frames processed: {frame_count}")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
