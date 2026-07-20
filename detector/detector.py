"""
DissTraqt Standalone CLI Visual Demo
Runs frame-by-frame detection with OpenCV visualization without web API requirement.
"""

import sys
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import cv2
import config

try:
    from detector.detector_core import DistractionDetectorCore
except ImportError:
    from detector_core import DistractionDetectorCore


def main():
    print("=" * 60)
    print("DissTraqt Standalone Visual Detector")
    print("=" * 60)

    cfg = config.load_config()
    cfg.WEB_API_ENABLED = False  # Disable web API for standalone mode

    detector = DistractionDetectorCore(cfg)
    print("✓ Standalone detector core initialized.")
    print(f"Opening video source: {cfg.VIDEO_SOURCE!r}")
    print("Press 'q' in the window to quit.\n")

    frame_count = 0

    try:
        tracker = detector.yolo_engine.model.track(
            source=cfg.VIDEO_SOURCE,
            classes=[0, 67],
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
                break

            annotated_frame, violations, states = detector.process_frame(frame, result)

            for v in violations:
                print(f"⚠️  VIOLATION: Person {v['person_id']} | {v['event_type'].upper()} | {v['duration']:.1f}s")

            cv2.imshow("DissTraqt Standalone Detector", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"✗ Error: {e}")

    finally:
        detector.close()
        cv2.destroyAllWindows()
        print(f"\nDone. Processed {frame_count} frames.")


if __name__ == '__main__':
    main()