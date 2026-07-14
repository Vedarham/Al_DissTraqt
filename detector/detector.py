"""
QUICK VISUAL DEMO — just run this and watch it work.

No CLI args, no saving to disk. Opens your webcam (or a video file if you
set VIDEO_SOURCE below), runs detection+tracking frame by frame, and shows
a live window so you can literally see the video capture -> detection ->
timer logic happening in real time.

Press 'q' in the window to quit.
"""

import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

# ---------------- SETTINGS (edit these directly) ----------------
VIDEO_SOURCE = './assets/classroom.mp4'              # 0 = default webcam. Or set to "path/to/video.mp4"
MODEL_WEIGHTS = "yolo26n.pt"  # auto-downloads on first run
DISTRACTION_THRESHOLD_SEC = 10
GRACE_PERIOD_SEC = 1.5
CONF_THRESHOLD = 0.35
PERSON_CLASS = 0
PHONE_CLASS = 67
# ------------------------------------------------------------------

model = YOLO(MODEL_WEIGHTS)

# per-person-track-id -> when their current phone-interaction streak started
interaction_start = {}
last_seen_with_phone = {}


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def phone_near_person(person_box, phone_box, expand_ratio=0.15):
    px1, py1, px2, py2 = person_box
    w, h = px2 - px1, py2 - py1
    ex, ey = w * expand_ratio, h * expand_ratio
    ex1, ey1, ex2, ey2 = px1 - ex, py1 - ey, px2 + ex, py2 + ey
    cx, cy = box_center(phone_box)
    return ex1 <= cx <= ex2 and ey1 <= cy <= ey2


print(f"Opening video source: {VIDEO_SOURCE!r} ...")
print("A window should pop up. Press 'q' to quit.\n")

# --- STEP 1: THIS is the actual video capture happening ---
# model.track(..., stream=True) opens VIDEO_SOURCE internally (webcam index
# or file path) using OpenCV under the hood, and yields one `result` per
# frame as it's read — so this for-loop IS the frame-by-frame capture loop.
frame_count = 0
for result in model.track(
    source=VIDEO_SOURCE,
    classes=[PERSON_CLASS, PHONE_CLASS],
    conf=CONF_THRESHOLD,
    persist=True,
    tracker="bytetrack.yaml",
    stream=True,
    verbose=False,
):
    frame_count += 1
    frame = result.orig_img            # the raw captured frame (numpy array, BGR)
    now = time.time()

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

    # attribute each phone to its nearest person
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

    # update timers + draw
    for tid, box, conf in people:
        phone_present = tid in people_with_phone

        if phone_present:
            if tid not in interaction_start:
                interaction_start[tid] = now
            last_seen_with_phone[tid] = now
        else:
            if tid in last_seen_with_phone and (now - last_seen_with_phone[tid]) > GRACE_PERIOD_SEC:
                interaction_start.pop(tid, None)
                last_seen_with_phone.pop(tid, None)

        duration = now - interaction_start[tid] if tid in interaction_start else 0.0
        distracted = duration >= DISTRACTION_THRESHOLD_SEC

        color = (0, 0, 255) if distracted else (0, 200, 0)
        label = f"ID {tid} person {conf:.2f}"
        if duration > 0:
            label += f" | phone {duration:.1f}s"
        if distracted:
            label += " DISTRACTED"

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for box, conf in phones:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)
        cv2.putText(frame, f"phone {conf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)

    # small on-screen counter so you can SEE frames being captured live
    cv2.putText(frame, f"frame #{frame_count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # --- STEP 2: display the frame we just captured + annotated ---
    cv2.imshow("Distraction Detector - live demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
print(f"\nDone. Processed {frame_count} frames.")