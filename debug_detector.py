"""
DissTraqt Debug Detector - Trace every detection and violation
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import requests
from pathlib import Path

# Configuration
VIDEO_SOURCE = './assets/classroom.mp4'
MODEL_WEIGHTS = "yolov8n.pt"
PHONE_THRESHOLD_SEC = 10
GAZE_AWAY_THRESHOLD_SEC = 5
HEAD_DOWN_THRESHOLD_SEC = 8
CONF_THRESHOLD = 0.35
PERSON_CLASS = 0
PHONE_CLASS = 67
API_BASE = 'http://localhost:5000/api'

print("="*60)
print("DissTraqt Debug Detector")
print("="*60)

# Test API first
print("\n1. Testing API connection...")
try:
    response = requests.get(f'{API_BASE}/summary', timeout=2)
    print("✓ API is responding")
except Exception as e:
    print(f"✗ API not responding: {e}")
    print(f"   Start Flask with: python run_dashboard.py")
    exit(1)

# Load YOLO
print("\n2. Loading YOLO...")
try:
    model = YOLO(MODEL_WEIGHTS)
    print("✓ YOLO loaded")
except Exception as e:
    print(f"✗ YOLO error: {e}")
    exit(1)

# Open video
print(f"\n3. Opening video: {VIDEO_SOURCE}")
try:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"✗ Cannot open video")
        exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ Video opened: {total_frames} frames @ {fps:.0f} FPS")
    cap.release()
except Exception as e:
    print(f"✗ Video error: {e}")
    exit(1)

# Process video
print(f"\n4. Processing video (this will take a while)...")
print("-"*60)

frame_count = 0
people_detections = 0
phone_detections = 0
violations_sent = 0
person_ids_seen = set()

print(f"{'Frame':<8} {'People':<8} {'Phones':<8} {'IDs':<20} {'API':<3}")
print("-"*60)

try:
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
        frame = result.orig_img
        
        people_this = 0
        phones_this = 0
        ids_this = set()
        
        if result.boxes is not None:
            ids = result.boxes.id
            clss = result.boxes.cls
            
            if ids is not None:
                for tid, cls in zip(ids.int().tolist(), clss.int().tolist()):
                    if cls == PERSON_CLASS:
                        people_this += 1
                        ids_this.add(tid)
                        person_ids_seen.add(tid)
                    elif cls == PHONE_CLASS:
                        phones_this += 1
        
        people_detections += people_this
        phone_detections += phones_this
        
        # Print every 100 frames
        if frame_count % 100 == 0:
            ids_str = str(list(ids_this)[:5]).replace('[', '').replace(']', '')
            api_str = 'OK' if violations_sent > 0 else '--'
            print(f"{frame_count:<8} {people_this:<8} {phones_this:<8} {ids_str:<20} {api_str:<3}")
        
        # Stop after 500 frames for testing
        if frame_count >= 500:
            print(f"\n(Stopped after 500 frames for testing)")
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DETECTION RESULTS:")
print("="*60)
print(f"Frames processed: {frame_count}")
print(f"Total people detections: {people_detections}")
print(f"Total phone detections: {phone_detections}")
print(f"Unique person IDs: {len(person_ids_seen)}")
print(f"Person IDs seen: {sorted(person_ids_seen)}")

print("\nCONCLUSION:")
if frame_count == 0:
    print("✗ No frames processed - check video file")
elif people_detections == 0:
    print("✗ No people detected - video might not have people")
    print("  Try with a different video or check YOLO model")
elif phone_detections == 0:
    print("⚠ No phones detected - video might not have phones")
    print("  Phone detection is optional, people detection is key")
else:
    print("✓ Detection is working!")
    print(f"  People: {people_detections} detections, {len(person_ids_seen)} unique")
    print(f"  Phones: {phone_detections} detections")
    print("\nNow run the full detector with:")
    print("  python detector_web.py")
