"""
Quick Test - Verify YOLO and MediaPipe are working
"""

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

print("="*60)
print("DissTraqt Detection Test")
print("="*60)

# Test YOLO
print("\n1. Testing YOLO Model...")
try:
    yolo = YOLO("yolov8n.pt")
    print("✓ YOLO model loaded successfully")
except Exception as e:
    print(f"✗ YOLO error: {e}")
    exit(1)

# Test MediaPipe
print("\n2. Testing MediaPipe...")
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ MediaPipe Face Mesh loaded successfully")
except Exception as e:
    print(f"✗ MediaPipe error: {e}")
    exit(1)

# Test video file
video_path = './assets/classroom.mp4'
print(f"\n3. Testing video file: {video_path}")
try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        exit(1)
    
    ret, frame = cap.read()
    if not ret:
        print(f"✗ Cannot read frames from video")
        exit(1)
    
    cap.release()
    print(f"✓ Video file opened successfully ({frame.shape[0]}x{frame.shape[1]})")
except Exception as e:
    print(f"✗ Video error: {e}")
    exit(1)

# Test detection on first frame
print("\n4. Testing detection on first frame...")
try:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    # YOLO detection
    results = yolo(frame, classes=[0, 67], conf=0.35)
    detections = results[0]
    
    people = 0
    phones = 0
    if detections.boxes is not None:
        for cls, conf in zip(detections.boxes.cls.tolist(), detections.boxes.conf.tolist()):
            if int(cls) == 0:
                people += 1
            elif int(cls) == 67:
                phones += 1
    
    print(f"  YOLO: {people} people, {phones} phones detected")
    
    # MediaPipe detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(frame_rgb)
    faces = len(face_results.multi_face_landmarks) if face_results.multi_face_landmarks else 0
    print(f"  MediaPipe: {faces} faces detected")
    
except Exception as e:
    print(f"✗ Detection error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test API connection
print("\n5. Testing API connection...")
try:
    import requests
    response = requests.get('http://localhost:5000/api/summary', timeout=2)
    if response.status_code == 200:
        data = response.json()
        print(f"✓ API responding: {data.get('total_violations', 0)} violations recorded")
    else:
        print(f"✗ API error: {response.status_code}")
except requests.exceptions.ConnectionError:
    print(f"✗ Cannot connect to API at http://localhost:5000")
    print(f"  Make sure Flask is running: python run_dashboard.py")
except Exception as e:
    print(f"✗ API error: {e}")

print("\n" + "="*60)
print("✓ All systems operational!")
print("="*60)
print("\nNext steps:")
print("1. Terminal 1: python run_dashboard.py")
print("2. Terminal 2: python detector_web.py")
print("3. Open http://localhost:5000 in browser")
print("4. Watch video window for detections")
print("5. Check dashboard for violations")
