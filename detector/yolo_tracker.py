"""
YOLO Person & Phone Tracker Engine
Uses Ultralytics YOLO with ByteTrack object tracking.
Associates detected phones to tracked persons via spatial proximity.
"""

from typing import List, Tuple, Set, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

PERSON_CLASS = 0
PHONE_CLASS = 67


def box_center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def phone_near_person(person_box: List[float], phone_box: List[float], expand_ratio: float = 0.20) -> bool:
    """Check if phone center falls within expanded person bounding box."""
    px1, py1, px2, py2 = person_box
    w, h = px2 - px1, py2 - py1
    ex, ey = w * expand_ratio, h * expand_ratio
    ex1, ey1, ex2, ey2 = px1 - ex, py1 - ey, px2 + ex, py2 + ey
    cx, cy = box_center(phone_box)
    return ex1 <= cx <= ex2 and ey1 <= cy <= ey2


class YOLOTrackerEngine:
    def __init__(self, model_weights: str = "yolov8n.pt"):
        self.model_weights = model_weights
        self.model = YOLO(model_weights)

    def extract_detections(self, result) -> Tuple[List[Tuple[int, List[float], float]], List[Tuple[List[float], float]]]:
        """
        Extracts person tracks and phone detections from YOLO tracking result.
        Returns:
            people: List of (track_id, [x1, y1, x2, y2], confidence)
            phones: List of ([x1, y1, x2, y2], confidence)
        """
        people: List[Tuple[int, List[float], float]] = []
        phones: List[Tuple[List[float], float]] = []

        if result.boxes is None:
            return people, phones

        clss = result.boxes.cls.int().tolist() if result.boxes.cls is not None else []
        confs = result.boxes.conf.tolist() if result.boxes.conf is not None else []
        xyxys = result.boxes.xyxy.tolist() if result.boxes.xyxy is not None else []

        ids = result.boxes.id.int().tolist() if result.boxes.id is not None else list(range(len(clss)))

        for tid, cls, conf, box in zip(ids, clss, confs, xyxys):
            if cls == PERSON_CLASS:
                people.append((tid, box, conf))
            elif cls == PHONE_CLASS:
                phones.append((box, conf))

        return people, phones

    @staticmethod
    def match_phones_to_people(
        people: List[Tuple[int, List[float], float]],
        phones: List[Tuple[List[float], float]]
    ) -> Set[int]:
        """
        Matches detected phones to nearest tracked person.
        Returns set of person track IDs using phones.
        """
        people_with_phone: Set[int] = set()

        for pbox, _ in phones:
            best_tid, best_dist = None, float('inf')
            pcx, pcy = box_center(pbox)

            for tid, box, _ in people:
                if not phone_near_person(box, pbox):
                    continue
                bcx, bcy = box_center(box)
                d = (pcx - bcx) ** 2 + (pcy - bcy) ** 2
                if d < best_dist:
                    best_dist, best_tid = d, tid

            if best_tid is not None:
                people_with_phone.add(best_tid)

        return people_with_phone


if __name__ == '__main__':
    print("=" * 60)
    print("DissTraqt YOLO Tracker Engine Self-Test")
    print("=" * 60)
    engine = YOLOTrackerEngine()
    print("✓ YOLOTrackerEngine initialized successfully!")
    print("=" * 60)
