"""
DissTraqt Web Dashboard Backend - Flask API

Provides endpoints for:
- Video processing and frame capture
- Real-time monitoring data
- Violation reports
- Person tracking visualization
"""

from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import json
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

# ============= CONFIGURATION =============
DATA_DIR = Path("./dashboard_data")
DATA_DIR.mkdir(exist_ok=True)

SCREENSHOTS_DIR = DATA_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ============= DATA CLASSES =============

@dataclass
class ViolationEvent:
    """Records a single violation event"""
    person_id: int
    event_type: str  # 'phone', 'gaze_away', 'head_down'
    timestamp: float
    duration: float
    screenshot_path: str = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class PersonReport:
    """Summary report for a person"""
    person_id: int
    total_phone_events: int = 0
    total_phone_duration: float = 0.0
    total_gaze_events: int = 0
    total_gaze_duration: float = 0.0
    total_head_events: int = 0
    total_head_duration: float = 0.0
    violations: list = None
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []
    
    @property
    def total_violations(self):
        return len(self.violations)
    
    @property
    def distraction_score(self):
        """0-100 score of total distraction"""
        return min(100, (self.total_phone_events * 10 + 
                        self.total_gaze_events * 5 + 
                        self.total_head_events * 5))
    
    def to_dict(self):
        return {
            'person_id': self.person_id,
            'total_phone_events': self.total_phone_events,
            'total_phone_duration': round(self.total_phone_duration, 2),
            'total_gaze_events': self.total_gaze_events,
            'total_gaze_duration': round(self.total_gaze_duration, 2),
            'total_head_events': self.total_head_events,
            'total_head_duration': round(self.total_head_duration, 2),
            'total_violations': self.total_violations,
            'distraction_score': self.distraction_score,
            'violations': [v.to_dict() if isinstance(v, ViolationEvent) else v for v in self.violations]
        }


# ============= GLOBAL STATE =============

class MonitoringState:
    def __init__(self):
        self.people_reports: dict = {}  # person_id -> PersonReport
        self.current_frame = None
        self.current_frame_time = None
        self.session_start_time = None
        self.processing = False
        self.lock = threading.Lock()
    
    def add_violation(self, person_id, event_type, duration, screenshot_path=None):
        """Record a violation event"""
        with self.lock:
            if person_id not in self.people_reports:
                self.people_reports[person_id] = PersonReport(person_id=person_id)
            
            report = self.people_reports[person_id]
            violation = ViolationEvent(
                person_id=person_id,
                event_type=event_type,
                timestamp=time.time(),
                duration=duration,
                screenshot_path=screenshot_path
            )
            
            report.violations.append(violation)
            
            # Update counters
            if event_type == 'phone':
                report.total_phone_events += 1
                report.total_phone_duration += duration
            elif event_type == 'gaze_away':
                report.total_gaze_events += 1
                report.total_gaze_duration += duration
            elif event_type == 'head_down':
                report.total_head_events += 1
                report.total_head_duration += duration
    
    def get_summary(self):
        """Get current monitoring summary"""
        with self.lock:
            return {
                'session_start': self.session_start_time,
                'total_people': len(self.people_reports),
                'total_violations': sum(r.total_violations for r in self.people_reports.values()),
                'people': [r.to_dict() for r in self.people_reports.values()],
            }
    
    def save_screenshot(self, frame, person_id, event_type):
        """Save screenshot of violation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"person_{person_id}_{event_type}_{timestamp}.jpg"
        filepath = SCREENSHOTS_DIR / filename
        
        cv2.imwrite(str(filepath), frame)
        return str(filepath.relative_to(DATA_DIR))
    
    def reset(self):
        """Reset monitoring state"""
        with self.lock:
            self.people_reports.clear()
            self.session_start_time = time.time()


monitor_state = MonitoringState()
monitor_state.session_start_time = time.time()


# ============= FLASK ROUTES =============

@app.route('/')
def index():
    """Serve main dashboard"""
    return render_template('dashboard.html')


@app.route('/api/summary')
def get_summary():
    """Get current monitoring summary"""
    return jsonify(monitor_state.get_summary())


@app.route('/api/person/<int:person_id>')
def get_person_details(person_id):
    """Get detailed report for a person"""
    with monitor_state.lock:
        if person_id in monitor_state.people_reports:
            return jsonify(monitor_state.people_reports[person_id].to_dict())
        return jsonify({'error': 'Person not found'}), 404


@app.route('/api/violations')
def get_violations():
    """Get all violations"""
    with monitor_state.lock:
        all_violations = []
        for report in monitor_state.people_reports.values():
            for violation in report.violations:
                v_dict = violation.to_dict() if isinstance(violation, ViolationEvent) else violation
                all_violations.append(v_dict)
        
        # Sort by timestamp
        all_violations.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(all_violations)


@app.route('/api/violations/type/<event_type>')
def get_violations_by_type(event_type):
    """Get violations of a specific type"""
    with monitor_state.lock:
        violations = []
        for report in monitor_state.people_reports.values():
            for violation in report.violations:
                v_dict = violation.to_dict() if isinstance(violation, ViolationEvent) else violation
                if v_dict['event_type'] == event_type:
                    violations.append(v_dict)
        
        violations.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(violations)


@app.route('/api/statistics')
def get_statistics():
    """Get statistical summary"""
    with monitor_state.lock:
        stats = {
            'total_people_tracked': len(monitor_state.people_reports),
            'phone_violations': sum(r.total_phone_events for r in monitor_state.people_reports.values()),
            'gaze_violations': sum(r.total_gaze_events for r in monitor_state.people_reports.values()),
            'head_violations': sum(r.total_head_events for r in monitor_state.people_reports.values()),
            'avg_distraction_score': 0,
            'most_distracted_person': None,
            'session_duration_minutes': round((time.time() - monitor_state.session_start_time) / 60, 2)
        }
        
        if monitor_state.people_reports:
            scores = [r.distraction_score for r in monitor_state.people_reports.values()]
            stats['avg_distraction_score'] = round(sum(scores) / len(scores), 1)
            
            most_distracted = max(monitor_state.people_reports.values(), 
                                 key=lambda r: r.distraction_score)
            stats['most_distracted_person'] = {
                'id': most_distracted.person_id,
                'score': most_distracted.distraction_score
            }
        
        return jsonify(stats)


@app.route('/api/report/export')
def export_report():
    """Export full report as JSON"""
    report_data = {
        'generated_at': datetime.now().isoformat(),
        'session_duration_minutes': round((time.time() - monitor_state.session_start_time) / 60, 2),
        'summary': monitor_state.get_summary(),
        'statistics': json.loads(jsonify(get_statistics()).get_json())
    }
    
    # Save to file
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = REPORTS_DIR / filename
    
    with open(filepath, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    return jsonify({
        'status': 'success',
        'file': filename,
        'path': str(filepath)
    })


@app.route('/api/test/add-violation', methods=['POST'])
def test_add_violation():
    """Test endpoint: Add a sample violation"""
    data = request.json or {}
    person_id = data.get('person_id', 1)
    event_type = data.get('event_type', 'phone')
    duration = data.get('duration', 10.0)
    
    monitor_state.add_violation(person_id, event_type, duration)
    
    return jsonify({
        'status': 'success',
        'message': f'Added {event_type} violation for person {person_id}'
    })


@app.route('/api/reset', methods=['POST'])
def reset_monitoring():
    """Reset monitoring state"""
    monitor_state.reset()
    return jsonify({'status': 'success', 'message': 'Monitoring state reset'})


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'people_tracked': len(monitor_state.people_reports)
    })


# ============= ERROR HANDLERS =============

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error', 'message': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("DissTraqt Dashboard Server Starting")
    print("="*60)
    print(f"📊 Dashboard URL: http://localhost:5000")
    print(f"📁 Data Directory: {DATA_DIR.absolute()}")
    print(f"📸 Screenshots: {SCREENSHOTS_DIR.absolute()}")
    print(f"📄 Reports: {REPORTS_DIR.absolute()}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
