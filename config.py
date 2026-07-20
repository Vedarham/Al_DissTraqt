"""
DissTraqt Central Configuration Manager
Handles system settings, distraction detection thresholds, and persistence via JSON.
Optimized for 18-second video clips with fast, responsive detection.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any

CONFIG_FILE_PATH = Path("./dashboard_data/config.json")

@dataclass
class DistractionConfig:
    # Duration thresholds (seconds) - Optimized for 18s video clips
    PHONE_THRESHOLD_SEC: float = 1.0
    GAZE_AWAY_THRESHOLD_SEC: float = 1.0
    HEAD_DOWN_THRESHOLD_SEC: float = 1.2

    # Angle sensitivity thresholds (degrees)
    GAZE_AWAY_THRESHOLD_DEG: float = 20.0
    HEAD_DOWN_THRESHOLD_DEG: float = 22.0
    HEAD_YAW_THRESHOLD_DEG: float = 25.0

    # Detection & grace periods
    CONF_THRESHOLD: float = 0.25
    GRACE_PERIOD_SEC: float = 1.0

    # Source & API settings
    VIDEO_SOURCE: str = "./assets/classroom.mp4"
    MODEL_WEIGHTS: str = "yolov8n.pt"
    API_BASE: str = "http://localhost:5000/api"
    WEB_API_ENABLED: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistractionConfig":
        valid_keys = cls().__dict__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        # Cast numeric values appropriately
        for key in ['PHONE_THRESHOLD_SEC', 'GAZE_AWAY_THRESHOLD_SEC', 'HEAD_DOWN_THRESHOLD_SEC',
                    'GAZE_AWAY_THRESHOLD_DEG', 'HEAD_DOWN_THRESHOLD_DEG', 'HEAD_YAW_THRESHOLD_DEG',
                    'CONF_THRESHOLD', 'GRACE_PERIOD_SEC']:
            if key in filtered:
                try:
                    filtered[key] = float(filtered[key])
                except (ValueError, TypeError):
                    pass
        return cls(**filtered)

def load_config() -> DistractionConfig:
    """Load configuration from disk if available, otherwise return defaults."""
    if CONFIG_FILE_PATH.exists():
        try:
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return DistractionConfig.from_dict(data)
        except Exception as e:
            print(f"⚠️ Warning loading config from {CONFIG_FILE_PATH}: {e}. Using defaults.")
    
    config = DistractionConfig()
    save_config(config)
    return config

def save_config(config: DistractionConfig) -> bool:
    """Save current configuration to disk."""
    try:
        CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error saving config to {CONFIG_FILE_PATH}: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("DissTraqt Config Manager Self-Test")
    print("=" * 60)
    cfg = load_config()
    print("✓ Config loaded successfully:")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    print("=" * 60)
