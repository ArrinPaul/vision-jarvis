import json
import os
from dataclasses import dataclass
from typing import Optional

DEFAULTS = {
    "cooldowns": {
        "playpause": 0.2,
        "like": 0.2,
        "swipe": 0.5,
        "palette": 0.5,
        "seek_update": 0.05
    },
    "thresholds": {
        "pinch_on": 0.035,
        "pinch_off": 0.045,
        "swipe_speed": 0.04,
        "peace_sep": 0.03,
        "fist_hold_sec": 1.0
    },
    "filters": {
        "min_cutoff": 1.2,
        "beta": 0.02,
        "d_cutoff": 1.0
    },
    "logging": {
        "enabled": False,
        "path": "gesture_events.jsonl"
    }
}


def load_config(path: str = "thresholds.json") -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Shallow merge
                cfg = DEFAULTS.copy()
                for k, v in (data or {}).items():
                    if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                        cfg[k].update(v)
                    else:
                        cfg[k] = v
                return cfg
        except Exception as e:
            print(f"Failed to load {path}, using defaults: {e}")
    return DEFAULTS.copy()


def save_config(cfg: dict, path: str = "thresholds.json"):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"Failed to save config: {e}")

