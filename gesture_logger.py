import json
import time
from typing import Optional


class GestureLogger:
    def __init__(self, enabled: bool = False, path: str = "gesture_events.jsonl"):
        self.enabled = enabled
        self.path = path

    def log(self, event: str, data: Optional[dict] = None):
        if not self.enabled:
            return
        entry = {
            "ts": time.time(),
            "event": event,
            "data": data or {},
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            # Non-fatal
            print(f"GestureLogger write failed: {e}")

