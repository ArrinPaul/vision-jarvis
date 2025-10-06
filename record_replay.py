import json
import time
from typing import List, Dict, Optional


class LandmarkRecorder:
    def __init__(self, path: str = "landmarks.jsonl"):
        self.path = path

    def write(self, landmarks: Optional[List[List[float]]], hand_info: Dict):
        entry = {
            "ts": time.time(),
            "landmarks": landmarks,
            "hand_info": hand_info,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


class LandmarkReplayer:
    def __init__(self, path: str):
        self.path = path
        self._file = None

    def __enter__(self):
        self._file = open(self.path, "r", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file:
            self._file.close()

    def __iter__(self):
        return self

    def __next__(self):
        line = self._file.readline()
        if not line:
            raise StopIteration
        obj = json.loads(line)
        return obj.get("landmarks"), obj.get("hand_info")

