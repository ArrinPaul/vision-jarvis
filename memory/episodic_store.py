"""Episodic Memory Store

Captures time-ordered 'episodes' – notable events, interactions, or state
summaries. Complements semantic memory (vector similarity) with chronological
context for temporal reasoning and narrative generation.

Phase 1 Goals:
- Append episode entries with timestamp, type, summary, and references
- Query recent episodes, range queries
- Simple compaction (merge older small events into summaries)
- Emits events over the event bus when new episodes are added

Future (Phase 2+):
- Importance scoring, decay, consolidation into long-term summaries
- Cross-link with knowledge graph entities
- Retrieval augmentation for LLM prompting
"""
from __future__ import annotations

import time
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.event_bus import get_event_bus

@dataclass
class Episode:
    id: str
    timestamp: float
    type: str
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)
    refs: List[str] = field(default_factory=list)  # semantic IDs, entity IDs, etc.

class EpisodicStore:
    def __init__(self, storage_path: str = "episodic_memory.json"):
        self.storage_path = storage_path
        self._episodes: List[Episode] = []
        self._lock = threading.RLock()
        self._event_bus = get_event_bus()
        self._load()

    # Persistence -------------------------------------------------------
    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for rec in data.get("episodes", []):
                    self._episodes.append(Episode(
                        id=rec["id"],
                        timestamp=rec["timestamp"],
                        type=rec["type"],
                        summary=rec["summary"],
                        data=rec.get("data", {}),
                        refs=rec.get("refs", [])
                    ))
            except Exception as e:
                print(f"[EpisodicStore] Load fail: {e}")

    def _persist(self):
        try:
            data = {"episodes": [
                {
                    "id": ep.id,
                    "timestamp": ep.timestamp,
                    "type": ep.type,
                    "summary": ep.summary,
                    "data": ep.data,
                    "refs": ep.refs
                } for ep in self._episodes
            ]}
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[EpisodicStore] Persist fail: {e}")

    # Core API ----------------------------------------------------------
    def add_episode(self, type: str, summary: str, *, data: Optional[Dict[str, Any]] = None, refs: Optional[List[str]] = None, persist: bool = False) -> str:
        import uuid
        ep = Episode(id=str(uuid.uuid4()), timestamp=time.time(), type=type, summary=summary, data=data or {}, refs=refs or [])
        with self._lock:
            self._episodes.append(ep)
            if persist:
                self._persist()
        self._event_bus.publish("memory.episodic.added", {"id": ep.id, "type": type})
        return ep.id

    def recent(self, limit: int = 20, types: Optional[List[str]] = None) -> List[Episode]:
        with self._lock:
            filtered = self._episodes if not types else [ep for ep in self._episodes if ep.type in types]
            return list(reversed(filtered[-limit:]))

    def range(self, start_ts: float, end_ts: float) -> List[Episode]:
        with self._lock:
            return [ep for ep in self._episodes if start_ts <= ep.timestamp <= end_ts]

    def size(self) -> int:
        return len(self._episodes)

    def compact(self, older_than: float, *, persist: bool = False):
        """Very simple compaction: group small events into a summary episode."""
        cutoff = time.time() - older_than
        with self._lock:
            older = [ep for ep in self._episodes if ep.timestamp < cutoff]
            if len(older) < 5:
                return  # Not enough to compact
            import uuid
            summary = {
                "count": len(older),
                "types": {},
            }
            for ep in older:
                summary["types"][ep.type] = summary["types"].get(ep.type, 0) + 1
            comp_id = str(uuid.uuid4())
            summary_ep = Episode(
                id=comp_id,
                timestamp=time.time(),
                type="summary.compaction",
                summary=f"Compacted {len(older)} episodes",
                data=summary,
                refs=[ep.id for ep in older]
            )
            # Remove older & add summary
            self._episodes = [ep for ep in self._episodes if ep.timestamp >= cutoff] + [summary_ep]
            if persist:
                self._persist()
        self._event_bus.publish("memory.episodic.compacted", {"id": comp_id, "removed": len(older)})

# Self-test
if __name__ == "__main__":
    store = EpisodicStore()
    for i in range(3):
        store.add_episode("interaction", f"User issued command {i}")
    print([ep.summary for ep in store.recent()])
