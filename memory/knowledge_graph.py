"""Knowledge Graph Layer

Stores entities and relations extracted from interactions, perception, and
system knowledge. Simplified for Phase 1 with in-memory structures and
basic query capabilities.

Phase 1 Features:
- Add / update entities with attributes
- Add directed relations with optional weight & metadata
- Query neighbors, entity retrieval, simple path existence (DFS limited)
- Event emission on mutations

Future (Phase 2+):
- Embedding alignment with semantic memory
- Graph algorithms (centrality, community detection)
- Temporal versioning, provenance tracking
- External persistence / graph DB adapter (Neo4j, Memgraph, RDF stores)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.event_bus import get_event_bus

@dataclass
class Entity:
    id: str
    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

@dataclass
class Relation:
    source: str
    target: str
    type: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

class KnowledgeGraph:
    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, List[Relation]] = {}  # source -> relations
        self._incoming: Dict[str, List[Relation]] = {}   # target -> relations
        self._lock = threading.RLock()
        self._event_bus = get_event_bus()

    # Entity Ops --------------------------------------------------------
    def upsert_entity(self, entity_id: str, type: str, **attributes) -> Entity:
        with self._lock:
            ent = self._entities.get(entity_id)
            if ent:
                ent.attributes.update(attributes)
                ent.type = type or ent.type
                ent.updated_at = time.time()
                action = "updated"
            else:
                ent = Entity(id=entity_id, type=type, attributes=attributes)
                self._entities[entity_id] = ent
                action = "created"
        self._event_bus.publish("memory.kg.entity", {"id": entity_id, "action": action})
        return ent

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def all_entities(self) -> List[Entity]:
        return list(self._entities.values())

    # Relation Ops ------------------------------------------------------
    def add_relation(self, source: str, target: str, type: str, *, weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> Relation:
        import uuid
        with self._lock:
            rel = Relation(source=source, target=target, type=type, weight=weight, metadata=metadata or {})
            self._relations.setdefault(source, []).append(rel)
            self._incoming.setdefault(target, []).append(rel)
        self._event_bus.publish("memory.kg.relation", {"source": source, "target": target, "type": type})
        return rel

    def get_relations_from(self, source: str, type: Optional[str] = None) -> List[Relation]:
        rels = self._relations.get(source, [])
        return [r for r in rels if type is None or r.type == type]

    def get_relations_to(self, target: str, type: Optional[str] = None) -> List[Relation]:
        rels = self._incoming.get(target, [])
        return [r for r in rels if type is None or r.type == type]

    # Simple Path -------------------------------------------------------
    def path_exists(self, start: str, end: str, max_depth: int = 4) -> bool:
        if start == end:
            return True
        visited: Set[str] = set()
        frontier: List[Tuple[str, int]] = [(start, 0)]
        while frontier:
            node, depth = frontier.pop()
            if depth >= max_depth:
                continue
            for rel in self._relations.get(node, []):
                if rel.target == end:
                    return True
                if rel.target not in visited:
                    visited.add(rel.target)
                    frontier.append((rel.target, depth + 1))
        return False

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entities": len(self._entities),
                "relations": sum(len(v) for v in self._relations.values())
            }

# Self-test
if __name__ == "__main__":
    kg = KnowledgeGraph()
    kg.upsert_entity("user:alice", "person", role="admin")
    kg.upsert_entity("device:camera1", "device", status="online")
    kg.add_relation("user:alice", "device:camera1", "controls")
    print(kg.path_exists("user:alice", "device:camera1"))
