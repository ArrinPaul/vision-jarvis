"""Semantic Memory Layer

Provides embedding-based storage and retrieval of textual (and lightweight
multimodal metadata) information for JARVIS. Designed to be pluggable so
that advanced vector DBs (FAISS, Chroma, Milvus) can be integrated later
without changing callers.

Features (Phase 1):
- Add item with text + optional metadata
- Fallback embedding (deterministic hashing + token stats) when
  no ML embedding model is available
- Similarity search (cosine over generated vectors)
- Basic persistence to JSON (optional, lazy)

Future (Phase 2+):
- Real embedding models (sentence-transformers, OpenAI, Gemini embeddings)
- Batch upsert, hybrid search, metadata filtering, TTL
- Event-driven auto-capture from other subsystems
"""
from __future__ import annotations

import os
import json
import math
import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:  # Optional heavy libs
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False

from core.event_bus import get_event_bus, Event

@dataclass
class SemanticItem:
    id: str
    text: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

class FallbackEmbedder:
    """Deterministic lightweight embedder.

    Strategy: tokenize on whitespace, hash each token to bucket, build a
    fixed-length vector with tf-like weighting + simple normalization.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for t in texts:
            buckets = [0.0] * self.dim
            tokens = t.lower().split()
            for tok in tokens:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                buckets[idx] += 1.0
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in buckets)) or 1.0
            vectors.append([v / norm for v in buckets])
        return vectors

class SemanticMemory:
    def __init__(self, storage_path: str = "semantic_memory.json", dim: int = 384):
        self.storage_path = storage_path
        self.dim = dim if _HAS_ST else 128  # fallback dimension
        self._lock = threading.RLock()
        self._items: Dict[str, SemanticItem] = {}
        self._model = SentenceTransformer("all-MiniLM-L6-v2") if _HAS_ST else None
        self._fallback = FallbackEmbedder(dim=self.dim if not _HAS_ST else 128)
        self._event_bus = get_event_bus()
        self._load()

    # Persistence -------------------------------------------------------
    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for rec in data.get("items", []):
                    self._items[rec["id"]] = SemanticItem(
                        id=rec["id"],
                        text=rec["text"],
                        vector=rec["vector"],
                        metadata=rec.get("metadata", {}),
                        created_at=rec.get("created_at", time.time())
                    )
            except Exception as e:
                print(f"[SemanticMemory] Failed to load: {e}")

    def _persist(self):
        try:
            data = {"items": [
                {
                    "id": it.id,
                    "text": it.text,
                    "vector": it.vector,
                    "metadata": it.metadata,
                    "created_at": it.created_at
                } for it in self._items.values()
            ]}
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[SemanticMemory] Persist error: {e}")

    # Core API ----------------------------------------------------------
    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None, *, persist: bool = False) -> str:
        import uuid
        with self._lock:
            vec = self._embed([text])[0]
            item_id = str(uuid.uuid4())
            item = SemanticItem(id=item_id, text=text, vector=vec, metadata=metadata or {})
            self._items[item_id] = item
            if persist:
                self._persist()
        # Emit event
        self._event_bus.publish("memory.semantic.added", {"id": item_id, "metadata": metadata or {}})
        return item_id

    def bulk_add(self, texts: List[str], metadatas: Optional[List[Optional[Dict[str, Any]]]] = None, *, persist: bool = False) -> List[str]:
        metadatas = metadatas or [None] * len(texts)
        ids: List[str] = []
        with self._lock:
            vectors = self._embed(texts)
            import uuid
            for text, meta, vec in zip(texts, metadatas, vectors):
                item_id = str(uuid.uuid4())
                self._items[item_id] = SemanticItem(id=item_id, text=text, vector=vec, metadata=meta or {})
                ids.append(item_id)
            if persist:
                self._persist()
        self._event_bus.publish("memory.semantic.bulk_added", {"count": len(ids)})
        return ids

    def get(self, item_id: str) -> Optional[SemanticItem]:
        return self._items.get(item_id)

    def search(self, query: str, k: int = 5) -> List[Tuple[SemanticItem, float]]:
        q_vec = self._embed([query])[0]
        with self._lock:
            scored: List[Tuple[SemanticItem, float]] = []
            for it in self._items.values():
                sim = self._cosine(q_vec, it.vector)
                scored.append((it, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def size(self) -> int:
        return len(self._items)

    # Helpers -----------------------------------------------------------
    def _embed(self, texts: List[str]) -> List[List[float]]:
        if self._model is not None:
            try:
                return [list(map(float, v)) for v in self._model.encode(texts)]
            except Exception as e:
                print(f"[SemanticMemory] Model encode failed, fallback: {e}")
        return self._fallback.embed(texts)

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        s = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return s / (na * nb)

# Simple self-test
if __name__ == "__main__":
    mem = SemanticMemory()
    mem.add("The quick brown fox jumps over the lazy dog", persist=False)
    mem.add("A fast auburn fox leaped above a sleepy canine", persist=False)
    res = mem.search("fast fox", k=2)
    for item, score in res:
        print(f"{score:.3f} :: {item.text[:50]}")
