"""Unified Memory Layer Exports

Provides convenient constructors / singletons for JARVIS advanced memory
layers added in Phase 1 (semantic, episodic, knowledge graph). This does
not remove or break existing `memory_store.py`; instead acts as an optional
augmentation path.
"""
from __future__ import annotations

from typing import Optional
from .semantic_memory import SemanticMemory
from .episodic_store import EpisodicStore
from .knowledge_graph import KnowledgeGraph

_semantic: Optional[SemanticMemory] = None
_episodic: Optional[EpisodicStore] = None
_kg: Optional[KnowledgeGraph] = None

def get_semantic_memory() -> SemanticMemory:
    global _semantic
    if _semantic is None:
        _semantic = SemanticMemory()
    return _semantic

def get_episodic_store() -> EpisodicStore:
    global _episodic
    if _episodic is None:
        _episodic = EpisodicStore()
    return _episodic

def get_knowledge_graph() -> KnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
    return _kg
