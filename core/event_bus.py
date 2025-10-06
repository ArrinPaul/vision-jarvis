"""Event Bus Infrastructure for JARVIS

Provides a lightweight publish/subscribe system for decoupled communication
between subsystems (vision, audio, memory, intent, actions, remote access, etc.).

Design Goals:
- Simple synchronous publish for low-latency in-thread dispatch
- Optional asynchronous queue-based dispatch for high-volume topics
- Wildcard topic subscription (e.g., "memory.*", "vision.hand.*")
- Event object with metadata, tracing, and timing
- Pluggable middleware (filters / transformers / metrics) per subscription
- Singleton access pattern via get_event_bus()

Future Extensions (Phase 2+):
- Persistence / replay for episodic memory capture
- Distributed propagation (remote nodes, mobile devices)
- Priority-based dispatch / cancellation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import threading
import queue
import fnmatch
import uuid

EventHandler = Callable[["Event"], Any]
Middleware = Callable[["Event"], Optional["Event"]]

@dataclass
class Event:
    topic: str
    payload: Dict[str, Any]
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

    def duration_ms(self) -> float:
        return (time.time() - self.created_at) * 1000.0

class Subscription:
    __slots__ = ("topic_pattern", "handler", "middleware", "async_mode", "queue", "thread", "active")

    def __init__(self, topic_pattern: str, handler: EventHandler, middleware: Optional[List[Middleware]] = None, async_mode: bool = False):
        self.topic_pattern = topic_pattern
        self.handler = handler
        self.middleware = middleware or []
        self.async_mode = async_mode
        self.queue: Optional[queue.Queue] = queue.Queue() if async_mode else None
        self.thread: Optional[threading.Thread] = None
        self.active = True
        if async_mode:
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def _run_loop(self):
        assert self.queue is not None
        while self.active:
            try:
                event = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._dispatch(event)
            except Exception as e:
                # Basic error logging; can be expanded
                print(f"[EventBus] Async handler error for {self.topic_pattern}: {e}")

    def _dispatch(self, event: Event):
        # Apply middleware chain
        for mw in self.middleware:
            try:
                event = mw(event)
                if event is None:
                    return  # Filtered out
            except Exception as e:
                print(f"[EventBus] Middleware error: {e}")
        self.handler(event)

    def submit(self, event: Event):
        if not self.active:
            return
        if self.async_mode and self.queue is not None:
            try:
                self.queue.put_nowait(event)
            except queue.Full:
                print(f"[EventBus] Dropping event (queue full) for pattern {self.topic_pattern}")
        else:
            try:
                self._dispatch(event)
            except Exception as e:
                print(f"[EventBus] Handler error for {self.topic_pattern}: {e}")

    def close(self):
        self.active = False

class EventBus:
    def __init__(self):
        self._subs: List[Subscription] = []
        self._lock = threading.RLock()
        self._metrics = {
            "published_total": 0,
            "delivered_total": 0,
            "dropped_total": 0,
            "subscriptions": 0
        }

    # Public API ---------------------------------------------------------
    def subscribe(self, topic_pattern: str, handler: EventHandler, *, middleware: Optional[List[Middleware]] = None, async_mode: bool = False) -> Subscription:
        """Subscribe a handler to a topic pattern.

        topic_pattern supports fnmatch wildcards ('*', '?', '[]').
        Set async_mode=True to process events on a dedicated thread.
        """
        sub = Subscription(topic_pattern, handler, middleware, async_mode)
        with self._lock:
            self._subs.append(sub)
            self._metrics["subscriptions"] = len(self._subs)
        return sub

    def unsubscribe(self, subscription: Subscription):
        with self._lock:
            if subscription in self._subs:
                subscription.close()
                self._subs.remove(subscription)
                self._metrics["subscriptions"] = len(self._subs)

    def publish(self, topic: str, payload: Dict[str, Any], *, source: Optional[str] = None, correlation_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Event:
        event = Event(topic=topic, payload=payload, source=source, correlation_id=correlation_id or str(uuid.uuid4()), meta=meta or {})
        with self._lock:
            subs_snapshot = list(self._subs)
        self._metrics["published_total"] += 1

        delivered = 0
        for sub in subs_snapshot:
            if fnmatch.fnmatch(topic, sub.topic_pattern):
                sub.submit(event)
                delivered += 1
        self._metrics["delivered_total"] += delivered
        return event

    def metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)

    def close(self):
        with self._lock:
            for sub in self._subs:
                sub.close()
            self._subs.clear()
            self._metrics["subscriptions"] = 0

# Singleton helper -------------------------------------------------------
_global_event_bus: Optional[EventBus] = None
_global_lock = threading.Lock()

def get_event_bus() -> EventBus:
    global _global_event_bus
    if _global_event_bus is None:
        with _global_lock:
            if _global_event_bus is None:
                _global_event_bus = EventBus()
    return _global_event_bus

# Simple self-test when run directly
if __name__ == "__main__":
    bus = get_event_bus()

    def printer(evt: Event):
        print(f"[Printer] {evt.topic} -> {evt.payload} (latency={evt.duration_ms():.2f}ms)")

    bus.subscribe("demo.*", printer)
    bus.publish("demo.start", {"msg": "hello"}, source="self_test")
    bus.publish("demo.update", {"count": 1})
    time.sleep(0.1)
