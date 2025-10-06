"""Intent Fusion Module

Combines multiple input signals (speech transcript, gesture classification,
contextual memory lookups, system state) into a unified structured Intent
object for downstream planners / action executors.

Phase 1 Scope:
- Deterministic rule-based fusion with scoring heuristics
- Input signals provided as dictionary parts
- Produces Intent with: user_text, gesture, inferred_action, confidence,
  slots (key parameters), and trace (explanation of fusion reasoning)

Future (Phase 2+):
- ML-based intent ranking, multi-turn dialogue grounding
- Integration with tool / action descriptor metadata for slot filling
- Adaptive weighting via feedback loop
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from core.event_bus import get_event_bus

@dataclass
class Intent:
    id: str
    user_text: Optional[str]
    gesture: Optional[str]
    inferred_action: Optional[str]
    confidence: float
    slots: Dict[str, Any] = field(default_factory=dict)
    trace: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

class IntentFusionEngine:
    def __init__(self):
        self._event_bus = get_event_bus()

    def fuse(self, *, speech: Optional[Dict[str, Any]] = None, gesture: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None, extras: Optional[Dict[str, Any]] = None) -> Intent:
        import uuid
        trace: List[str] = []
        user_text = None
        gesture_label = None
        inferred_action = None
        confidence = 0.0
        slots: Dict[str, Any] = {}

        # Speech extraction
        if speech and speech.get("text"):
            user_text = speech["text"].strip()
            base_conf = float(speech.get("confidence", 0.6))
            confidence = max(confidence, base_conf)
            trace.append(f"Speech provided (conf={base_conf:.2f})")

        # Gesture extraction
        if gesture and gesture.get("label"):
            gesture_label = gesture["label"]
            g_conf = float(gesture.get("confidence", 0.5))
            confidence = max(confidence, g_conf)
            trace.append(f"Gesture provided (label={gesture_label}, conf={g_conf:.2f})")

        # Rule-based action inference
        inferred_action, action_score, action_slots, rule_explanation = self._infer_action(user_text, gesture_label, context or {})
        if inferred_action:
            confidence = max(confidence, action_score)
            slots.update(action_slots)
            trace.append(rule_explanation)

        # Context adjustments
        if context:
            if context.get("recent_failures"):
                confidence *= 0.95
                trace.append("Adjusted confidence due to recent failures")
            if context.get("priority_mode"):
                confidence *= 1.05
                trace.append("Boosted confidence due to priority mode")

        # Normalize confidence
        confidence = max(0.0, min(1.0, confidence))

        intent = Intent(
            id=str(uuid.uuid4()),
            user_text=user_text,
            gesture=gesture_label,
            inferred_action=inferred_action,
            confidence=confidence,
            slots=slots,
            trace=trace
        )
        self._event_bus.publish("intent.fused", {"id": intent.id, "action": intent.inferred_action, "confidence": intent.confidence})
        return intent

    # Simple heuristic rule set -----------------------------------------
    def _infer_action(self, user_text: Optional[str], gesture_label: Optional[str], context: Dict[str, Any]):
        if not user_text and not gesture_label:
            return None, 0.0, {}, "No actionable signals"

        # Examples of deterministic mapping
        normalized = (user_text or "").lower()

        # Mapping table (expandable)
        if any(kw in normalized for kw in ["open camera", "start camera", "show camera"]):
            return "camera.open", 0.85, {}, "Rule: camera open phrase"
        if "screenshot" in normalized:
            return "system.screenshot", 0.8, {}, "Rule: screenshot keyword"
        if "volume" in normalized and ("up" in normalized or "increase" in normalized):
            return "system.volume_up", 0.75, {"delta": 10}, "Rule: volume increase phrase"
        if gesture_label == "thumbs_up":
            return "acknowledge", 0.6, {}, "Rule: thumbs_up gesture -> acknowledge"
        if gesture_label == "pinch" and "drag" in normalized:
            return "ui.drag_mode", 0.55, {}, "Rule: pinch + drag phrase"

        # Fallback
        return None, 0.3, {}, "Low-confidence fallback"

# Self-test
if __name__ == "__main__":
    engine = IntentFusionEngine()
    intent = engine.fuse(speech={"text": "please open camera", "confidence": 0.9}, gesture={"label": "neutral", "confidence": 0.4})
    print(intent)
