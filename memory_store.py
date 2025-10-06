import json
import time
from typing import Dict, Any, List, Optional
from collections import deque
import logging


class MemoryStore:
    """Short-term memory for Jarvis-like context and preferences"""
    
    def __init__(self, max_entries: int = 100, persistence_file: Optional[str] = "jarvis_memory.json"):
        self.max_entries = max_entries
        self.persistence_file = persistence_file
        self.logger = logging.getLogger(__name__)
        
        # Memory categories
        self.recent_actions = deque(maxlen=max_entries)
        self.preferences = {}
        self.context_history = deque(maxlen=50)
        self.user_patterns = {}
        
        # Load persistent memory
        self._load_memory()
    
    def remember_action(self, action: str, args: Dict[str, Any], result: str, success: bool = True):
        """Remember a user action and its outcome"""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "args": args,
            "result": result,
            "success": success
        }
        self.recent_actions.append(entry)
        self.logger.debug(f"Remembered action: {action}")
    
    def remember_preference(self, key: str, value: Any):
        """Remember a user preference"""
        self.preferences[key] = {
            "value": value,
            "timestamp": time.time(),
            "usage_count": self.preferences.get(key, {}).get("usage_count", 0) + 1
        }
        self.logger.debug(f"Remembered preference: {key} = {value}")
        self._save_memory()
    
    def remember_context(self, context_type: str, data: Dict[str, Any]):
        """Remember contextual information"""
        entry = {
            "timestamp": time.time(),
            "type": context_type,
            "data": data
        }
        self.context_history.append(entry)
    
    def get_last_action(self, action_type: Optional[str] = None) -> Optional[Dict]:
        """Get the most recent action, optionally filtered by type"""
        for action in reversed(self.recent_actions):
            if action_type is None or action["action"] == action_type:
                return action
        return None
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference"""
        pref = self.preferences.get(key)
        if pref:
            # Update usage count
            pref["usage_count"] += 1
            return pref["value"]
        return default
    
    def get_recent_actions(self, count: int = 5, action_type: Optional[str] = None) -> List[Dict]:
        """Get recent actions, optionally filtered by type"""
        actions = []
        for action in reversed(self.recent_actions):
            if action_type is None or action["action"] == action_type:
                actions.append(action)
                if len(actions) >= count:
                    break
        return actions
    
    def get_context_summary(self) -> str:
        """Get a summary of recent context"""
        if not self.context_history:
            return "No recent context"
        
        recent_contexts = list(self.context_history)[-5:]
        summary_parts = []
        
        for ctx in recent_contexts:
            ctx_type = ctx["type"]
            data = ctx["data"]
            
            if ctx_type == "media_state":
                if data.get("playing"):
                    summary_parts.append(f"Media playing: {data.get('title', 'unknown')}")
            elif ctx_type == "active_window":
                summary_parts.append(f"Active: {data.get('title', 'unknown app')}")
            elif ctx_type == "system_state":
                if data.get("do_not_disturb"):
                    summary_parts.append("Do not disturb mode")
        
        return "; ".join(summary_parts) if summary_parts else "Normal operation"
    
    def learn_pattern(self, pattern_key: str, data: Any):
        """Learn user patterns for anticipatory behavior"""
        if pattern_key not in self.user_patterns:
            self.user_patterns[pattern_key] = []
        
        self.user_patterns[pattern_key].append({
            "timestamp": time.time(),
            "data": data
        })
        
        # Keep only recent patterns
        if len(self.user_patterns[pattern_key]) > 20:
            self.user_patterns[pattern_key] = self.user_patterns[pattern_key][-20:]
    
    def get_pattern_prediction(self, pattern_key: str) -> Optional[Any]:
        """Get prediction based on learned patterns"""
        patterns = self.user_patterns.get(pattern_key, [])
        if len(patterns) < 3:
            return None
        
        # Simple frequency-based prediction
        recent_patterns = patterns[-10:]
        data_counts = {}
        
        for pattern in recent_patterns:
            data = str(pattern["data"])
            data_counts[data] = data_counts.get(data, 0) + 1
        
        if data_counts:
            most_common = max(data_counts.items(), key=lambda x: x[1])
            return most_common[0]
        
        return None
    
    def handle_follow_up(self, query: str) -> Optional[str]:
        """Handle follow-up queries like 'same as before', 'undo that'"""
        query_lower = query.lower().strip()
        
        if any(phrase in query_lower for phrase in ["same as before", "repeat that", "do it again"]):
            last_action = self.get_last_action()
            if last_action and last_action["success"]:
                return f"Repeating: {last_action['action']} with {last_action['args']}"
        
        elif any(phrase in query_lower for phrase in ["undo that", "reverse that", "go back"]):
            last_action = self.get_last_action()
            if last_action:
                # Simple undo logic for common actions
                action = last_action["action"]
                if action == "set_volume":
                    return "I can't undo volume changes automatically. Please specify the desired volume."
                elif action in ["open", "close"]:
                    opposite = "close" if action == "open" else "open"
                    app = last_action["args"].get("app", "")
                    return f"To undo, try: {opposite} {app}"
                else:
                    return f"I can't automatically undo '{action}'. Please specify what you'd like to do."
        
        elif any(phrase in query_lower for phrase in ["what did i", "what was", "last action"]):
            last_action = self.get_last_action()
            if last_action:
                return f"Your last action was: {last_action['action']} - {last_action['result']}"
        
        return None
    
    def _load_memory(self):
        """Load persistent memory from file"""
        if not self.persistence_file:
            return
        
        try:
            with open(self.persistence_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.preferences = data.get("preferences", {})
                self.user_patterns = data.get("user_patterns", {})
                self.logger.info("Memory loaded from file")
        except FileNotFoundError:
            self.logger.info("No existing memory file found")
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Memory file corrupted, starting fresh: {e}")
            self.preferences = {}
            self.user_patterns = {}
        except Exception as e:
            self.logger.error(f"Failed to load memory: {e}")
    
    def _save_memory(self):
        """Save persistent memory to file"""
        if not self.persistence_file:
            return

        try:
            data = {
                "preferences": self.preferences,
                "user_patterns": self.user_patterns,
                "last_saved": time.time()
            }
            # Create backup before saving
            import os
            if os.path.exists(self.persistence_file):
                backup_file = f"{self.persistence_file}.backup"
                try:
                    import shutil
                    shutil.copy2(self.persistence_file, backup_file)
                except Exception:
                    pass  # Backup failed, but continue with save

            with open(self.persistence_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save memory: {e}")
    
    def cleanup_old_entries(self, max_age_hours: int = 24):
        """Clean up old memory entries"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        # Clean recent actions
        self.recent_actions = deque(
            [action for action in self.recent_actions if action["timestamp"] > cutoff_time],
            maxlen=self.max_entries
        )
        
        # Clean context history
        self.context_history = deque(
            [ctx for ctx in self.context_history if ctx["timestamp"] > cutoff_time],
            maxlen=50
        )
        
        self.logger.info(f"Cleaned up memory entries older than {max_age_hours} hours")


def create_memory_store(max_entries: int = 100, persistence_file: Optional[str] = "jarvis_memory.json"):
    """Factory function to create memory store"""
    return MemoryStore(max_entries, persistence_file)
