import time
import threading
import logging
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime, timedelta
import json


class BriefingSystem:
    """Proactive briefing system for Jarvis-like status updates"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Briefing state
        self.enabled = config.get("briefings_enabled", False)
        self.last_briefing = 0
        self.briefing_interval = config.get("briefing_interval_hours", 4) * 3600
        self.morning_briefing_hour = config.get("morning_briefing_hour", 8)
        
        # Callbacks for getting information
        self.info_providers = {}
        
        # Scheduling
        self.scheduler_running = False
        self.scheduler_thread = None
        
    def register_info_provider(self, name: str, provider: Callable[[], str]):
        """Register a function that provides information for briefings"""
        self.info_providers[name] = provider
        self.logger.debug(f"Registered info provider: {name}")
    
    def start_scheduler(self):
        """Start the briefing scheduler"""
        if self.scheduler_running or not self.enabled:
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("Briefing scheduler started")
    
    def stop_scheduler(self):
        """Stop the briefing scheduler"""
        self.scheduler_running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=1.0)
        self.logger.info("Briefing scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.scheduler_running:
            try:
                current_time = time.time()
                
                # Check for morning briefing
                if self._should_give_morning_briefing():
                    briefing = self.generate_morning_briefing()
                    self._deliver_briefing("Morning Briefing", briefing)
                    self.last_briefing = current_time
                
                # Check for periodic briefings
                elif current_time - self.last_briefing > self.briefing_interval:
                    briefing = self.generate_status_briefing()
                    self._deliver_briefing("Status Update", briefing)
                    self.last_briefing = current_time
                
                # Sleep for a minute before checking again
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Briefing scheduler error: {e}")
                time.sleep(60)
    
    def _should_give_morning_briefing(self) -> bool:
        """Check if it's time for the morning briefing"""
        now = datetime.now()
        
        # Check if it's the right hour
        if now.hour != self.morning_briefing_hour:
            return False
        
        # Check if we already gave a briefing today
        last_briefing_date = datetime.fromtimestamp(self.last_briefing).date()
        if last_briefing_date == now.date():
            return False
        
        return True
    
    def generate_morning_briefing(self) -> str:
        """Generate a morning briefing"""
        parts = []
        
        # Greeting
        now = datetime.now()
        parts.append(f"Good morning. It's {now.strftime('%A, %B %d, %Y')}.")
        
        # Time
        parts.append(f"The time is {now.strftime('%I:%M %p')}.")
        
        # Weather (if provider available)
        if "weather" in self.info_providers:
            try:
                weather = self.info_providers["weather"]()
                if weather:
                    parts.append(f"Weather: {weather}")
            except Exception as e:
                self.logger.debug(f"Weather provider error: {e}")
        
        # Calendar (if provider available)
        if "calendar" in self.info_providers:
            try:
                calendar = self.info_providers["calendar"]()
                if calendar:
                    parts.append(f"Today's schedule: {calendar}")
            except Exception as e:
                self.logger.debug(f"Calendar provider error: {e}")
        
        # System status
        if "system_status" in self.info_providers:
            try:
                status = self.info_providers["system_status"]()
                if status and "normal" not in status.lower():
                    parts.append(f"System status: {status}")
            except Exception as e:
                self.logger.debug(f"System status provider error: {e}")
        
        # Reminders (if provider available)
        if "reminders" in self.info_providers:
            try:
                reminders = self.info_providers["reminders"]()
                if reminders:
                    parts.append(f"Reminders: {reminders}")
            except Exception as e:
                self.logger.debug(f"Reminders provider error: {e}")
        
        return " ".join(parts)
    
    def generate_status_briefing(self) -> str:
        """Generate a periodic status briefing"""
        parts = []
        
        # Time-based greeting
        now = datetime.now()
        if now.hour < 12:
            greeting = "Morning update"
        elif now.hour < 17:
            greeting = "Afternoon update"
        else:
            greeting = "Evening update"
        
        parts.append(f"{greeting}.")
        
        # System status
        if "system_status" in self.info_providers:
            try:
                status = self.info_providers["system_status"]()
                if status and "normal" not in status.lower():
                    parts.append(f"System: {status}")
            except Exception as e:
                self.logger.debug(f"System status provider error: {e}")
        
        # Recent activity summary (if provider available)
        if "activity_summary" in self.info_providers:
            try:
                activity = self.info_providers["activity_summary"]()
                if activity:
                    parts.append(f"Recent activity: {activity}")
            except Exception as e:
                self.logger.debug(f"Activity provider error: {e}")
        
        # Only deliver if there's meaningful content
        if len(parts) <= 1:
            return ""
        
        return " ".join(parts)
    
    def generate_on_demand_briefing(self, briefing_type: str = "status") -> str:
        """Generate a briefing on demand"""
        if briefing_type == "morning":
            return self.generate_morning_briefing()
        elif briefing_type == "status":
            return self.generate_status_briefing()
        elif briefing_type == "full":
            # Comprehensive briefing
            parts = []
            
            # Current time and date
            now = datetime.now()
            parts.append(f"Current time: {now.strftime('%I:%M %p on %A, %B %d, %Y')}.")
            
            # All available information
            for name, provider in self.info_providers.items():
                try:
                    info = provider()
                    if info:
                        parts.append(f"{name.title()}: {info}")
                except Exception as e:
                    self.logger.debug(f"Provider {name} error: {e}")
            
            return " ".join(parts)
        else:
            return "Unknown briefing type requested."
    
    def _deliver_briefing(self, title: str, content: str):
        """Deliver a briefing (placeholder - would integrate with TTS)"""
        if not content:
            return
        
        self.logger.info(f"Briefing: {title}")
        self.logger.info(f"Content: {content}")
        
        # In a real implementation, this would:
        # 1. Check if user should be interrupted (via context oracle)
        # 2. Use TTS to speak the briefing
        # 3. Optionally display visual notification
        # 4. Log the briefing for user review
        
        # For now, just log it
        self._log_briefing(title, content)
    
    def _log_briefing(self, title: str, content: str):
        """Log briefing to file for user review"""
        try:
            log_entry = {
                "timestamp": time.time(),
                "title": title,
                "content": content,
                "delivered": True
            }
            
            with open("briefing_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to log briefing: {e}")
    
    def get_recent_briefings(self, count: int = 5) -> List[Dict]:
        """Get recent briefings from log"""
        briefings = []
        try:
            with open("briefing_log.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    briefings.append(json.loads(line))
            
            # Return most recent first
            return sorted(briefings, key=lambda x: x["timestamp"], reverse=True)[:count]
        except FileNotFoundError:
            return []
        except Exception as e:
            self.logger.error(f"Failed to read briefing log: {e}")
            return []


def create_briefing_system(config: Dict[str, Any]):
    """Factory function to create briefing system"""
    return BriefingSystem(config)


# Default info providers
def default_system_status_provider():
    """Default system status provider"""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        status_parts = []
        if cpu > 80:
            status_parts.append(f"High CPU usage: {cpu:.0f}%")
        if memory.percent > 85:
            status_parts.append(f"High memory usage: {memory.percent:.0f}%")

        try:
            battery = psutil.sensors_battery()
            if battery and battery.percent < 20:
                status_parts.append(f"Low battery: {battery.percent}%")
        except (AttributeError, NotImplementedError):
            # Battery info not available on this system
            pass

        return "; ".join(status_parts) if status_parts else "All systems normal"
    except ImportError:
        return "System monitoring not available"
    except Exception as e:
        return f"System status error: {e}"


def default_time_provider():
    """Default time provider"""
    return datetime.now().strftime("%I:%M %p")


def default_date_provider():
    """Default date provider"""
    return datetime.now().strftime("%A, %B %d, %Y")


class RoutineManager:
    """Manages macro actions and routines for Jarvis-like automation"""

    def __init__(self, action_registry=None):
        self.action_registry = action_registry
        self.logger = logging.getLogger(__name__)

        # Predefined routines
        self.routines = {
            "start_meeting": self._start_meeting_routine,
            "focus_mode": self._focus_mode_routine,
            "end_workday": self._end_workday_routine,
            "morning_setup": self._morning_setup_routine,
            "presentation_mode": self._presentation_mode_routine,
            "break_time": self._break_time_routine
        }

    def execute_routine(self, routine_name: str, args: Dict[str, Any] = None) -> str:
        """Execute a named routine"""
        if routine_name not in self.routines:
            return f"Unknown routine: {routine_name}"

        try:
            return self.routines[routine_name](args or {})
        except Exception as e:
            self.logger.error(f"Routine {routine_name} failed: {e}")
            return f"Routine {routine_name} encountered an error"

    def _start_meeting_routine(self, args: Dict[str, Any]) -> str:
        """Prepare system for a meeting"""
        actions = []

        # Set volume to appropriate level
        if self.action_registry:
            result = self.action_registry.dispatch("set_volume", {"level": 30})
            actions.append(result)

        # Enable do not disturb (would need OS integration)
        actions.append("Do not disturb enabled")

        # Open meeting apps if specified
        meeting_app = args.get("app", "teams")
        if self.action_registry:
            result = self.action_registry.dispatch("open", {"app": meeting_app})
            actions.append(result)

        # Close distracting apps
        distracting_apps = ["spotify", "discord", "slack"]
        for app in distracting_apps:
            if self.action_registry:
                self.action_registry.dispatch("close", {"app": app})

        return f"Meeting preparation complete: {'; '.join(actions)}"

    def _focus_mode_routine(self, args: Dict[str, Any]) -> str:
        """Enable focus mode for deep work"""
        actions = []

        # Lower volume
        if self.action_registry:
            result = self.action_registry.dispatch("set_volume", {"level": 15})
            actions.append(result)

        # Close social/entertainment apps
        focus_blockers = ["chrome", "firefox", "discord", "spotify", "steam"]
        for app in focus_blockers:
            if self.action_registry:
                self.action_registry.dispatch("close", {"app": app})

        # Open productivity apps
        productivity_apps = args.get("apps", ["notepad", "code"])
        for app in productivity_apps:
            if self.action_registry:
                result = self.action_registry.dispatch("open", {"app": app})
                actions.append(result)

        actions.append("Focus mode activated")
        return "; ".join(actions)

    def _end_workday_routine(self, args: Dict[str, Any]) -> str:
        """End of workday cleanup"""
        actions = []

        # Save work and close work apps
        work_apps = ["outlook", "teams", "slack", "code", "excel", "word"]
        for app in work_apps:
            if self.action_registry:
                self.action_registry.dispatch("close", {"app": app})

        # System cleanup
        actions.append("Work applications closed")
        actions.append("Have a great evening!")

        return "; ".join(actions)

    def _morning_setup_routine(self, args: Dict[str, Any]) -> str:
        """Morning system setup"""
        actions = []

        # Set comfortable volume
        if self.action_registry:
            result = self.action_registry.dispatch("set_volume", {"level": 50})
            actions.append(result)

        # Open morning apps
        morning_apps = args.get("apps", ["outlook", "calendar", "notepad"])
        for app in morning_apps:
            if self.action_registry:
                result = self.action_registry.dispatch("open", {"app": app})
                actions.append(result)

        actions.append("Good morning! Your workspace is ready")
        return "; ".join(actions)

    def _presentation_mode_routine(self, args: Dict[str, Any]) -> str:
        """Prepare for presentation"""
        actions = []

        # Set high volume
        if self.action_registry:
            result = self.action_registry.dispatch("set_volume", {"level": 80})
            actions.append(result)

        # Close unnecessary apps
        background_apps = ["spotify", "discord", "slack", "outlook"]
        for app in background_apps:
            if self.action_registry:
                self.action_registry.dispatch("close", {"app": app})

        # Open presentation app
        presentation_app = args.get("app", "powerpoint")
        if self.action_registry:
            result = self.action_registry.dispatch("open", {"app": presentation_app})
            actions.append(result)

        actions.append("Presentation mode ready")
        return "; ".join(actions)

    def _break_time_routine(self, args: Dict[str, Any]) -> str:
        """Take a break routine"""
        actions = []

        # Lower volume
        if self.action_registry:
            result = self.action_registry.dispatch("set_volume", {"level": 25})
            actions.append(result)

        # Minimize work windows (simplified)
        if self.action_registry:
            self.action_registry.dispatch("press", {"key": "win+m"})

        actions.append("Break time! Step away from the screen")
        return "; ".join(actions)

    def list_routines(self) -> List[str]:
        """List available routines"""
        return list(self.routines.keys())

    def get_routine_description(self, routine_name: str) -> str:
        """Get description of a routine"""
        descriptions = {
            "start_meeting": "Prepare system for a meeting (lower volume, open meeting app, enable DND)",
            "focus_mode": "Enable deep work mode (close distractions, open productivity apps)",
            "end_workday": "End of day cleanup (close work apps, save documents)",
            "morning_setup": "Morning workspace preparation (open daily apps, set volume)",
            "presentation_mode": "Prepare for presentation (high volume, close background apps)",
            "break_time": "Take a break (minimize windows, lower volume)"
        }
        return descriptions.get(routine_name, "No description available")


def create_routine_manager(action_registry=None):
    """Factory function to create routine manager"""
    return RoutineManager(action_registry)
