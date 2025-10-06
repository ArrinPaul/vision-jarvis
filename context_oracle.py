import logging
import time
from typing import Dict, Any, Optional, List
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import win32gui
    import win32process
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    # Provide fallback for non-Windows systems
    win32gui = None
    win32process = None


class ContextOracle:
    """Provides situational awareness for Jarvis-like behavior"""
    
    def __init__(self, update_interval: float = 2.0):
        self.update_interval = update_interval
        self.logger = logging.getLogger(__name__)
        
        # Current context state
        self.current_context = {
            "active_window": {"title": "", "process": "", "timestamp": 0},
            "media_state": {"playing": False, "title": "", "volume": 0, "timestamp": 0},
            "system_state": {"cpu_usage": 0, "memory_usage": 0, "battery": None, "timestamp": 0},
            "do_not_disturb": False,
            "focus_mode": False,
            "last_user_activity": time.time()
        }
        
        # Background monitoring
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start background context monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Context monitoring started")
    
    def stop_monitoring(self):
        """Stop background context monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Context monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                self._update_active_window()
                self._update_system_state()
                self._update_media_state()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Context monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _update_active_window(self):
        """Update active window information"""
        if not WIN32_AVAILABLE:
            return
        
        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                window_title = win32gui.GetWindowText(hwnd)
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                
                if PSUTIL_AVAILABLE:
                    try:
                        process = psutil.Process(pid)
                        process_name = process.name()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        process_name = "unknown"
                else:
                    process_name = "unknown"
                
                self.current_context["active_window"] = {
                    "title": window_title,
                    "process": process_name,
                    "timestamp": time.time()
                }
        except Exception as e:
            self.logger.debug(f"Failed to get active window: {e}")
    
    def _update_system_state(self):
        """Update system state information"""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            cpu_usage = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            battery_info = None
            try:
                battery = psutil.sensors_battery()
                if battery:
                    battery_info = {
                        "percent": battery.percent,
                        "plugged": battery.power_plugged,
                        "time_left": battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
                    }
            except (AttributeError, NotImplementedError, Exception):
                pass
            
            self.current_context["system_state"] = {
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "battery": battery_info,
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.debug(f"Failed to get system state: {e}")
    
    def _update_media_state(self):
        """Update media state (simplified - would need OS-specific media APIs)"""
        # This is a placeholder - real implementation would use:
        # - Windows: Windows.Media.Control APIs
        # - macOS: MediaPlayer framework
        # - Linux: MPRIS D-Bus interface
        
        # For now, just maintain timestamp
        if "timestamp" not in self.current_context["media_state"]:
            self.current_context["media_state"]["timestamp"] = time.time()
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get current context snapshot"""
        return self.current_context.copy()
    
    def get_active_app(self) -> str:
        """Get the currently active application"""
        return self.current_context["active_window"]["process"]
    
    def get_active_window_title(self) -> str:
        """Get the currently active window title"""
        return self.current_context["active_window"]["title"]
    
    def is_media_playing(self) -> bool:
        """Check if media is currently playing"""
        return self.current_context["media_state"]["playing"]
    
    def is_system_busy(self) -> bool:
        """Check if system is under high load"""
        cpu = self.current_context["system_state"]["cpu_usage"]
        memory = self.current_context["system_state"]["memory_usage"]
        return cpu > 80 or memory > 85
    
    def is_focus_mode(self) -> bool:
        """Check if user is in focus mode"""
        return self.current_context["focus_mode"]
    
    def is_do_not_disturb(self) -> bool:
        """Check if do not disturb is enabled"""
        return self.current_context["do_not_disturb"]
    
    def set_focus_mode(self, enabled: bool):
        """Set focus mode state"""
        self.current_context["focus_mode"] = enabled
        self.logger.info(f"Focus mode {'enabled' if enabled else 'disabled'}")
    
    def set_do_not_disturb(self, enabled: bool):
        """Set do not disturb state"""
        self.current_context["do_not_disturb"] = enabled
        self.logger.info(f"Do not disturb {'enabled' if enabled else 'disabled'}")
    
    def update_user_activity(self):
        """Update last user activity timestamp"""
        self.current_context["last_user_activity"] = time.time()
    
    def get_context_for_gesture(self, gesture_type: str) -> Dict[str, Any]:
        """Get relevant context for interpreting a gesture"""
        context = {
            "gesture": gesture_type,
            "active_app": self.get_active_app(),
            "media_playing": self.is_media_playing(),
            "system_busy": self.is_system_busy(),
            "focus_mode": self.is_focus_mode(),
            "timestamp": time.time()
        }
        
        # Add gesture-specific context
        if gesture_type in ["swipe", "next", "previous"]:
            context["media_context"] = self.current_context["media_state"]
        elif gesture_type in ["volume", "rotation"]:
            context["audio_context"] = {
                "current_volume": self.current_context["media_state"].get("volume", 50)
            }
        
        return context
    
    def should_interrupt_user(self) -> bool:
        """Determine if it's appropriate to interrupt the user"""
        if self.is_do_not_disturb() or self.is_focus_mode():
            return False
        
        if self.is_system_busy():
            return False
        
        # Check if user has been inactive
        inactive_time = time.time() - self.current_context["last_user_activity"]
        if inactive_time < 30:  # User was active in last 30 seconds
            return False
        
        # Check active application context
        active_app = self.get_active_app().lower()
        if any(app in active_app for app in ["game", "video", "stream", "meeting", "zoom", "teams"]):
            return False
        
        return True
    
    def get_contextual_greeting(self) -> str:
        """Get a context-appropriate greeting"""
        current_hour = time.localtime().tm_hour
        
        if current_hour < 12:
            greeting = "Good morning"
        elif current_hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"
        
        # Add context
        if self.is_focus_mode():
            return f"{greeting}. I see you're in focus mode."
        elif self.is_system_busy():
            return f"{greeting}. Your system seems busy right now."
        elif self.is_media_playing():
            return f"{greeting}. I notice you have media playing."
        else:
            return f"{greeting}. How can I assist you?"
    
    def get_status_summary(self) -> str:
        """Get a brief status summary"""
        parts = []
        
        # System status
        sys_state = self.current_context["system_state"]
        if sys_state["cpu_usage"] > 50:
            parts.append(f"CPU at {sys_state['cpu_usage']:.0f}%")
        
        # Battery status
        if sys_state["battery"]:
            battery = sys_state["battery"]
            if battery["percent"] < 20:
                parts.append(f"Battery low: {battery['percent']}%")
            elif not battery["plugged"]:
                parts.append(f"On battery: {battery['percent']}%")
        
        # Active context
        if self.is_focus_mode():
            parts.append("Focus mode active")
        if self.is_do_not_disturb():
            parts.append("Do not disturb")
        
        return "; ".join(parts) if parts else "All systems normal"


def create_context_oracle(update_interval: float = 2.0):
    """Factory function to create context oracle"""
    return ContextOracle(update_interval)
