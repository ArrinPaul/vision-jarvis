import os
import subprocess
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
except Exception:
    AudioUtilities = None
    IAudioEndpointVolume = None

try:
    from briefing_system import create_routine_manager, create_briefing_system
    BRIEFING_AVAILABLE = True
except ImportError:
    BRIEFING_AVAILABLE = False

# ----------------------------------------------------------------------
# Action metadata layer (Phase 1)
# ----------------------------------------------------------------------
@dataclass
class ActionDescriptor:
    name: str
    func: Callable[[Dict[str, Any]], Any]
    description: str = ""
    category: str = "general"
    args_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)  # capabilities / permissions
    returns: str = "string"

    def invoke(self, args: Optional[Dict[str, Any]] = None):
        return self.func(args or {})

# Helper to build schema entries succinctly
_DEF = lambda typ, req=False, default=None, desc="": {"type": typ, "required": req, "default": default, "description": desc}

class ActionRegistry:
    def __init__(self):
        # Initialize routine manager if available
        if BRIEFING_AVAILABLE:
            self.routine_manager = create_routine_manager(self)
            self.briefing_system = create_briefing_system({
                "briefings_enabled": False,  # Default off
                "briefing_interval_hours": 4,
                "morning_briefing_hour": 8
            })
        else:
            self.routine_manager = None
            self.briefing_system = None
        # New: descriptor registry (name -> ActionDescriptor)
        self._descriptors: Dict[str, ActionDescriptor] = {}
        self._build_descriptors()

    # ------------------------------------------------------------------
    # Volume helpers
    def _volume_endpoint(self):
        if AudioUtilities is None:
            return None
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))

    # ------------------------------------------------------------------
    # Original action methods (kept for backward compatibility)
    # ------------------------------------------------------------------
    def open(self, args: Dict[str, Any]):
        app = args.get("app")
        if not app:
            return "Missing app name"
        try:
            subprocess.Popen(app)
            return f"Opening {app}"
        except Exception:
            # Try via start if not directly executable
            os.system(f'start "" "{app}"')
            return f"Opening {app}"

    def close(self, args: Dict[str, Any]):
        app = args.get("app")
        if not app:
            return "Missing app name"
        os.system(f'taskkill /im "{app}.exe" /f')
        return f"Closing {app}"

    def playpause(self, args: Dict[str, Any]):
        if pyautogui:
            pyautogui.press("playpause")
        return "Toggling play/pause"

    def next(self, args: Dict[str, Any]):
        if pyautogui:
            pyautogui.press("nexttrack")
        return "Next track"

    def previous(self, args: Dict[str, Any]):
        if pyautogui:
            pyautogui.press("prevtrack")
        return "Previous track"

    def set_volume(self, args: Dict[str, Any]):
        level = int(args.get("level", 50))
        vol = self._volume_endpoint()
        if vol is None:
            return "Volume control not available"
        scalar = max(0.0, min(1.0, level / 100.0))
        vol.SetMasterVolumeLevelScalar(scalar, None)
        return f"Volume set to {level}%"

    def volume(self, args: Dict[str, Any]):
        delta = int(args.get("delta", 10))
        vol = self._volume_endpoint()
        if vol is None:
            return "Volume control not available"
        cur = vol.GetMasterVolumeLevelScalar()
        new = max(0.0, min(1.0, cur + (delta / 100.0)))
        vol.SetMasterVolumeLevelScalar(new, None)
        return f"Volume {'up' if delta>0 else 'down'}"

    def mute(self, args: Dict[str, Any]):
        vol = self._volume_endpoint()
        if vol is None:
            return "Volume control not available"
        muted = args.get("muted")
        if muted is None:
            vol.SetMute(not vol.GetMute(), None)
        else:
            vol.SetMute(bool(muted), None)
        return "Mute toggled"

    def press(self, args: Dict[str, Any]):
        if pyautogui:
            pyautogui.press(args.get("key", "enter"))
        return "Key pressed"

    def type(self, args: Dict[str, Any]):
        if pyautogui:
            pyautogui.typewrite(args.get("text", ""))
        return "Typed"

    def mouse_move(self, args: Dict[str, Any]):
        if pyautogui:
            dx = int(args.get("dx", 0))
            dy = int(args.get("dy", 0))
            pyautogui.moveRel(dx, dy, duration=0.0)
        return "Mouse moved"

    def click(self, args: Dict[str, Any]):
        if pyautogui:
            pyautogui.click()
        return "Clicked"

    def scroll(self, args: Dict[str, Any]):
        if pyautogui:
            amount = int(args.get("amount", -500))
            pyautogui.scroll(amount)
        return "Scrolled"

    def screenshot(self, args: Dict[str, Any]):
        if pyautogui:
            img = pyautogui.screenshot()
            path = args.get("path", "screenshot.png")
            img.save(path)
            return f"Screenshot saved to {path}"
        return "Screenshot not available"

    def lock(self, args: Dict[str, Any]):
        os.system("rundll32.exe user32.dll,LockWorkStation")
        return "Locking workstation"

    def search(self, args: Dict[str, Any]):
        query = args.get("query")
        if not query:
            return "Missing search query"
        os.system(f'start "" "https://www.bing.com/search?q={query}"')
        return f"Searching for {query}"

    def google(self, args: Dict[str, Any]):
        query = args.get("query")
        if not query:
            return "Missing search query"
        os.system(f'start "" "https://www.google.com/search?q={query}"')
        return f"Googling {query}"

    def windows_tile(self, args: Dict[str, Any]):
        layout = args.get("layout", "left")
        if pyautogui:
            if layout == "left":
                pyautogui.hotkey("win", "left")
            elif layout == "right":
                pyautogui.hotkey("win", "right")
            elif layout == "up":
                pyautogui.hotkey("win", "up")
            elif layout == "down":
                pyautogui.hotkey("win", "down")
        return f"Tiled window {layout}"

    def system_status(self, args: Dict[str, Any]):
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            battery = psutil.sensors_battery()
            status = f"CPU: {cpu}%, RAM: {memory.percent}%"
            if battery:
                status += f", Battery: {battery.percent}%"
            return status
        except ImportError:
            return "System monitoring not available (psutil not installed)"
        except Exception as e:
            return f"System status unavailable: {e}"

    def calendar(self, args: Dict[str, Any]):
        action = args.get("action", "today")
        if action == "today":
            import datetime
            return f"Today is {datetime.date.today().strftime('%A, %B %d, %Y')}"
        elif action == "next_event":
            return "No calendar integration configured"
        return "Calendar action not supported"

    def routine(self, args: Dict[str, Any]):
        if not self.routine_manager:
            return "Routine system not available"
        routine_name = args.get("name", "")
        routine_args = args.get("args", {})
        if not routine_name:
            available = self.routine_manager.list_routines()
            return f"Available routines: {', '.join(available)}"
        return self.routine_manager.execute_routine(routine_name, routine_args)

    def briefing(self, args: Dict[str, Any]):
        if not self.briefing_system:
            return "Briefing system not available"
        briefing_type = args.get("type", "status")
        return self.briefing_system.generate_on_demand_briefing(briefing_type)

    def enable_briefings(self, args: Dict[str, Any]):
        if not self.briefing_system:
            return "Briefing system not available"
        enabled = args.get("enabled", True)
        self.briefing_system.enabled = enabled
        if enabled:
            self.briefing_system.start_scheduler()
            return "Automatic briefings enabled"
        else:
            self.briefing_system.stop_scheduler()
            return "Automatic briefings disabled"

    # ------------------------------------------------------------------
    # Descriptor construction
    # ------------------------------------------------------------------
    def _build_descriptors(self):
        def reg(name: str, func: Callable[[Dict[str, Any]], Any], description: str, category: str = "system", schema: Dict[str, Dict[str, Any]] = None, aliases: List[str] = None, requires: List[str] = None, returns: str = "string"):
            desc = ActionDescriptor(name=name, func=func, description=description, category=category, args_schema=schema or {}, aliases=aliases or [], requires=requires or [], returns=returns)
            self._descriptors[name] = desc
            for al in desc.aliases:
                self._descriptors[al] = desc
        # Register core actions
        reg("open", self.open, "Open an application or file", schema={"app": _DEF("string", True, desc="Application path or name")})
        reg("close", self.close, "Close an application by image name", schema={"app": _DEF("string", True)})
        reg("playpause", self.playpause, "Toggle media playback", category="media")
        reg("next", self.next, "Next media track", category="media")
        reg("previous", self.previous, "Previous media track", category="media")
        reg("set_volume", self.set_volume, "Set absolute volume level", category="media", schema={"level": _DEF("int", True, 50)})
        reg("volume", self.volume, "Relative volume adjust", category="media", schema={"delta": _DEF("int", False, 10)})
        reg("mute", self.mute, "Toggle or set mute state", category="media", schema={"muted": _DEF("bool", False)})
        reg("press", self.press, "Press a keyboard key", category="input", schema={"key": _DEF("string", False, "enter")})
        reg("type", self.type, "Type text", category="input", schema={"text": _DEF("string", False, "")})
        reg("mouse_move", self.mouse_move, "Move mouse relatively", category="input", schema={"dx": _DEF("int", False, 0), "dy": _DEF("int", False, 0)})
        reg("click", self.click, "Mouse click", category="input")
        reg("scroll", self.scroll, "Mouse scroll", category="input", schema={"amount": _DEF("int", False, -500)})
        reg("screenshot", self.screenshot, "Take a screenshot", category="system", schema={"path": _DEF("string", False, "screenshot.png")})
        reg("lock", self.lock, "Lock workstation", category="system")
        reg("search", self.search, "Web search via Bing", category="web", schema={"query": _DEF("string", True)})
        reg("google", self.google, "Web search via Google", category="web", schema={"query": _DEF("string", True)})
        reg("windows_tile", self.windows_tile, "Snap window position", category="system", schema={"layout": _DEF("string", False, "left")})
        reg("system_status", self.system_status, "Get system resource summary", category="system")
        reg("calendar", self.calendar, "Calendar query (stub)", category="productivity", schema={"action": _DEF("string", False, "today")})
        reg("routine", self.routine, "Execute or list routines", category="automation", schema={"name": _DEF("string", False), "args": _DEF("object", False, {})})
        reg("briefing", self.briefing, "Generate status briefing", category="automation", schema={"type": _DEF("string", False, "status")})
        reg("enable_briefings", self.enable_briefings, "Enable/disable auto briefings", category="automation", schema={"enabled": _DEF("bool", False, True)})

    # ------------------------------------------------------------------
    # Public descriptor APIs
    # ------------------------------------------------------------------
    def list_actions(self, *, include_aliases: bool = False) -> List[str]:
        if include_aliases:
            return sorted(self._descriptors.keys())
        # Filter out alias duplicates: only keep canonical names (no duplicates by comparing object id)
        seen = set()
        names = []
        for name, desc in self._descriptors.items():
            if id(desc) not in seen:
                seen.add(id(desc))
                names.append(name)
        return sorted(names)

    def get_descriptor(self, name: str) -> Optional[ActionDescriptor]:
        return self._descriptors.get(name)

    # Backward-compatible dispatch
    def dispatch(self, action: str, args: Dict[str, Any]):
        # Prefer descriptor, fallback to original attribute lookup
        desc = self._descriptors.get(action)
        if desc:
            return desc.invoke(args)
        fn = getattr(self, action, None)
        if not fn:
            return f"Unknown action: {action}"
        return fn(args or {})

