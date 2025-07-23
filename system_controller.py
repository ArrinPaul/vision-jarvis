import os
import subprocess
import pyautogui
import psutil
import webbrowser
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

class SystemController:
    def __init__(self):
        self.volume_controller = self.setup_volume_control()
    
    def setup_volume_control(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, 
            CLSCTX_ALL, 
            None
        )
        return cast(interface, POINTER(IAudioEndpointVolume))
    
    def open_application(self, app_name):
        apps = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "explorer": "explorer.exe",
            "browser": "start chrome",
            "word": "winword.exe",
            "excel": "excel.exe",
            "powerpoint": "powerpnt.exe",
            "paint": "mspaint.exe",
            "settings": "start ms-settings:",
            "task manager": "taskmgr.exe"
        }
        
        if app_name in apps:
            os.system(apps[app_name])
            return f"Opening {app_name}"
        
        # Try to find application in system path
        try:
            subprocess.Popen(app_name)
            return f"Opening {app_name}"
        except:
            return f"Couldn't find {app_name}"

    def close_application(self, app_name):
        for proc in psutil.process_iter(['name']):
            if proc.info['name'].lower() == app_name.lower():
                proc.kill()
                return f"Closed {app_name}"
        return f"{app_name} not found"

    def volume_up(self):
        current_vol = self.volume_controller.GetMasterVolumeLevelScalar()
        new_vol = min(1.0, current_vol + 0.1)
        self.volume_controller.SetMasterVolumeLevelScalar(new_vol, None)
        return f"Volume increased to {int(new_vol*100)}%"

    def volume_down(self):
        current_vol = self.volume_controller.GetMasterVolumeLevelScalar()
        new_vol = max(0.0, current_vol - 0.1)
        self.volume_controller.SetMasterVolumeLevelScalar(new_vol, None)
        return f"Volume decreased to {int(new_vol*100)}%"

    def mute(self):
        is_muted = self.volume_controller.GetMute()
        self.volume_controller.SetMute(not is_muted, None)
        return "Muted" if not is_muted else "Unmuted"

    def set_volume(self, level):
        vol = max(0.0, min(1.0, float(level)/100))
        self.volume_controller.SetMasterVolumeLevelScalar(vol, None)
        return f"Volume set to {level}%"

    def press_key(self, key_name):
        keys = {
            "enter": "enter",
            "space": "space",
            "tab": "tab",
            "escape": "esc",
            "backspace": "backspace"
        }
        if key_name in keys:
            pyautogui.press(keys[key_name])
            return f"Pressed {key_name}"
        return "Key not recognized"

    def hotkey(self, *keys):
        try:
            pyautogui.hotkey(*keys)
            return f"Pressed {'+'.join(keys)}"
        except:
            return "Could not execute hotkey"

    def type_text(self, text):
        pyautogui.write(text)
        return f"Typed: {text}"

    def mouse_move(self, direction, distance=100):
        directions = {
            "up": (0, -distance),
            "down": (0, distance),
            "left": (-distance, 0),
            "right": (distance, 0)
        }
        if direction in directions:
            dx, dy = directions[direction]
            pyautogui.move(dx, dy)
            return f"Moved mouse {direction}"
        return "Invalid direction"

    def click(self, button="left"):
        pyautogui.click(button=button)
        return f"{button} clicked"

    def scroll(self, direction="up", amount=3):
        amount = amount if direction == "up" else -amount
        pyautogui.scroll(amount)
        return f"Scrolled {direction}"

    def take_screenshot(self):
        screenshot = pyautogui.screenshot()
        screenshot.save("screenshot.png")
        return "Screenshot saved as screenshot.png"

    def lock_pc(self):
        os.system("rundll32.exe user32.dll,LockWorkStation")
        return "PC locked"

    def shutdown(self):
        os.system("shutdown /s /t 0")
        return "Shutting down"

    def restart(self):
        os.system("shutdown /r /t 0")
        return "Restarting"