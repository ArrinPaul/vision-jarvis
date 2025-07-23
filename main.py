import cv2
import time
import numpy as np
from hand_tracker import HandTracker
from voice_assistant import VoiceAssistant
from camera_module import CameraModule
from canvas_module import CanvasModule
from utils import load_icon
import sys

def overlay_image(background, overlay, x, y):
    """Overlay an RGBA image on a BGR background at position (x, y)."""
    if overlay is None:
        return background
    h, w = overlay.shape[:2]
    if overlay.shape[2] == 4:
        alpha_overlay = overlay[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay
        for c in range(0, 3):
            background[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:, :, c] +
                                           alpha_background * background[y:y+h, x:x+w, c])
    else:
        background[y:y+h, x:x+w] = overlay
    return background

# Constants
WIDTH, HEIGHT = 1280, 720
HOVER_TIME = 2  # seconds to hover for activation


class MainApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)
        self.cap.set(3, WIDTH)
        self.cap.set(4, HEIGHT)
        self.tracker = HandTracker()
        self.active_module = None
        self.hover_start = None
        self.last_hand_time = time.time()
        self.last_detection = time.time()
        self.detection_freq = 0.5  # Detection frequency when idle (seconds)
        self.mic_icon = load_icon("mic_icon.png")
        self.camera_icon = load_icon("camera_icon.png")
        self.paint_icon = load_icon("paint_icon.png")
        self.icons = [
            {"name": "Voice Assistant", "pos": (WIDTH//4 - 60, 100), "icon": self.mic_icon},
            {"name": "Camera", "pos": (WIDTH//2 - 60, 100), "icon": self.camera_icon},
            {"name": "Canvas", "pos": (3*WIDTH//4 - 60, 100), "icon": self.paint_icon}
        ]
        self.camera_module = CameraModule()
        self.canvas_module = CanvasModule()
        self.voice_assistant = VoiceAssistant()
        self.running = True  # <-- Add this line

    def check_hover(self, x, y):
        for icon in self.icons:
            ix, iy = icon["pos"]
            if ix <= x <= ix + 120 and iy <= y <= iy + 120:
                return icon["name"]
        return None
    
    def draw_main_ui(self, img):
        # Draw background
        cv2.rectangle(img, (0, 0), (WIDTH, 80), (50, 50, 50), -1)
        cv2.putText(img, "Gesture Control System", (WIDTH//2 - 180, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw icons
        for icon in self.icons:
            x, y = icon["pos"]
            img = overlay_image(img, icon["icon"], x, y)
            
            # Draw label
            cv2.putText(img, icon["name"], (x, y + 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw hover timer if active
        if self.hover_start:
            elapsed = time.time() - self.hover_start
            progress = min(1.0, elapsed / HOVER_TIME)
            cv2.rectangle(img, (WIDTH//2 - 100, HEIGHT - 50), 
                         (WIDTH//2 - 100 + int(200 * progress), HEIGHT - 30), 
                         (0, 255, 0), -1)
            cv2.rectangle(img, (WIDTH//2 - 100, HEIGHT - 50), 
                         (WIDTH//2 + 100, HEIGHT - 30), (255, 255, 255), 2)
        
        return img
    
    def run(self):
        try:
            while self.running:
                current_time = time.time()  # Move this to the top
                success, img = self.cap.read()
                if not success:
                    continue

                img = cv2.flip(img, 1)

                # Always run hand detection for best tracking
                img, results, fps = self.tracker.find_hands(img)
                lm_list, hand_info = self.tracker.find_position(img, results)

                # Convert to legacy format for compatibility with existing modules
                legacy_lm_list = self.tracker.get_legacy_landmark_list(lm_list)

                if self.active_module:
                    # Run active module
                    if self.active_module == "Voice Assistant":
                        img, should_exit = self.voice_assistant.run(img, legacy_lm_list)
                    elif self.active_module == "Camera":
                        img, should_exit = self.camera_module.run(img, legacy_lm_list)
                    elif self.active_module == "Canvas":
                        img, should_exit = self.canvas_module.run(img, legacy_lm_list)

                    if should_exit:
                        self.active_module = None
                        # Reset modules
                        self.camera_module = CameraModule()
                        self.canvas_module = CanvasModule()
                        self.voice_assistant = VoiceAssistant()
                else:
                    # Main interface
                    img = self.draw_main_ui(img)

                    if legacy_lm_list:
                        self.last_hand_time = current_time
                        index_tip = legacy_lm_list[8]  # Index finger tip
                        x, y = index_tip[1], index_tip[2]

                        # Draw cursor
                        cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)

                        # Check hover
                        hovered = self.check_hover(x, y)
                        if hovered:
                            if self.hover_start is None:
                                self.hover_start = time.time()
                            elif time.time() - self.hover_start >= HOVER_TIME:
                                self.active_module = hovered
                                self.hover_start = None
                        else:
                            self.hover_start = None
                    else:
                        self.hover_start = None

                # Show FPS
                delta = time.time() - current_time
                fps = 1 / delta if delta > 0 else 0
                cv2.putText(
                    img,
                    f"FPS: {int(fps)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Gesture Control", img)

                # Check for window close event and 'q' key
                key = cv2.waitKey(1) & 0xFF
                if (
                    key == ord("q")
                    or cv2.getWindowProperty("Gesture Control", cv2.WND_PROP_VISIBLE)
                    < 1
                ):
                    break

        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources and close windows."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = MainApp()
    app.run()