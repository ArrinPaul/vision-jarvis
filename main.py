import cv2
import time
import numpy as np
import gc  # For garbage collection
from hand_tracker import HandTracker
from voice_assistant import VoiceAssistant
from camera_module import CameraModule
from canvas_module import CanvasModule
from utils import load_icon
import sys
import json
from vision_tools import create_vision_manager_safe


def overlay_image(background, overlay, x, y):
    """Overlay an RGBA image on a BGR background at position (x, y)."""
    if overlay is None:
        return background
    h, w = overlay.shape[:2]
    if overlay.shape[2] == 4:
        alpha_overlay = overlay[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay
        for c in range(0, 3):
            background[y : y + h, x : x + w, c] = (
                alpha_overlay * overlay[:, :, c]
                + alpha_background * background[y : y + h, x : x + w, c]
            )
    else:
        background[y : y + h, x : x + w] = overlay
    return background


# Constants
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 720
HOVER_TIME = 2  # seconds to hover for activation


class MainApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)

        # Initialize with default size
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)

        # Create window with resizable property
        cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gesture Control", self.width, self.height)

        # Enhanced hand tracker with better settings for UI interaction
        self.tracker = HandTracker(
            max_hands=1,  # Single hand for better performance
            detection_con=0.8,  # Higher detection confidence
            track_con=0.7,  # Higher tracking confidence
            static_image_mode=False,
        )

        self.active_module = None
        self.hover_start = None
        self.last_hand_time = time.time()
        self.last_detection = time.time()
        self.detection_freq = 0.5  # Detection frequency when idle (seconds)

        # Enhanced gesture tracking
        self.gesture_history = []
        self.stable_gesture = None
        self.gesture_confidence_threshold = 0.7
        self.gesture_stability_frames = 3

        # Cursor smoothing for better UI interaction
        self.cursor_history = []
        self.cursor_smooth_factor = 0.3
        self.last_cursor_pos = None

        # Click detection
        self.click_cooldown = 0
        self.click_cooldown_time = 0.5  # Seconds between clicks
        self.pinch_threshold = 40
        self.is_pinching = False
        self.pinch_start_time = None

        # Coordinate scaling factors - initialize with defaults
        self.x_scale = 1.0
        self.y_scale = 1.0
        self.overall_scale = 1.0

        # Camera resolution tracking
        self.camera_width = self.width
        self.camera_height = self.height

        # Debug mode for coordinate tracking
        self.debug_mode = False  # Set to True to show coordinate debug info

        self.mic_icon = load_icon("mic_icon.png")
        self.camera_icon = load_icon("camera_icon.png")
        self.paint_icon = load_icon("paint_icon.png")

        # Initialize icons with dynamic positioning
        self.update_icon_positions()

        # Initialize scaling factors
        self.get_window_size()

        # Initialize modules with error handling
        self.camera_module = None
        self.canvas_module = None
        self.voice_assistant = None
        self.vision_manager = None
        
        # Initialize camera module
        try:
            self.camera_module = CameraModule()
            print("✓ Camera module initialized successfully")
        except Exception as e:
            print(f"✗ Camera module initialization failed: {e}")
            
        # Initialize canvas module
        try:
            self.canvas_module = CanvasModule()
            print("✓ Canvas module initialized successfully")
        except Exception as e:
            print(f"✗ Canvas module initialization failed: {e}")
            
        # Initialize voice assistant
        try:
            self.voice_assistant = VoiceAssistant("voice_config.json")
            print("✓ Voice assistant initialized successfully")
        except Exception as e:
            print(f"✗ Voice assistant initialization failed: {e}")
            print("Voice features will not be available")
            
        # Initialize vision manager for object detection (optional)
        try:
            vm_config = {
                "features": {"vision_object_detection": True},
                "vision_model_path": "models/yolov8n.onnx"
            }
            self.vision_manager = create_vision_manager_safe(vm_config)
            print("✓ Vision manager initialized successfully")
        except Exception as e:
            print(f"✗ Vision manager initialization failed: {e}")
            print("Vision detection features will not be available")
        self.running = True

    def update_icon_positions(self):
        """Update icon positions based on current window size"""
        self.icons = [
            {
                "name": "Voice Assistant",
                "pos": (self.width // 4 - 60, 100),
                "icon": self.mic_icon,
            },
            {
                "name": "Camera",
                "pos": (self.width // 2 - 60, 100),
                "icon": self.camera_icon,
            },
            {
                "name": "Canvas",
                "pos": (3 * self.width // 4 - 60, 100),
                "icon": self.paint_icon,
            },
        ]

    def get_window_size(self):
        """Get current window size and update scaling factors"""
        try:
            # Get the current window size
            window_rect = cv2.getWindowImageRect("Gesture Control")
            if window_rect[2] > 0 and window_rect[3] > 0:
                new_width, new_height = window_rect[2], window_rect[3]
                if new_width != self.width or new_height != self.height:
                    self.width = new_width
                    self.height = new_height
                    self.update_icon_positions()

                    # Update scaling factors for coordinate transformation
                    self.x_scale = self.width / DEFAULT_WIDTH
                    self.y_scale = self.height / DEFAULT_HEIGHT
                    self.overall_scale = min(self.x_scale, self.y_scale)

                    # Update gesture detection thresholds based on screen size
                    self.update_gesture_thresholds()
        except:
            # Fallback to default if window size detection fails
            self.x_scale = 1.0
            self.y_scale = 1.0
            self.overall_scale = 1.0

    def update_gesture_thresholds(self):
        """Update gesture detection thresholds based on current screen size"""
        # Scale pinch threshold based on screen size
        base_pinch_threshold = 40
        self.pinch_threshold = int(base_pinch_threshold * self.overall_scale)

        # Scale click cooldown based on screen size (larger screens need more time)
        base_cooldown = 0.5
        self.click_cooldown_time = base_cooldown * (1 + (self.overall_scale - 1) * 0.5)

        # Scale cursor smoothing factor (less smoothing for larger screens)
        base_smooth_factor = 0.3
        self.cursor_smooth_factor = max(0.1, base_smooth_factor / self.overall_scale)

    def check_hover(self, x, y):
        """Enhanced hover detection with better precision and scaling"""
        for icon in self.icons:
            ix, iy = icon["pos"]
            # Dynamic detection margin based on screen size
            detection_margin = int(20 * self.overall_scale)
            icon_size = int(120 * self.overall_scale)

            if (
                ix - detection_margin <= x <= ix + icon_size + detection_margin
                and iy - detection_margin <= y <= iy + icon_size + detection_margin
            ):
                return icon["name"]
        return None

    def transform_coordinates(self, norm_x, norm_y):
        """
        Transform MediaPipe normalized coordinates (0-1) to screen coordinates.
        MediaPipe returns normalized coordinates regardless of camera resolution.
        """
        if norm_x is None or norm_y is None:
            return None, None

        # MediaPipe coordinates are normalized (0-1), clamp to ensure valid range
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))

        # Transform to screen coordinates
        screen_x = int(norm_x * self.width)
        screen_y = int(norm_y * self.height)

        if self.debug_mode:
            print(
                f"Transform: norm({norm_x:.3f}, {norm_y:.3f}) -> screen({screen_x}, {screen_y})"
            )

        return screen_x, screen_y

    def check_pinch_click(self):
        """Process click gestures using the new hand tracker's pinch detection"""
        current_time = time.time()

        # Check click cooldown
        if current_time < self.click_cooldown:
            return False

        # Use the hand tracker's pinch detection
        is_pinching, distance = self.tracker.detect_pinch()

        # Detect pinch start
        if is_pinching and not self.is_pinching:
            self.is_pinching = True
            self.pinch_start_time = current_time
            return False

        # Detect pinch release (click)
        elif not is_pinching and self.is_pinching:
            self.is_pinching = False
            if (
                self.pinch_start_time and (current_time - self.pinch_start_time) > 0.1
            ):  # Minimum pinch duration
                self.click_cooldown = current_time + self.click_cooldown_time
                return True

        return False

    def smooth_cursor_position(self, x, y):
        """Apply smoothing to cursor position for better stability with adaptive smoothing"""
        if self.last_cursor_pos is None:
            self.last_cursor_pos = (x, y)
            return x, y

        # Apply exponential smoothing with dynamic factor
        smoothed_x = int(
            self.cursor_smooth_factor * x
            + (1 - self.cursor_smooth_factor) * self.last_cursor_pos[0]
        )
        smoothed_y = int(
            self.cursor_smooth_factor * y
            + (1 - self.cursor_smooth_factor) * self.last_cursor_pos[1]
        )

        self.last_cursor_pos = (smoothed_x, smoothed_y)
        return smoothed_x, smoothed_y

    def detect_stable_gesture(self, gesture, confidence):
        """Detect stable gestures over multiple frames"""
        current_time = time.time()

        # Add current gesture to history
        self.gesture_history.append(
            {"gesture": gesture, "confidence": confidence, "time": current_time}
        )

        # Remove old entries (keep only last 2 seconds)
        self.gesture_history = [
            g for g in self.gesture_history if current_time - g["time"] < 2.0
        ]

        # Check for stable gesture
        if len(self.gesture_history) >= self.gesture_stability_frames:
            recent_gestures = self.gesture_history[-self.gesture_stability_frames :]

            # Check if all recent gestures are the same and confident
            if all(
                g["gesture"] == gesture
                and g["confidence"] >= self.gesture_confidence_threshold
                for g in recent_gestures
            ):
                self.stable_gesture = gesture
                return True

        return False

    def draw_main_ui(self, img):
        # Update window size if changed
        self.get_window_size()

        # Resize image if window size changed
        img = cv2.resize(img, (self.width, self.height))

        # Draw modern gradient background header
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, 120), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        # Add subtle gradient effect
        for i in range(120):
            alpha = 1.0 - (i / 120) * 0.3
            color = (int(60 * alpha), int(60 * alpha), int(60 * alpha))
            cv2.line(img, (0, i), (self.width, i), color, 1)

        # Draw main title with better styling
        title_text = "AI Gesture Control System"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.2
        thickness = 2

        # Get text size for centering
        text_size = cv2.getTextSize(title_text, font, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = 50

        # Draw shadow effect
        cv2.putText(
            img,
            title_text,
            (text_x + 2, text_y + 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 1,
        )
        # Draw main text
        cv2.putText(
            img,
            title_text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        # Draw subtitle
        subtitle = "Point and hover to activate modules"
        subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        subtitle_x = (self.width - subtitle_size[0]) // 2
        cv2.putText(
            img,
            subtitle,
            (subtitle_x, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        # Draw icons with modern styling
        for i, icon in enumerate(self.icons):
            x, y = icon["pos"]

            # Draw hover effect background
            if self.hover_start:
                hovered_icon = None
                # This will be set in the main loop when we detect hover
                cv2.circle(img, (x + 60, y + 60), 80, (0, 255, 255), 3)

            # Draw icon background circle
            cv2.circle(img, (x + 60, y + 60), 70, (80, 80, 80), -1)
            cv2.circle(img, (x + 60, y + 60), 70, (120, 120, 120), 2)

            # Overlay icon
            img = overlay_image(img, icon["icon"], x, y)

            # Draw modern label with background
            label_font = cv2.FONT_HERSHEY_SIMPLEX
            label_scale = 0.8
            label_thickness = 2
            label_size = cv2.getTextSize(
                icon["name"], label_font, label_scale, label_thickness
            )[0]
            label_x = x + (120 - label_size[0]) // 2
            label_y = y + 180

            # Label background
            cv2.rectangle(
                img,
                (label_x - 10, label_y - 25),
                (label_x + label_size[0] + 10, label_y + 5),
                (50, 50, 50),
                -1,
            )
            cv2.rectangle(
                img,
                (label_x - 10, label_y - 25),
                (label_x + label_size[0] + 10, label_y + 5),
                (100, 100, 100),
                1,
            )

            # Draw label
            cv2.putText(
                img,
                icon["name"],
                (label_x, label_y),
                label_font,
                label_scale,
                (255, 255, 255),
                label_thickness,
            )

        # Draw modern hover progress bar
        if self.hover_start:
            elapsed = time.time() - self.hover_start
            progress = min(1.0, elapsed / HOVER_TIME)

            # Progress bar background
            bar_width = 300
            bar_height = 20
            bar_x = (self.width - bar_width) // 2
            bar_y = self.height - 80

            # Background
            cv2.rectangle(
                img,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (60, 60, 60),
                -1,
            )
            cv2.rectangle(
                img,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (120, 120, 120),
                2,
            )

            # Progress fill with gradient
            progress_width = int(bar_width * progress)
            if progress_width > 0:
                # Create gradient effect
                for i in range(progress_width):
                    alpha = i / progress_width
                    color = (
                        int(0 * (1 - alpha) + 100 * alpha),
                        int(255 * (1 - alpha) + 255 * alpha),
                        int(0 * (1 - alpha) + 50 * alpha),
                    )
                    cv2.line(
                        img,
                        (bar_x + i, bar_y),
                        (bar_x + i, bar_y + bar_height),
                        color,
                        1,
                    )

            # Progress text
            progress_text = f"Activating... {int(progress * 100)}%"
            progress_text_size = cv2.getTextSize(
                progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )[0]
            progress_text_x = (self.width - progress_text_size[0]) // 2
            cv2.putText(
                img,
                progress_text,
                (progress_text_x, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        return img

    def draw_enhanced_cursor(self, img, x, y, gesture, confidence, hand_info):
        """Draw an enhanced cursor with gesture feedback and dynamic sizing"""
        # Ensure coordinates are integers
        x, y = int(x), int(y)

        # Dynamic cursor size based on screen scale
        base_cursor_size = 12
        cursor_size = int(base_cursor_size * self.overall_scale)

        # Color based on gesture confidence
        if confidence > 0.8:
            cursor_color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.5:
            cursor_color = (0, 255, 255)  # Yellow for medium confidence
        else:
            cursor_color = (0, 0, 255)  # Red for low confidence

        # Draw cursor layers with dynamic sizing
        outline_size = cursor_size + int(5 * self.overall_scale)
        cv2.circle(
            img, (x, y), outline_size, (0, 0, 0), max(1, int(2 * self.overall_scale))
        )  # Black outline
        cv2.circle(
            img, (x, y), cursor_size, cursor_color, max(1, int(3 * self.overall_scale))
        )  # Main cursor
        cv2.circle(
            img,
            (x, y),
            max(1, cursor_size - int(5 * self.overall_scale)),
            (255, 255, 255),
            -1,
        )  # White center
        cv2.circle(
            img,
            (x, y),
            max(1, cursor_size - int(8 * self.overall_scale)),
            cursor_color,
            -1,
        )  # Colored center

        # Draw gesture-specific indicators with scaling
        if gesture == "pinch" and confidence > 0.7:
            # Draw pinch indicator
            indicator_radius = cursor_size + int(15 * self.overall_scale)
            cv2.circle(
                img,
                (x, y),
                indicator_radius,
                (255, 0, 255),
                max(1, int(2 * self.overall_scale)),
            )
            font_scale = 0.5 * self.overall_scale
            cv2.putText(
                img,
                "PINCH",
                (x - int(30 * self.overall_scale), y - int(25 * self.overall_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 0, 255),
                max(1, int(2 * self.overall_scale)),
            )
        elif gesture == "point" and confidence > 0.7:
            # Draw pointing indicator
            arrow_length = int(30 * self.overall_scale)
            cv2.arrowedLine(
                img,
                (x, y),
                (x + arrow_length, y - arrow_length),
                (0, 255, 255),
                max(1, int(3 * self.overall_scale)),
            )

        # Show hand confidence with scaled text
        if hand_info and hand_info.get("confidence", 0) > 0:
            conf_text = f"{hand_info['confidence']:.2f}"
            font_scale = 0.4 * self.overall_scale
            text_offset_x = int(20 * self.overall_scale)
            text_offset_y = int(20 * self.overall_scale)
            cv2.putText(
                img,
                conf_text,
                (x + text_offset_x, y + text_offset_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                max(1, int(1 * self.overall_scale)),
            )

    def draw_gesture_info(self, img, gesture, confidence, hand_info):
        """Draw gesture information panel with dynamic scaling"""
        # Scale panel size based on screen size
        base_panel_width, base_panel_height = 250, 140
        panel_width = int(base_panel_width * self.overall_scale)
        panel_height = int(base_panel_height * self.overall_scale)

        panel_x = int(10 * self.overall_scale)
        panel_y = self.height - panel_height - int(10 * self.overall_scale)

        # Draw info panel background
        cv2.rectangle(
            img,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (30, 30, 30),
            -1,
        )
        cv2.rectangle(
            img,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (100, 100, 100),
            max(1, int(2 * self.overall_scale)),
        )

        # Dynamic font scaling
        base_font_scale = 0.6
        title_font_scale = base_font_scale * self.overall_scale
        info_font_scale = 0.5 * self.overall_scale
        thickness = max(1, int(2 * self.overall_scale))
        info_thickness = max(1, int(1 * self.overall_scale))

        # Title
        title_y = panel_y + int(20 * self.overall_scale)
        cv2.putText(
            img,
            "Hand Tracking Info",
            (panel_x + int(10 * self.overall_scale), title_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            title_font_scale,
            (255, 255, 255),
            thickness,
        )

        # Gesture info
        line_height = int(20 * self.overall_scale)
        current_y = title_y + line_height

        gesture_text = f"Gesture: {gesture}"
        cv2.putText(
            img,
            gesture_text,
            (panel_x + int(10 * self.overall_scale), current_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            info_font_scale,
            (0, 255, 255),
            info_thickness,
        )

        current_y += line_height
        confidence_text = f"Confidence: {confidence:.2f}"
        cv2.putText(
            img,
            confidence_text,
            (panel_x + int(10 * self.overall_scale), current_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            info_font_scale,
            (0, 255, 255),
            info_thickness,
        )

        # Hand info
        if hand_info and hand_info.get("detected", False):
            handedness = hand_info.get("handedness", "Unknown")
            hand_conf = hand_info.get("confidence", 0)
            hand_size = hand_info.get("hand_size", 0)

            current_y += line_height
            handedness_text = f"Hand: {handedness}"
            cv2.putText(
                img,
                handedness_text,
                (panel_x + int(10 * self.overall_scale), current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                info_font_scale,
                (255, 255, 0),
                info_thickness,
            )

            current_y += line_height
            hand_conf_text = f"Hand Conf: {hand_conf:.2f}"
            cv2.putText(
                img,
                hand_conf_text,
                (panel_x + int(10 * self.overall_scale), current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                info_font_scale,
                (255, 255, 0),
                info_thickness,
            )

            current_y += line_height
            hand_size_text = f"Hand Size: {hand_size:.3f}"
            cv2.putText(
                img,
                hand_size_text,
                (panel_x + int(10 * self.overall_scale), current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                info_font_scale,
                (255, 255, 0),
                info_thickness,
            )

        # Pinch status indicator with scaling
        if self.is_pinching:
            indicator_radius = int(8 * self.overall_scale)
            indicator_x = panel_x + panel_width - int(20 * self.overall_scale)
            indicator_y = panel_y + int(20 * self.overall_scale)

            cv2.circle(
                img, (indicator_x, indicator_y), indicator_radius, (0, 255, 0), -1
            )
            cv2.putText(
                img,
                "PINCH",
                (
                    panel_x + panel_width - int(60 * self.overall_scale),
                    panel_y + int(25 * self.overall_scale),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4 * self.overall_scale,
                (0, 255, 0),
                info_thickness,
            )

    def draw_debug_info(self, img, norm_x, norm_y, screen_x, screen_y):
        """Draw debug information for coordinate transformation"""
        debug_panel_x = self.width - int(300 * self.overall_scale)
        debug_panel_y = int(10 * self.overall_scale)
        debug_panel_width = int(290 * self.overall_scale)
        debug_panel_height = int(150 * self.overall_scale)  # Increased height

        # Debug panel background
        cv2.rectangle(
            img,
            (debug_panel_x, debug_panel_y),
            (debug_panel_x + debug_panel_width, debug_panel_y + debug_panel_height),
            (40, 40, 40),
            -1,
        )
        cv2.rectangle(
            img,
            (debug_panel_x, debug_panel_y),
            (debug_panel_x + debug_panel_width, debug_panel_y + debug_panel_height),
            (100, 100, 100),
            max(1, int(1 * self.overall_scale)),
        )

        font_scale = 0.4 * self.overall_scale
        thickness = max(1, int(1 * self.overall_scale))
        line_height = int(15 * self.overall_scale)
        start_y = debug_panel_y + int(15 * self.overall_scale)

        # Debug information
        debug_texts = [
            f"Normalized: {norm_x:.3f}, {norm_y:.3f}",
            f"Screen: {screen_x}, {screen_y}",
            f"Hand Detected: {self.tracker.hand_detected}",
            f"Handedness: {self.tracker.current_handedness}",
            f"Win Size: {self.width}x{self.height}",
            f"Scale: {self.overall_scale:.2f}",
            f"FPS: {self.tracker.get_fps():.1f}",
            f"Pinching: {self.is_pinching}",
        ]

        for i, text in enumerate(debug_texts):
            cv2.putText(
                img,
                text,
                (
                    debug_panel_x + int(5 * self.overall_scale),
                    start_y + i * line_height,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

    def draw_landmarks_on_display(self, img):
        """Draw hand landmarks on the resized display image"""
        if not self.tracker.hand_detected or not self.tracker.current_landmarks:
            return img

        h, w, _ = img.shape

        # Draw landmark connections
        connections = [
            # Thumb
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            # Index finger
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            # Middle finger
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            # Ring finger
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            # Pinky
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            # Palm
            (0, 5),
            (5, 9),
            (9, 13),
            (13, 17),
        ]

        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(self.tracker.current_landmarks) and end_idx < len(
                self.tracker.current_landmarks
            ):
                start_lm = self.tracker.current_landmarks[start_idx]
                end_lm = self.tracker.current_landmarks[end_idx]

                start_x, start_y = int(start_lm[0] * w), int(start_lm[1] * h)
                end_x, end_y = int(end_lm[0] * w), int(end_lm[1] * h)

                cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # Draw landmarks
        for i, lm in enumerate(self.tracker.current_landmarks):
            x, y = int(lm[0] * w), int(lm[1] * h)

            # Color code important landmarks
            if i == self.tracker.INDEX_TIP:
                color = (0, 0, 255)  # Red for index tip
                radius = 8
            elif i == self.tracker.THUMB_TIP:
                color = (255, 0, 0)  # Blue for thumb tip
                radius = 8
            elif i in [
                self.tracker.INDEX_TIP,
                self.tracker.MIDDLE_TIP,
                self.tracker.RING_TIP,
                self.tracker.PINKY_TIP,
            ]:
                color = (0, 255, 255)  # Yellow for fingertips
                radius = 6
            else:
                color = (255, 255, 255)  # White for other landmarks
                radius = 4

            cv2.circle(img, (x, y), radius, color, -1)
            cv2.circle(img, (x, y), radius + 2, (0, 0, 0), 2)  # Black outline

        return img

    def run(self):
        try:
            while self.running:
                current_time = time.time()
                success, img = self.cap.read()
                if not success:
                    continue

                img = cv2.flip(img, 1)

                # Store original camera frame dimensions before any processing
                original_height, original_width = img.shape[:2]
                self.camera_height = original_height
                self.camera_width = original_width

                # Check for window resize and update dimensions
                self.get_window_size()

                # Run hand detection on ORIGINAL camera image (not resized)
                # This ensures landmark coordinates are in normalized space (0-1)
                img_for_detection = img.copy()
                img_for_detection, landmarks, hand_info = self.tracker.find_hands(
                    img_for_detection, draw=False
                )

                # NOW resize image to match window size for display
                img = cv2.resize(img, (self.width, self.height))

                # Convert to legacy format for compatibility with existing modules
                legacy_lm_list = self.tracker.get_legacy_landmark_list()

                # Get gesture information for enhanced interaction
                gesture = "none"
                gesture_confidence = 0.0
                if hand_info["detected"]:
                    gesture, gesture_confidence = self.tracker.get_gesture()

                # Detect stable gestures
                stable_gesture_detected = self.detect_stable_gesture(
                    gesture, gesture_confidence
                )

                if self.active_module:
                    # Run active module with current dimensions
                    if self.active_module == "Voice Assistant":
                        img, should_exit = self.voice_assistant.run(
                            img, legacy_lm_list, (self.width, self.height)
                        )
                    elif self.active_module == "Camera":
                        img, should_exit = self.camera_module.run(img, legacy_lm_list)
                    elif self.active_module == "Canvas":
                        img, should_exit = self.canvas_module.run(img, legacy_lm_list)

                    if should_exit:
                        self.active_module = None
                        # Properly cleanup and reset modules to prevent memory leaks
                        try:
                            print("Exiting module, starting cleanup...")

                            # Clean up existing modules before creating new ones
                            if hasattr(self, "camera_module") and self.camera_module:
                                self.camera_module.cleanup()
                                self.camera_module = None

                            if hasattr(self, "canvas_module") and self.canvas_module:
                                self.canvas_module.cleanup()
                                self.canvas_module = None

                            if (
                                hasattr(self, "voice_assistant")
                                and self.voice_assistant
                            ):
                                self.voice_assistant.cleanup()
                                self.voice_assistant = None

                            # Force garbage collection to free memory immediately
                            gc.collect()

                            # Small delay to ensure cleanup completes
                            time.sleep(0.1)

                            # Create fresh instances
                            self.camera_module = CameraModule()
                            self.canvas_module = CanvasModule()
                            self.voice_assistant = VoiceAssistant("voice_config.json")
                            self.tracker.reset_tracking()  # Reset tracking history

                            print("Modules cleaned up and reset successfully")



                        except Exception as e:
                            print(f"Error during module cleanup: {e}")
                            # Fallback: force garbage collection and create new instances
                            gc.collect()
                            time.sleep(0.1)
                            self.camera_module = CameraModule()
                            self.canvas_module = CanvasModule()
                            self.voice_assistant = VoiceAssistant("voice_config.json")
                            self.tracker.reset_tracking()
                else:
                    # Main interface with enhanced gesture interaction
                    img = self.draw_main_ui(img)

                    # Draw hand landmarks on the display image
                    if hand_info["detected"]:
                        img = self.draw_landmarks_on_display(img)

                    if legacy_lm_list and hand_info["confidence"] > 0.7:
                        self.last_hand_time = current_time

                        # Get index finger tip position (normalized coordinates)
                        index_finger_pos = self.tracker.get_index_finger_tip()

                        if index_finger_pos:
                            # Transform normalized coordinates to screen coordinates
                            x, y = self.transform_coordinates(
                                index_finger_pos[0], index_finger_pos[1]
                            )

                            # Apply cursor smoothing
                            x, y = self.smooth_cursor_position(x, y)

                            # Ensure coordinates are within screen bounds
                            x = max(0, min(self.width - 1, x))
                            y = max(0, min(self.height - 1, y))

                            # Draw enhanced cursor with gesture feedback
                            self.draw_enhanced_cursor(
                                img, x, y, gesture, gesture_confidence, hand_info
                            )

                            # Show gesture information
                            self.draw_gesture_info(
                                img, gesture, gesture_confidence, hand_info
                            )

                            # Debug coordinate information if enabled
                            if self.debug_mode:
                                self.draw_debug_info(
                                    img, index_finger_pos[0], index_finger_pos[1], x, y
                                )

                            # Check for click gesture using new pinch detection
                            clicked = self.check_pinch_click()

                            # Check hover with enhanced detection
                            hovered = self.check_hover(x, y)
                            if hovered:
                                if self.hover_start is None:
                                    self.hover_start = time.time()
                                elif (
                                    time.time() - self.hover_start >= HOVER_TIME
                                    or clicked
                                ):
                                    self.active_module = hovered
                                    self.hover_start = None
                                    print(
                                        f"Activated {hovered} module"
                                    )  # Debug feedback
                            else:
                                self.hover_start = None
                    else:
                        self.hover_start = None
                        self.last_cursor_pos = (
                            None  # Reset cursor smoothing when hand not detected
                        )


                    # Vision overlay in main interface
                    if hasattr(self, "vision_manager") and self.vision_manager:
                        try:
                            analysis = self.vision_manager.analyze_frame(img)
                            # Draw bounding boxes
                            for obj in analysis.get("objects", []):
                                x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
                                conf = obj["confidence"]
                                label = f"{obj['class_name']} {conf:.2f}"
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(img, label, (x1, max(20, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                            # Draw scene description
                            scene = analysis.get("scene_description", "")
                            if scene:
                                cv2.putText(img, scene, (10, self.height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        except Exception:
                            pass
                # Show enhanced FPS counter
                delta = time.time() - current_time
                fps = 1 / delta if delta > 0 else 0

                # FPS background
                fps_text = f"FPS: {int(fps)}"
                fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[
                    0
                ]
                cv2.rectangle(img, (5, 5), (fps_size[0] + 15, 35), (0, 0, 0), -1)
                cv2.rectangle(img, (5, 5), (fps_size[0] + 15, 35), (100, 100, 100), 1)

                cv2.putText(
                    img,
                    fps_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Show resolution info
                res_text = f"{self.width}x{self.height}"
                res_size = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[
                    0
                ]
                cv2.rectangle(
                    img,
                    (self.width - res_size[0] - 15, 5),
                    (self.width - 5, 25),
                    (0, 0, 0),
                    -1,
                )
                cv2.rectangle(
                    img,
                    (self.width - res_size[0] - 15, 5),
                    (self.width - 5, 25),
                    (100, 100, 100),
                    1,
                )
                cv2.putText(
                    img,
                    res_text,
                    (self.width - res_size[0] - 10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
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
                elif key == ord("d"):
                    # Toggle debug mode
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")

        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources and close windows."""
        try:
            print("Starting application cleanup...")

            # Cleanup all module resources
            if hasattr(self, "voice_assistant") and self.voice_assistant:
                try:
                    self.voice_assistant.cleanup()
                    print("Voice assistant cleaned up")
                except Exception as e:
                    print(f"Error cleaning up voice assistant: {e}")

            if hasattr(self, "camera_module") and self.camera_module:
                try:
                    self.camera_module.cleanup()
                    print("Camera module cleaned up")
                except Exception as e:
                    print(f"Error cleaning up camera module: {e}")

            if hasattr(self, "canvas_module") and self.canvas_module:
                try:
                    self.canvas_module.cleanup()
                    print("Canvas module cleaned up")
                except Exception as e:
                    print(f"Error cleaning up canvas module: {e}")

            # Clean up hand tracker
            if hasattr(self, "tracker") and self.tracker:
                try:
                    if hasattr(self.tracker, "cleanup"):
                        self.tracker.cleanup()
                    print("Hand tracker cleaned up")
                except Exception as e:
                    print(f"Error cleaning up hand tracker: {e}")

            # Release camera resources
            if hasattr(self, "cap") and self.cap:
                try:
                    self.cap.release()
                    print("Camera released")
                except Exception as e:
                    print(f"Error releasing camera: {e}")

            # Close all OpenCV windows
            try:
                cv2.destroyAllWindows()
                print("OpenCV windows closed")
            except Exception as e:
                print(f"Error closing OpenCV windows: {e}")

            print("Application cleanup completed successfully")

        except Exception as e:
            print(f"Error during main cleanup: {e}")
            # Force cleanup even if there are errors
            try:
                if hasattr(self, "cap") and self.cap:
                    self.cap.release()
                cv2.destroyAllWindows()
            except:
                pass

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection"""
        try:
            self.cleanup()
        except:
            pass  # Silently handle cleanup errors during destruction


if __name__ == "__main__":
    app = MainApp()
    app.run()
