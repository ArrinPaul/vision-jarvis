import cv2
import time
import os
import numpy as np
from datetime import datetime
from collections import deque
from utils import load_icon, overlay_image, is_point_in_rect


class CameraModule:
    def __init__(self):
        # Hover and interaction states
        self.hover_start = None
        self.capture_hover_start = None
        self.last_hand_time = time.time()

        # Capture states and settings
        self.captured = False
        self.capture_time = 0
        self.captured_img = None
        self.capture_countdown = 0
        self.capture_ready = False
        self.continuous_capture = False
        self.capture_interval = 2.0  # seconds between auto captures
        self.last_auto_capture = 0

        # UI elements and visual feedback
        self.return_icon = load_icon("return_icon.png", (80, 80))
        self.camera_icon = load_icon("camera_icon.png", (120, 120))

        # Gesture detection improvements
        self.hover_time = 1.2  # Reduced hover time for better responsiveness
        self.capture_hover_time = 0.8  # Quick capture activation

        # Position stabilization
        self.position_history = deque(maxlen=5)
        self.last_stable_position = None

        # Photo management
        self.photo_count = 0
        self.session_photos = []

        # Create album directory with better organization
        self.album_dir = "album"
        self.session_dir = None
        self._create_session_directory()

        # Display settings
        self.preview_size = (320, 240)  # Larger preview window
        self.preview_duration = 4.0  # Longer preview duration

    def _create_session_directory(self):
        """Create a session-based directory for better photo organization"""
        if not os.path.exists(self.album_dir):
            os.makedirs(self.album_dir)

        # Create session directory with timestamp
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.album_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)

    def _stabilize_position(self, position):
        """Stabilize cursor position to reduce jitter"""
        self.position_history.append(position)

        if len(self.position_history) < 3:
            return position

        # Use weighted average with more weight on recent positions
        weights = [0.5, 0.3, 0.2]  # Most recent gets highest weight
        x_avg = sum(
            pos[0] * weight
            for pos, weight in zip(reversed(list(self.position_history)[-3:]), weights)
        )
        y_avg = sum(
            pos[1] * weight
            for pos, weight in zip(reversed(list(self.position_history)[-3:]), weights)
        )

        return (int(x_avg), int(y_avg))

    def _get_safe_filename(self):
        """Generate a safe, unique filename for captured images"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.photo_count += 1
        filename = f"capture_{timestamp}_{self.photo_count:03d}.jpg"
        return os.path.join(self.session_dir, filename)

    def _draw_capture_ui(self, img):
        """Draw enhanced capture interface with visual feedback"""
        # Draw main camera icon with pulsing effect when ready
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

        if self.capture_ready:
            # Pulsing effect when ready to capture
            pulse = int(10 * abs(np.sin(time.time() * 6)))  # Fast pulse
            icon_size = 120 + pulse
            temp_icon = cv2.resize(self.camera_icon, (icon_size, icon_size))
            img = overlay_image(
                img, temp_icon, center_x - icon_size // 2, center_y - icon_size // 2
            )
        else:
            img = overlay_image(img, self.camera_icon, center_x - 60, center_y - 60)

        # Draw capture modes info
        mode_text = "CONTINUOUS" if self.continuous_capture else "SINGLE SHOT"
        cv2.putText(
            img,
            f"Mode: {mode_text}",
            (20, img.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Draw photo counter
        cv2.putText(
            img,
            f"Photos: {self.photo_count}",
            (20, img.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Draw instructions
        instructions = [
            "Peace sign (V) to capture photo",
            "Hover return button to exit",
            "Thumbs up for continuous mode",
        ]

        for i, instruction in enumerate(instructions):
            cv2.putText(
                img,
                instruction,
                (img.shape[1] - 300, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )

        return img

    def _draw_countdown(self, img, countdown):
        """Draw capture countdown"""
        if countdown > 0:
            # Draw countdown circle
            center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
            radius = 80

            # Background circle
            cv2.circle(img, (center_x, center_y), radius, (0, 0, 0), -1)
            cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), 3)

            # Countdown number
            cv2.putText(
                img,
                str(int(countdown)),
                (center_x - 30, center_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3,
            )

        return img

    def _draw_capture_preview(self, img):
        """Draw captured image preview with enhanced display"""
        if self.captured and time.time() - self.capture_time < self.preview_duration:
            # Calculate preview position (top-right corner)
            preview_x = img.shape[1] - self.preview_size[0] - 20
            preview_y = 20

            # Resize captured image for preview
            small_img = cv2.resize(self.captured_img, self.preview_size)

            # Add border and overlay
            cv2.rectangle(
                img,
                (preview_x - 5, preview_y - 5),
                (
                    preview_x + self.preview_size[0] + 5,
                    preview_y + self.preview_size[1] + 5,
                ),
                (0, 255, 0),
                3,
            )

            img[
                preview_y : preview_y + self.preview_size[1],
                preview_x : preview_x + self.preview_size[0],
            ] = small_img

            # Add "CAPTURED!" text with fade effect
            remaining_time = self.preview_duration - (time.time() - self.capture_time)
            alpha = min(1.0, remaining_time / self.preview_duration)
            text_color = (0, int(255 * alpha), 0)

            cv2.putText(
                img,
                "CAPTURED!",
                (preview_x, preview_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2,
            )

        return img

    def run(self, img, lm_list):
        should_exit = False
        current_time = time.time()

        # Store clean image for potential capture (before UI overlays)
        self.clean_img = img.copy()

        # Draw enhanced UI
        img = self._draw_capture_ui(img)

        # Draw return button
        return_rect = (20, 20, 100, 100)
        img = overlay_image(img, self.return_icon, 20, 20)

        # Handle capture countdown
        if self.capture_countdown > 0:
            img = self._draw_countdown(img, self.capture_countdown)
            self.capture_countdown = max(0, self.capture_countdown - 0.1)

            if self.capture_countdown <= 0:
                self._take_photo(self.clean_img)

        # Display captured image preview
        img = self._draw_capture_preview(img)

        # Process hand gestures
        if lm_list:
            self.last_hand_time = current_time

            # Get stabilized finger positions
            index_tip = lm_list[8]
            thumb_tip = lm_list[4]

            # Convert normalized coordinates to pixel coordinates
            img_height, img_width = img.shape[:2]
            raw_position = (
                int(index_tip[1] * img_width),
                int(index_tip[2] * img_height),
            )
            x, y = self._stabilize_position(raw_position)
            self.last_stable_position = (x, y)

            # Enhanced cursor with hover feedback
            cursor_color = (0, 255, 0)
            cursor_size = 10

            # Show hover progress for return button
            if is_point_in_rect(x, y, return_rect):
                if self.hover_start is None:
                    self.hover_start = current_time

                hover_progress = (current_time - self.hover_start) / self.hover_time
                if hover_progress >= 1.0:
                    should_exit = True
                else:
                    # Visual feedback for hover progress
                    progress_radius = int(15 + hover_progress * 20)
                    cv2.circle(img, (x, y), progress_radius, (255, 255, 0), 2)
                    cursor_color = (255, 255, 0)
            else:
                self.hover_start = None

            # Draw main cursor
            cv2.circle(img, (x, y), cursor_size, cursor_color, cv2.FILLED)

            # Enhanced gesture detection
            self._process_gestures(img, lm_list, x, y, current_time)

            # Show gesture feedback
            self._draw_gesture_feedback(img, lm_list, x, y)

        else:
            # Handle brief hand tracking losses
            if current_time - self.last_hand_time < 0.5 and self.last_stable_position:
                x, y = self.last_stable_position
                cv2.circle(
                    img, (x, y), 8, (128, 128, 128), 2
                )  # Gray cursor for tracking loss

        # Auto-capture in continuous mode
        if (
            self.continuous_capture
            and current_time - self.last_auto_capture > self.capture_interval
            and lm_list
        ):  # Only auto-capture when hand is detected
            self._start_capture_countdown()
            self.last_auto_capture = current_time

        return img, should_exit

    def _process_gestures(self, img, lm_list, x, y, current_time):
        """Process hand gestures for camera control"""
        thumb_tip = lm_list[4]
        index_tip = lm_list[8]
        middle_tip = lm_list[12]

        # Peace sign gesture for photo capture
        if self._detect_peace_sign(lm_list):
            if self.capture_hover_start is None:
                self.capture_hover_start = current_time
                self.capture_ready = True

            hover_time = current_time - self.capture_hover_start
            if hover_time >= self.capture_hover_time:
                if (
                    not self.captured or current_time - self.capture_time > 1.0
                ):  # Prevent rapid captures
                    self._start_capture_countdown()
                    self.capture_hover_start = None
            else:
                # Show capture progress
                progress = hover_time / self.capture_hover_time
                progress_radius = int(20 + progress * 15)
                cv2.circle(img, (x, y), progress_radius, (0, 0, 255), 3)
        else:
            self.capture_hover_start = None
            self.capture_ready = False

        # Thumbs up for continuous mode toggle
        if self._detect_thumbs_up(lm_list):
            if not hasattr(self, "thumbs_detected") or not self.thumbs_detected:
                self.continuous_capture = not self.continuous_capture
                self.thumbs_detected = True

                # Visual feedback
                mode_text = (
                    "CONTINUOUS ON" if self.continuous_capture else "CONTINUOUS OFF"
                )
                cv2.putText(
                    img,
                    mode_text,
                    (x - 80, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                )
        else:
            self.thumbs_detected = False

    def _detect_peace_sign(self, lm_list):
        """Detect peace sign gesture (index and middle fingers up) - Enhanced version"""
        if len(lm_list) < 21:
            return False

        # Enhanced peace sign detection with better accuracy
        index_up = lm_list[8][2] < lm_list[6][2]  # Index tip above PIP
        middle_up = lm_list[12][2] < lm_list[10][2]  # Middle tip above PIP
        ring_down = lm_list[16][2] > lm_list[14][2]  # Ring tip below PIP
        pinky_down = lm_list[20][2] > lm_list[18][2]  # Pinky tip below PIP

        # Check thumb position - should be folded or neutral
        thumb_folded = lm_list[4][2] > lm_list[3][2]  # Thumb tip below IP joint

        # Additional check: ensure index and middle are reasonably separated
        finger_separation = abs(lm_list[8][1] - lm_list[12][1])  # Horizontal distance
        min_separation = 30  # Minimum pixels apart

        return (
            index_up
            and middle_up
            and ring_down
            and pinky_down
            and finger_separation > min_separation
        )

    def _detect_thumbs_up(self, lm_list):
        """Detect thumbs up gesture (only thumb extended upward)"""
        if len(lm_list) < 21:
            return False

        # Thumb up detection
        thumb_up = lm_list[4][2] < lm_list[3][2]  # Thumb tip above IP joint

        # Other fingers should be folded down
        index_down = lm_list[8][2] > lm_list[6][2]  # Index tip below PIP
        middle_down = lm_list[12][2] > lm_list[10][2]  # Middle tip below PIP
        ring_down = lm_list[16][2] > lm_list[14][2]  # Ring tip below PIP
        pinky_down = lm_list[20][2] > lm_list[18][2]  # Pinky tip below PIP

        # Additional check: thumb should be pointing upward (not sideways)
        thumb_vertical = abs(lm_list[4][2] - lm_list[2][2]) > abs(
            lm_list[4][1] - lm_list[2][1]
        )

        return (
            thumb_up
            and index_down
            and middle_down
            and ring_down
            and pinky_down
            and thumb_vertical
        )

    def _draw_gesture_feedback(self, img, lm_list, x, y):
        """Draw visual feedback for detected gestures"""
        feedback_y = y - 50

        if self._detect_peace_sign(lm_list):
            cv2.putText(
                img,
                "PEACE SIGN - READY TO CAPTURE!",
                (x - 100, feedback_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )  # Yellow text
        elif self._detect_thumbs_up(lm_list):
            cv2.putText(
                img,
                "THUMBS UP - TOGGLE MODE!",
                (x - 80, feedback_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2,
            )  # Magenta text

    def _start_capture_countdown(self):
        """Start the capture countdown sequence"""
        if self.capture_countdown <= 0:  # Only start if not already counting
            self.capture_countdown = 3.0  # 3 second countdown

    def _take_photo(self, clean_img):
        """Capture and save the photo with enhanced features"""
        try:
            # Use the clean image (without UI overlay) for capture
            self.captured_img = clean_img.copy()

            # Generate unique filename
            filename = self._get_safe_filename()

            # Save image with metadata
            success = cv2.imwrite(filename, self.captured_img)

            if success:
                self.captured = True
                self.capture_time = time.time()
                self.session_photos.append(filename)

                # Save session metadata
                self._save_session_metadata()

                print(f"Photo saved: {filename}")
            else:
                print("Failed to save photo!")

        except Exception as e:
            print(f"Error taking photo: {e}")

    def _save_session_metadata(self):
        """Save session metadata for photo management"""
        try:
            metadata = {
                "session_start": self.session_dir,
                "photo_count": self.photo_count,
                "photos": self.session_photos,
                "last_capture": time.time(),
            }

            metadata_file = os.path.join(self.session_dir, "session_info.txt")
            with open(metadata_file, "w") as f:
                f.write(f"Session: {os.path.basename(self.session_dir)}\n")
                f.write(f"Photos taken: {self.photo_count}\n")
                f.write(
                    f"Last capture: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write("\nPhotos:\n")
                for photo in self.session_photos:
                    f.write(f"- {os.path.basename(photo)}\n")
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def cleanup(self):
        """Clean up resources when module is being destroyed"""
        try:
            # Clear captured image data
            if hasattr(self, "captured_img"):
                self.captured_img = None

            # Clear position tracking data
            if hasattr(self, "position_history"):
                self.position_history.clear()

            # Clear session photos list
            if hasattr(self, "session_photos"):
                self.session_photos.clear()

            # Reset state variables
            self.captured = False
            self.capture_ready = False
            self.continuous_capture = False
            self.last_stable_position = None

            # Save final metadata before cleanup
            try:
                self._save_session_metadata()
            except:
                pass  # Don't fail cleanup if metadata save fails

            print("Camera module cleaned up successfully")

        except Exception as e:
            print(f"Error during camera cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection"""
        try:
            self.cleanup()
        except:
            pass  # Silently handle cleanup errors during destruction
