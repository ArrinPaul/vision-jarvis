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
        self.pinch_threshold = 35  # More sensitive pinch detection
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
        x_avg = sum(pos[0] * weight for pos, weight in zip(reversed(list(self.position_history)[-3:]), weights))
        y_avg = sum(pos[1] * weight for pos, weight in zip(reversed(list(self.position_history)[-3:]), weights))
        
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
        center_x, center_y = img.shape[1]//2, img.shape[0]//2
        
        if self.capture_ready:
            # Pulsing effect when ready to capture
            pulse = int(10 * abs(np.sin(time.time() * 6)))  # Fast pulse
            icon_size = 120 + pulse
            temp_icon = cv2.resize(self.camera_icon, (icon_size, icon_size))
            img = overlay_image(img, temp_icon, center_x - icon_size//2, center_y - icon_size//2)
        else:
            img = overlay_image(img, self.camera_icon, center_x - 60, center_y - 60)
        
        # Draw capture modes info
        mode_text = "CONTINUOUS" if self.continuous_capture else "SINGLE SHOT"
        cv2.putText(img, f"Mode: {mode_text}", (20, img.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw photo counter
        cv2.putText(img, f"Photos: {self.photo_count}", (20, img.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw instructions
        instructions = [
            "Pinch to capture photo",
            "Hover return button to exit",
            "Peace sign for continuous mode"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(img, instruction, (img.shape[1] - 300, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return img
    
    def _draw_countdown(self, img, countdown):
        """Draw capture countdown"""
        if countdown > 0:
            # Draw countdown circle
            center_x, center_y = img.shape[1]//2, img.shape[0]//2
            radius = 80
            
            # Background circle
            cv2.circle(img, (center_x, center_y), radius, (0, 0, 0), -1)
            cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), 3)
            
            # Countdown number
            cv2.putText(img, str(int(countdown)), 
                       (center_x - 30, center_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
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
            cv2.rectangle(img, (preview_x - 5, preview_y - 5), 
                         (preview_x + self.preview_size[0] + 5, preview_y + self.preview_size[1] + 5),
                         (0, 255, 0), 3)
            
            img[preview_y:preview_y + self.preview_size[1], 
                preview_x:preview_x + self.preview_size[0]] = small_img
            
            # Add "CAPTURED!" text with fade effect
            remaining_time = self.preview_duration - (time.time() - self.capture_time)
            alpha = min(1.0, remaining_time / self.preview_duration)
            text_color = (0, int(255 * alpha), 0)
            
            cv2.putText(img, "CAPTURED!", (preview_x, preview_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        return img
        should_exit = False
        
        # Draw UI
        center_x, center_y = img.shape[1]//2, img.shape[0]//2
        img = overlay_image(img, self.camera_icon, center_x - 60, center_y - 60)
        
        # Draw return button
        return_rect = (20, 20, 100, 100)
        img = overlay_image(img, self.return_icon, 20, 20)
        
        # Display captured image if recent
        if self.captured and time.time() - self.capture_time < 3:
            # Show captured image in a small window
            small_img = cv2.resize(self.captured_img, (300, 200))
            img[50:250, img.shape[1]-350:img.shape[1]-50] = small_img
            cv2.putText(img, "Captured!", (img.shape[1]-340, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Process hand gestures
        if lm_list:
            index_tip = lm_list[8]
            x, y = index_tip[1], index_tip[2]
            
            # Draw cursor
            cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)
            
            # Check return button
            if is_point_in_rect(x, y, return_rect):
                if self.hover_start is None:
                    self.hover_start = time.time()
                elif time.time() - self.hover_start >= 1.5:
                    should_exit = True
            else:
                self.hover_start = None
                
            # Check for pinch gesture (capture)
            thumb_tip = lm_list[4]
            index_tip = lm_list[8]
            distance = ((thumb_tip[1]-index_tip[1])**2 + 
                        (thumb_tip[2]-index_tip[2])**2)**0.5
            if distance < 30:  # Pinch threshold
                if not self.captured:
                    # Capture image
                    self.captured_img = img.copy()
                    filename = f"album/capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, self.captured_img)
                    self.captured = True
                    self.capture_time = time.time()
    
        return img, should_exit