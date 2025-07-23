import cv2
import time
import os
import numpy as np
from utils import load_icon, overlay_image, is_point_in_rect

class CameraModule:
    def __init__(self):
        self.hover_start = None
        self.captured = False
        self.capture_time = 0
        self.captured_img = None
        self.return_icon = load_icon("return_icon.png", (80, 80))
        self.camera_icon = load_icon("camera_icon.png", (120, 120))
        
        # Create album directory
        if not os.path.exists("album"):
            os.makedirs("album")
    
    def run(self, img, lm_list):
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