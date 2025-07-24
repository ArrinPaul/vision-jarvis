import cv2
import mediapipe as mp
import math
import time
from collections import deque
import numpy as np


class HandTracker:
    def __init__(
        self,
        max_hands=1,
        detection_con=0.7,
        track_con=0.5,
        static_image_mode=False,
    ):
        """
        Completely refactored Hand Tracker with consistent coordinate handling

        Args:
            max_hands: Maximum number of hands to detect
            detection_con: Detection confidence threshold (0.0-1.0)
            track_con: Tracking confidence threshold (0.0-1.0)
            static_image_mode: Whether to treat images as static
        """
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con,
            model_complexity=1,  # Better accuracy
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        # Tracking state
        self.current_landmarks = None
        self.current_handedness = None
        self.hand_detected = False
        self.detection_confidence = 0.0

        # Smoothing and history
        self.position_history = deque(maxlen=3)
        self.gesture_history = deque(maxlen=5)

        # Landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.THUMB_IP = 3
        self.INDEX_TIP = 8
        self.INDEX_PIP = 6
        self.INDEX_MCP = 5
        self.MIDDLE_TIP = 12
        self.MIDDLE_PIP = 10
        self.RING_TIP = 16
        self.PINKY_TIP = 20

        # Gesture thresholds (normalized coordinates 0-1)
        self.PINCH_THRESHOLD = 0.04
        self.POINT_THRESHOLD = 0.03
        self.HAND_SIZE_THRESHOLD = 0.02  # Minimum hand size to consider valid

        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()

    def find_hands(self, img, draw=True):
        """
        Detect hands in the image and return landmarks in normalized coordinates

        Args:
            img: Input image (BGR)
            draw: Whether to draw landmarks on the image

        Returns:
            img: Image with drawings (if draw=True)
            landmarks: List of normalized landmarks [x, y, z] or None
            hand_info: Dictionary with hand information
        """
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False

        # Process the image
        results = self.hands.process(img_rgb)

        # Convert back to BGR
        img_rgb.flags.writeable = True
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Reset state
        self.current_landmarks = None
        self.current_handedness = None
        self.hand_detected = False
        self.detection_confidence = 0.0

        hand_info = {
            "detected": False,
            "handedness": "Unknown",
            "confidence": 0.0,
            "landmarks_count": 0,
            "hand_size": 0.0,
            "center": (0.5, 0.5),
        }

        if results.multi_hand_landmarks:
            # Get the first (primary) hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Extract normalized landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # Get handedness info
            if results.multi_handedness:
                handedness_info = results.multi_handedness[0]
                hand_label = handedness_info.classification[0].label
                hand_confidence = handedness_info.classification[0].score
            else:
                hand_label = "Unknown"
                hand_confidence = 0.0

            # Update state
            self.current_landmarks = landmarks
            self.current_handedness = hand_label
            self.hand_detected = True
            self.detection_confidence = hand_confidence

            # Calculate hand metrics
            hand_size = self._calculate_hand_size(landmarks)
            center = self._calculate_center(landmarks)

            # Update hand info
            hand_info.update(
                {
                    "detected": True,
                    "handedness": hand_label,
                    "confidence": hand_confidence,
                    "landmarks_count": len(landmarks),
                    "hand_size": hand_size,
                    "center": center,
                }
            )

            # Draw landmarks if requested
            if draw:
                self._draw_landmarks(img, hand_landmarks)

        # Update FPS
        self._update_fps()

        return img, self.current_landmarks, hand_info

    def get_finger_position(self, landmark_id):
        """
        Get the position of a specific landmark in normalized coordinates

        Args:
            landmark_id: MediaPipe landmark ID (0-20)

        Returns:
            (x, y): Normalized coordinates (0-1) or None if not detected
        """
        if not self.hand_detected or not self.current_landmarks:
            return None

        if 0 <= landmark_id < len(self.current_landmarks):
            lm = self.current_landmarks[landmark_id]
            return (lm[0], lm[1])  # Return x, y (normalized)

        return None

    def get_index_finger_tip(self):
        """Get index finger tip position in normalized coordinates"""
        return self.get_finger_position(self.INDEX_TIP)

    def get_thumb_tip(self):
        """Get thumb tip position in normalized coordinates"""
        return self.get_finger_position(self.THUMB_TIP)

    def detect_pinch(self):
        """
        Detect pinch gesture between thumb and index finger

        Returns:
            (is_pinching, distance): Boolean and normalized distance
        """
        if not self.hand_detected:
            return False, 1.0

        thumb_pos = self.get_thumb_tip()
        index_pos = self.get_index_finger_tip()

        if thumb_pos is None or index_pos is None:
            return False, 1.0

        # Calculate distance
        distance = math.sqrt(
            (thumb_pos[0] - index_pos[0]) ** 2 + (thumb_pos[1] - index_pos[1]) ** 2
        )

        is_pinching = distance < self.PINCH_THRESHOLD
        return is_pinching, distance

    def detect_pointing(self):
        """
        Detect pointing gesture (index finger extended, others folded)

        Returns:
            (is_pointing, confidence): Boolean and confidence score
        """
        if not self.hand_detected:
            return False, 0.0

        fingers_up = self._get_fingers_up()

        # Index finger up, others down (except thumb which can vary)
        if len(fingers_up) >= 5:
            index_up = fingers_up[1] == 1  # Index finger
            middle_down = fingers_up[2] == 0  # Middle finger
            ring_down = fingers_up[3] == 0  # Ring finger
            pinky_down = fingers_up[4] == 0  # Pinky finger

            if index_up and middle_down and ring_down and pinky_down:
                return True, 0.9

        return False, 0.0

    def get_gesture(self):
        """
        Detect and return the current gesture

        Returns:
            (gesture_name, confidence): Gesture string and confidence score
        """
        if not self.hand_detected:
            return "none", 0.0

        # Check for pinch first (highest priority)
        is_pinching, pinch_distance = self.detect_pinch()
        if is_pinching:
            confidence = max(0.5, 1.0 - (pinch_distance / self.PINCH_THRESHOLD))
            return "pinch", min(1.0, confidence)

        # Check for pointing
        is_pointing, point_confidence = self.detect_pointing()
        if is_pointing:
            return "point", point_confidence

        # Default to open hand
        return "open", 0.5

    def _get_fingers_up(self):
        """
        Determine which fingers are extended

        Returns:
            List of 5 integers (0 or 1) for [thumb, index, middle, ring, pinky]
        """
        if not self.current_landmarks:
            return [0, 0, 0, 0, 0]

        fingers = []

        # Thumb (compare tip with IP joint, considering handedness)
        thumb_tip = self.current_landmarks[self.THUMB_TIP]
        thumb_ip = self.current_landmarks[self.THUMB_IP]

        if self.current_handedness == "Right":
            fingers.append(1 if thumb_tip[0] > thumb_ip[0] else 0)
        else:
            fingers.append(1 if thumb_tip[0] < thumb_ip[0] else 0)

        # Other fingers (compare tip with PIP joint)
        finger_tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        finger_pips = [self.INDEX_PIP, self.MIDDLE_PIP, 14, 18]  # PIP joints

        for tip_id, pip_id in zip(finger_tips, finger_pips):
            tip = self.current_landmarks[tip_id]
            pip = self.current_landmarks[pip_id]
            fingers.append(1 if tip[1] < pip[1] else 0)  # Finger up if tip above PIP

        return fingers

    def _calculate_hand_size(self, landmarks):
        """Calculate hand size from wrist to middle finger tip"""
        if len(landmarks) < 13:
            return 0.0

        wrist = landmarks[self.WRIST]
        middle_tip = landmarks[self.MIDDLE_TIP]

        distance = math.sqrt(
            (middle_tip[0] - wrist[0]) ** 2 + (middle_tip[1] - wrist[1]) ** 2
        )

        return distance

    def _calculate_center(self, landmarks):
        """Calculate center point of the hand"""
        if not landmarks:
            return (0.5, 0.5)

        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]

        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        return (center_x, center_y)

    def _draw_landmarks(self, img, hand_landmarks):
        """Draw hand landmarks and connections on the image"""
        # Draw landmarks
        self.mp_draw.draw_landmarks(
            img,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw_styles.get_default_hand_landmarks_style(),
            self.mp_draw_styles.get_default_hand_connections_style(),
        )

        # Highlight important points
        h, w, _ = img.shape

        # Index finger tip (red circle)
        if len(hand_landmarks.landmark) > self.INDEX_TIP:
            lm = hand_landmarks.landmark[self.INDEX_TIP]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)

        # Thumb tip (blue circle)
        if len(hand_landmarks.landmark) > self.THUMB_TIP:
            lm = hand_landmarks.landmark[self.THUMB_TIP]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 8, (255, 0, 0), -1)

    def _update_fps(self):
        """Update FPS counter"""
        current_time = time.time()
        if self.last_time > 0:
            fps = 1.0 / (current_time - self.last_time)
            self.fps_counter.append(fps)
        self.last_time = current_time

    def get_fps(self):
        """Get current average FPS"""
        if len(self.fps_counter) == 0:
            return 0
        return sum(self.fps_counter) / len(self.fps_counter)

    def reset_tracking(self):
        """Reset all tracking state and history"""
        self.current_landmarks = None
        self.current_handedness = None
        self.hand_detected = False
        self.detection_confidence = 0.0
        self.position_history.clear()
        self.gesture_history.clear()
        self.fps_counter.clear()

    def get_legacy_landmark_list(self):
        """
        Convert to legacy format for backward compatibility

        Returns:
            List in format [[id, x, y], ...] where x,y are normalized coordinates
        """
        if not self.current_landmarks:
            return []

        legacy_list = []
        for i, lm in enumerate(self.current_landmarks):
            legacy_list.append([i, lm[0], lm[1]])  # [id, x_norm, y_norm]

        return legacy_list
