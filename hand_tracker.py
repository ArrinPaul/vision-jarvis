import cv2
import mediapipe as mp
import time
import math
from collections import deque

# Try to import numpy, fallback to math if not available
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("NumPy not available, using basic math operations")


class HandTracker:
    def __init__(
        self, max_hands=1, detection_con=0.7, track_con=0.5, static_image_mode=False
    ):
        """
        Enhanced HandTracker optimized for single hand detection

        Args:
            max_hands: Maximum number of hands to detect (default: 1 for optimal performance)
            detection_con: Minimum confidence for hand detection (0.0-1.0)
            track_con: Minimum confidence for hand tracking (0.0-1.0)
            static_image_mode: Whether to process images as static (False for video)
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con,
            model_complexity=1,  # 0=lite, 1=full (better accuracy)
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()

        # Gesture smoothing and filtering
        self.gesture_history = deque(maxlen=5)
        self.position_smoothing = deque(maxlen=3)

        # Enhanced landmark indices for better gesture recognition
        self.finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        self.finger_pips = [3, 6, 10, 14, 18]  # PIP joints
        self.finger_mcp = [2, 5, 9, 13, 17]  # MCP joints

        # Calibration for dynamic thresholds
        self.hand_size_history = deque(maxlen=10)
        self.pinch_threshold_ratio = 0.08  # Ratio of hand size for pinch detection

    def find_hands(self, img, draw=True, draw_style="landmarks"):
        """
        Enhanced hand detection with better drawing options and performance tracking

        Args:
            img: Input image
            draw: Whether to draw landmarks and connections
            draw_style: Drawing style - "landmarks", "connections", "both", "box"

        Returns:
            img: Processed image with drawings
            results: MediaPipe results object
            fps: Current FPS
        """
        # Performance tracking
        current_time = time.time()
        fps = 1 / (current_time - self.last_time) if self.last_time else 0
        self.last_time = current_time
        self.fps_counter.append(fps)
        avg_fps = sum(self.fps_counter) / len(self.fps_counter)

        # Process image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False  # Performance optimization
        results = self.hands.process(img_rgb)
        img_rgb.flags.writeable = True

        # Draw results based on style
        if results.multi_hand_landmarks and draw:
            for hand_idx, hand_lms in enumerate(results.multi_hand_landmarks):
                if draw_style in ["landmarks", "both"]:
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_lms,
                        None,
                        self.mp_draw_styles.get_default_hand_landmarks_style(),
                        None,
                    )

                if draw_style in ["connections", "both"]:
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS,
                        None,
                        self.mp_draw_styles.get_default_hand_connections_style(),
                    )

                if draw_style == "box":
                    self._draw_bounding_box(img, hand_lms)

        return img, results, avg_fps

    def _draw_bounding_box(self, img, hand_landmarks):
        """Draw bounding box around hand"""
        h, w, _ = img.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        cv2.rectangle(
            img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2
        )

    def find_position(self, img, results, hand_no=0, draw=True, smooth=True):
        """
        Enhanced position finding with smoothing and additional hand information

        Args:
            img: Input image
            results: MediaPipe results
            hand_no: Hand index (0 or 1)
            draw: Whether to draw landmark points
            smooth: Whether to apply position smoothing

        Returns:
            lm_list: List of landmark positions [id, x, y, z]
            hand_info: Dictionary with hand information
        """
        lm_list = []
        hand_info = {
            "handedness": None,
            "confidence": 0,
            "hand_size": 0,
            "center": (0, 0),
            "bbox": None,
        }

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > hand_no:
            hand = results.multi_hand_landmarks[hand_no]
            h, w, c = img.shape

            # Get handedness and confidence
            if results.multi_handedness and len(results.multi_handedness) > hand_no:
                handedness = results.multi_handedness[hand_no]
                hand_info["handedness"] = handedness.classification[0].label
                hand_info["confidence"] = handedness.classification[0].score

            # Extract landmarks with 3D coordinates
            positions = []
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cz = lm.z  # Depth information
                positions.append([id, cx, cy, cz])

                if draw:
                    # Color-code different finger parts
                    color = self._get_landmark_color(id)
                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

            # Apply smoothing if enabled
            if smooth and len(self.position_smoothing) > 0:
                positions = self._smooth_positions(positions)

            self.position_smoothing.append(positions)
            lm_list = positions

            # Calculate hand metrics
            hand_info["hand_size"] = self._calculate_hand_size(lm_list)
            hand_info["center"] = self._calculate_hand_center(lm_list)
            hand_info["bbox"] = self._calculate_bounding_box(lm_list)

            self.hand_size_history.append(hand_info["hand_size"])

        return lm_list, hand_info

    def _get_landmark_color(self, landmark_id):
        """Get color for landmark based on finger part"""
        if landmark_id in [0, 1, 2, 3, 4]:  # Thumb
            return (0, 0, 255)  # Red
        elif landmark_id in [5, 6, 7, 8]:  # Index
            return (0, 255, 0)  # Green
        elif landmark_id in [9, 10, 11, 12]:  # Middle
            return (255, 0, 0)  # Blue
        elif landmark_id in [13, 14, 15, 16]:  # Ring
            return (0, 255, 255)  # Yellow
        elif landmark_id in [17, 18, 19, 20]:  # Pinky
            return (255, 0, 255)  # Magenta
        else:
            return (255, 255, 255)  # White for wrist

    def _smooth_positions(self, current_positions):
        """Apply smoothing to landmark positions"""
        if len(self.position_smoothing) == 0:
            return current_positions

        smoothed = []
        alpha = 0.7  # Smoothing factor

        for i, (id, cx, cy, cz) in enumerate(current_positions):
            if len(self.position_smoothing) > 0:
                prev_positions = self.position_smoothing[-1]
                if i < len(prev_positions):
                    prev_cx, prev_cy = prev_positions[i][1], prev_positions[i][2]
                    cx = int(alpha * cx + (1 - alpha) * prev_cx)
                    cy = int(alpha * cy + (1 - alpha) * prev_cy)

            smoothed.append([id, cx, cy, cz])

        return smoothed

    def _calculate_hand_size(self, lm_list):
        """Calculate hand size based on wrist to middle finger tip distance"""
        if len(lm_list) < 21:
            return 0

        wrist = lm_list[0]
        middle_tip = lm_list[12]

        distance = math.sqrt(
            (middle_tip[1] - wrist[1]) ** 2 + (middle_tip[2] - wrist[2]) ** 2
        )

        return distance

    def _calculate_hand_center(self, lm_list):
        """Calculate center point of hand"""
        if not lm_list:
            return (0, 0)

        x_coords = [lm[1] for lm in lm_list]
        y_coords = [lm[2] for lm in lm_list]

        center_x = sum(x_coords) // len(x_coords)
        center_y = sum(y_coords) // len(y_coords)

        return (center_x, center_y)

    def _calculate_bounding_box(self, lm_list):
        """Calculate bounding box of hand"""
        if not lm_list:
            return None

        x_coords = [lm[1] for lm in lm_list]
        y_coords = [lm[2] for lm in lm_list]

        return {
            "x_min": min(x_coords),
            "x_max": max(x_coords),
            "y_min": min(y_coords),
            "y_max": max(y_coords),
        }

    def get_fingers_state(self, lm_list, hand_info=None):
        """
        Enhanced finger state detection with improved accuracy

        Args:
            lm_list: List of landmark positions
            hand_info: Hand information dictionary

        Returns:
            fingers: List of finger states [thumb, index, middle, ring, pinky]
        """
        if len(lm_list) < 21:
            return []
            
        fingers = []
        handedness = hand_info.get("handedness", "Right") if hand_info else "Right"

        # Thumb - Different logic for left/right hands
        if handedness == "Right":
            # For right hand, thumb up when tip is to the right of joint
            if lm_list[4][1] > lm_list[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # For left hand, thumb up when tip is to the left of joint
            if lm_list[4][1] < lm_list[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Other fingers - Compare tip with PIP joint
        for finger_id in range(1, 5):
            tip_id = self.finger_tips[finger_id]
            pip_id = self.finger_pips[finger_id]

            # Finger is up if tip is above PIP joint
            if lm_list[tip_id][2] < lm_list[pip_id][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

    def get_gesture(self, lm_list, hand_info=None, smooth=True):
        """
        Enhanced gesture recognition with more gestures and smoothing

        Args:
            lm_list: List of landmark positions
            hand_info: Hand information dictionary
            smooth: Whether to apply gesture smoothing

        Returns:
            gesture: Detected gesture string
            confidence: Confidence score for the gesture
        """
        if len(lm_list) < 21:
            return "none", 0.0

        fingers = self.get_fingers_state(lm_list, hand_info)

        # Calculate dynamic thresholds based on hand size
        hand_size = hand_info.get("hand_size", 100) if hand_info else 100
        avg_hand_size = (
            sum(self.hand_size_history) / len(self.hand_size_history)
            if self.hand_size_history
            else hand_size
        )
        pinch_threshold = max(20, avg_hand_size * self.pinch_threshold_ratio)

        # Key landmarks
        thumb_tip = lm_list[4]
        index_tip = lm_list[8]
        middle_tip = lm_list[12]
        ring_tip = lm_list[16]
        pinky_tip = lm_list[20]

        # Distance calculations
        thumb_index_dist = self._calculate_distance(thumb_tip, index_tip)
        thumb_middle_dist = self._calculate_distance(thumb_tip, middle_tip)
        index_middle_dist = self._calculate_distance(index_tip, middle_tip)

        gesture = "unknown"
        confidence = 0.5

        # Enhanced gesture recognition
        if thumb_index_dist < pinch_threshold:
            gesture = "pinch"
            confidence = min(
                1.0, (pinch_threshold - thumb_index_dist) / pinch_threshold + 0.5
            )

        elif thumb_middle_dist < pinch_threshold:
            gesture = "precision_pinch"
            confidence = min(
                1.0, (pinch_threshold - thumb_middle_dist) / pinch_threshold + 0.5
            )

        elif fingers == [0, 1, 0, 0, 0]:
            gesture = "point"
            confidence = 0.9

        elif fingers == [0, 1, 1, 0, 0]:
            gesture = "peace"
            confidence = 0.9

        elif fingers == [1, 1, 1, 0, 0]:
            gesture = "three"
            confidence = 0.8

        elif fingers == [1, 1, 1, 1, 0]:
            gesture = "four"
            confidence = 0.8

        elif all(fingers):
            gesture = "open"
            confidence = 0.9

        elif not any(fingers):
            gesture = "fist"
            confidence = 0.9

        elif fingers == [1, 0, 0, 0, 0]:
            gesture = "thumbs_up"
            confidence = 0.8

        elif fingers == [0, 0, 0, 0, 1]:
            gesture = "pinky"
            confidence = 0.7

        elif fingers == [1, 0, 0, 0, 1]:
            gesture = "rock_on"
            confidence = 0.7

        elif fingers == [0, 1, 0, 0, 1]:
            gesture = "call_me"
            confidence = 0.7

        # L-shape detection
        elif fingers == [1, 1, 0, 0, 0]:
            # Check if thumb and index form an L
            if self._is_l_shape(lm_list):
                gesture = "l_shape"
                confidence = 0.8

        # OK sign detection (thumb-index circle with other fingers up)
        elif thumb_index_dist < pinch_threshold * 1.5 and fingers[2:] == [1, 1, 1]:
            gesture = "ok"
            confidence = 0.8

        # Apply gesture smoothing
        if smooth:
            gesture = self._smooth_gesture(gesture, confidence)

        return gesture, confidence

    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

    def _is_l_shape(self, lm_list):
        """Check if thumb and index finger form an L shape"""
        thumb_tip = lm_list[4]
        thumb_ip = lm_list[3]
        index_tip = lm_list[8]
        index_pip = lm_list[6]

        # Calculate angles
        thumb_angle = math.atan2(thumb_tip[2] - thumb_ip[2], thumb_tip[1] - thumb_ip[1])
        index_angle = math.atan2(
            index_tip[2] - index_pip[2], index_tip[1] - index_pip[1]
        )

        angle_diff = abs(thumb_angle - index_angle)
        # Check if angle is close to 90 degrees (π/2 radians)
        return (
            abs(angle_diff - math.pi / 2) < 0.5
            or abs(angle_diff - 3 * math.pi / 2) < 0.5
        )

    def _smooth_gesture(self, current_gesture, confidence):
        """Apply temporal smoothing to gesture recognition"""
        self.gesture_history.append((current_gesture, confidence))

        if len(self.gesture_history) < 3:
            return current_gesture

        # Count occurrences of each gesture in recent history
        gesture_counts = {}
        total_confidence = 0

        for gesture, conf in self.gesture_history:
            if gesture not in gesture_counts:
                gesture_counts[gesture] = {"count": 0, "confidence": 0}
            gesture_counts[gesture]["count"] += 1
            gesture_counts[gesture]["confidence"] += conf
            total_confidence += conf

        # Return the most frequent gesture with highest average confidence
        best_gesture = current_gesture
        best_score = 0

        for gesture, data in gesture_counts.items():
            avg_confidence = data["confidence"] / data["count"]
            score = data["count"] * avg_confidence

            if score > best_score:
                best_score = score
                best_gesture = gesture

        return best_gesture

    def get_hand_landmarks_3d(self, results, hand_no=0):
        """
        Get 3D world coordinates of hand landmarks

        Returns:
            landmarks_3d: List of 3D coordinates [(x, y, z), ...]
        """
        landmarks_3d = []

        if (
            results.multi_hand_world_landmarks
            and len(results.multi_hand_world_landmarks) > hand_no
        ):

            hand_world = results.multi_hand_world_landmarks[hand_no]

            for landmark in hand_world.landmark:
                landmarks_3d.append((landmark.x, landmark.y, landmark.z))

        return landmarks_3d

    def calculate_gesture_velocity(self, lm_list):
        """
        Calculate velocity of key landmarks for dynamic gesture recognition

        Returns:
            velocities: Dictionary of landmark velocities
        """
        if len(self.position_smoothing) < 2:
            return {}

        current_pos = lm_list
        prev_pos = (
            self.position_smoothing[-2]
            if len(self.position_smoothing) >= 2
            else lm_list
        )

        velocities = {}
        key_landmarks = [4, 8, 12, 16, 20]  # Fingertips

        for lm_id in key_landmarks:
            if lm_id < len(current_pos) and lm_id < len(prev_pos):
                curr = current_pos[lm_id]
                prev = prev_pos[lm_id]

                vel_x = curr[1] - prev[1]
                vel_y = curr[2] - prev[2]

                velocity_magnitude = math.sqrt(vel_x**2 + vel_y**2)
                velocities[lm_id] = {
                    "magnitude": velocity_magnitude,
                    "direction": math.atan2(vel_y, vel_x),
                }

        return velocities

    def get_legacy_landmark_list(self, lm_list):
        """
        Convert enhanced landmark format to legacy format for backward compatibility

        Args:
            lm_list: Enhanced landmark list with [id, x, y, z] format

        Returns:
            legacy_lm_list: Legacy format with [id, x, y] for compatibility
        """
        if not lm_list:
            return []

        # Convert from [id, x, y, z] to [id, x, y] format
        legacy_list = []
        for landmark in lm_list:
            if len(landmark) >= 3:
                legacy_list.append([landmark[0], landmark[1], landmark[2]])

        return legacy_list

    def reset_tracking(self):
        """Reset all tracking history and smoothing buffers"""
        self.gesture_history.clear()
        self.position_smoothing.clear()
        self.hand_size_history.clear()
        self.fps_counter.clear()
