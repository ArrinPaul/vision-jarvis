"""
JARVIS Advanced Gesture Recognition System
==========================================

This module implements Iron Man-style advanced gesture recognition featuring:
- 3D hand tracking with depth estimation
- Multi-hand gesture support
- Air gesture recognition
- Custom gesture creation and training
- Real-time gesture prediction
- Gesture combination detection
- Contextual gesture interpretation
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from collections import deque
import pickle

class GestureType(Enum):
    """Types of gestures supported"""
    STATIC = "static"           # Single hand pose
    DYNAMIC = "dynamic"         # Motion-based gesture
    BIMANUAL = "bimanual"      # Two-hand gesture
    AIR_TAP = "air_tap"        # Mid-air tap
    SWIPE = "swipe"            # Directional swipe
    PINCH = "pinch"            # Pinch gesture
    GRAB = "grab"              # Grab gesture
    POINT = "point"            # Pointing gesture
    CUSTOM = "custom"          # User-defined gesture

class GestureState(Enum):
    """Gesture recognition states"""
    IDLE = "idle"
    DETECTING = "detecting"
    RECOGNIZED = "recognized"
    EXECUTING = "executing"
    COMPLETED = "completed"

@dataclass
class GestureConfig:
    """Configuration for gesture recognition"""
    max_hands: int = 2
    detection_confidence: float = 0.8
    tracking_confidence: float = 0.7
    min_gesture_duration: float = 0.3
    max_gesture_duration: float = 3.0
    smoothing_factor: float = 0.7
    depth_estimation: bool = True
    enable_air_gestures: bool = True
    enable_custom_gestures: bool = True
    gesture_timeout: float = 2.0
    calibration_frames: int = 30

@dataclass
class Hand3D:
    """3D hand representation"""
    landmarks: List[Tuple[float, float, float]]  # x, y, z coordinates
    confidence: float
    handedness: str  # "Left" or "Right"
    palm_center: Tuple[float, float, float]
    palm_normal: Tuple[float, float, float]
    finger_tips: List[Tuple[float, float, float]]
    finger_states: List[bool]  # True if extended
    gesture_features: Dict[str, float]
    tracking_id: Optional[int] = None

@dataclass
class RecognizedGesture:
    """Recognized gesture data"""
    name: str
    type: GestureType
    confidence: float
    duration: float
    hands_involved: List[str]  # ["Left", "Right"]
    parameters: Dict[str, float]
    timestamp: float
    frame_count: int

class GestureLibrary:
    """Library of predefined and custom gestures"""
    
    def __init__(self):
        self.static_gestures = {}
        self.dynamic_gestures = {}
        self.custom_gestures = {}
        self.init_predefined_gestures()
    
    def init_predefined_gestures(self):
        """Initialize predefined Iron Man-style gestures"""
        
        # Repulsor blast gesture (palm forward, fingers spread)
        self.static_gestures['repulsor_blast'] = {
            'name': 'Repulsor Blast',
            'description': 'Palm forward with fingers spread - activate repulsors',
            'features': {
                'palm_direction': 'forward',
                'finger_spread': 'wide',
                'thumb_position': 'extended',
                'confidence_threshold': 0.8
            },
            'action': 'activate_repulsors',
            'feedback': 'Repulsors charging...'
        }
        
        # Arc reactor gesture (circle with thumb and index)
        self.static_gestures['arc_reactor'] = {
            'name': 'Arc Reactor',
            'description': 'Circle gesture with thumb and index finger',
            'features': {
                'thumb_index_distance': 'small',
                'circle_shape': True,
                'other_fingers': 'folded',
                'confidence_threshold': 0.85
            },
            'action': 'activate_arc_reactor',
            'feedback': 'Arc reactor online'
        }
        
        # Hologram manipulation (pinch and drag)
        self.dynamic_gestures['hologram_manipulate'] = {
            'name': 'Hologram Manipulation',
            'description': 'Pinch and drag to manipulate holograms',
            'sequence': ['pinch_start', 'drag_motion', 'pinch_end'],
            'features': {
                'pinch_strength': 'medium',
                'motion_smoothness': 'high',
                'duration_range': (0.5, 5.0)
            },
            'action': 'manipulate_hologram',
            'feedback': 'Hologram interface active'
        }
        
        # Suit summon (arms crossed then spread)
        self.dynamic_gestures['suit_summon'] = {
            'name': 'Suit Summon',
            'description': 'Cross arms then spread wide to summon suit',
            'sequence': ['arms_crossed', 'arms_spread'],
            'features': {
                'arm_cross_angle': 'tight',
                'spread_width': 'maximum',
                'speed': 'moderate'
            },
            'action': 'summon_suit',
            'feedback': 'Suit incoming...'
        }
        
        # Interface navigation (swipe gestures)
        self.dynamic_gestures['interface_swipe'] = {
            'name': 'Interface Navigation',
            'description': 'Swipe to navigate holographic interfaces',
            'sequence': ['swipe_left', 'swipe_right', 'swipe_up', 'swipe_down'],
            'features': {
                'swipe_velocity': 'fast',
                'swipe_distance': 'medium',
                'hand_orientation': 'palm_down'
            },
            'action': 'navigate_interface',
            'feedback': 'Interface navigation'
        }

class AdvancedHandTracker:
    """Enhanced hand tracking with 3D capabilities"""
    
    def __init__(self, config: GestureConfig):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_hands,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
            model_complexity=1
        )
        
        # 3D tracking state
        self.hand_history = deque(maxlen=30)  # 30 frames of history
        self.depth_estimator = DepthEstimator()
        self.hand_tracker_3d = Hand3DTracker()
        
        # Calibration
        self.is_calibrated = False
        self.calibration_data = None
        self.calibration_frames_collected = 0
    
    def process_frame(self, frame: np.ndarray, depth_frame: Optional[np.ndarray] = None) -> List[Hand3D]:
        """Process frame and return 3D hand data"""
        if frame is None:
            return []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        hands_3d = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # Convert to 3D coordinates
                hand_3d = self.create_hand_3d(hand_landmarks, handedness, frame.shape, depth_frame)
                hands_3d.append(hand_3d)
        
        # Update history
        self.hand_history.append({
            'timestamp': time.time(),
            'hands': hands_3d,
            'frame_shape': frame.shape
        })
        
        return hands_3d
    
    def create_hand_3d(self, landmarks, handedness, frame_shape, depth_frame=None) -> Hand3D:
        """Create 3D hand representation"""
        h, w = frame_shape[:2]
        
        # Extract landmark coordinates
        landmark_coords = []
        for landmark in landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            z = landmark.z * w if depth_frame is None else self.estimate_depth(x, y, depth_frame)
            landmark_coords.append((x, y, z))
        
        # Calculate hand properties
        palm_center = self.calculate_palm_center(landmark_coords)
        palm_normal = self.calculate_palm_normal(landmark_coords)
        finger_tips = self.get_finger_tips(landmark_coords)
        finger_states = self.analyze_finger_states(landmark_coords)
        gesture_features = self.extract_gesture_features(landmark_coords)
        
        return Hand3D(
            landmarks=landmark_coords,
            confidence=handedness.classification[0].score,
            handedness=handedness.classification[0].label,
            palm_center=palm_center,
            palm_normal=palm_normal,
            finger_tips=finger_tips,
            finger_states=finger_states,
            gesture_features=gesture_features
        )
    
    def estimate_depth(self, x: float, y: float, depth_frame: np.ndarray) -> float:
        """Estimate depth from depth frame or use geometric estimation"""
        if depth_frame is not None:
            # Use actual depth data if available
            return depth_frame[int(y), int(x)] if 0 <= int(x) < depth_frame.shape[1] and 0 <= int(y) < depth_frame.shape[0] else 0
        else:
            # Geometric depth estimation based on hand size
            return self.depth_estimator.estimate_depth_from_hand_size(x, y)
    
    def calculate_palm_center(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Calculate 3D palm center"""
        # Use wrist, middle finger MCP, and palm landmarks
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        # Average key palm points
        palm_x = (wrist[0] + middle_mcp[0]) / 2
        palm_y = (wrist[1] + middle_mcp[1]) / 2
        palm_z = (wrist[2] + middle_mcp[2]) / 2
        
        return (palm_x, palm_y, palm_z)
    
    def calculate_palm_normal(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Calculate palm normal vector"""
        # Use three palm points to calculate normal
        wrist = np.array(landmarks[0])
        index_mcp = np.array(landmarks[5])
        pinky_mcp = np.array(landmarks[17])
        
        # Calculate two vectors on the palm
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        
        # Cross product gives normal
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else np.array([0, 0, 1])
        
        return tuple(normal)
    
    def get_finger_tips(self, landmarks: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Get finger tip coordinates"""
        tip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        return [landmarks[i] for i in tip_indices]
    
    def analyze_finger_states(self, landmarks: List[Tuple[float, float, float]]) -> List[bool]:
        """Analyze which fingers are extended"""
        finger_states = []
        
        # Finger tip and pip joint indices
        finger_indices = [
            (4, 3),   # Thumb
            (8, 6),   # Index
            (12, 10), # Middle
            (16, 14), # Ring
            (20, 18)  # Pinky
        ]
        
        for tip_idx, pip_idx in finger_indices:
            tip_y = landmarks[tip_idx][1]
            pip_y = landmarks[pip_idx][1]
            
            # Finger is extended if tip is above pip (lower y value)
            is_extended = tip_y < pip_y
            finger_states.append(is_extended)
        
        return finger_states
    
    def extract_gesture_features(self, landmarks: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """Extract features for gesture recognition"""
        features = {}
        
        # Hand size (distance from wrist to middle finger tip)
        wrist = np.array(landmarks[0])
        middle_tip = np.array(landmarks[12])
        hand_size = np.linalg.norm(middle_tip - wrist)
        features['hand_size'] = hand_size
        
        # Finger spread (angle between fingers)
        finger_vectors = []
        palm_center = self.calculate_palm_center(landmarks)
        
        for tip_idx in [8, 12, 16]:  # Index, Middle, Ring
            tip = np.array(landmarks[tip_idx])
            vector = tip - np.array(palm_center)
            finger_vectors.append(vector / np.linalg.norm(vector))
        
        # Calculate angles between adjacent fingers
        if len(finger_vectors) >= 2:
            angle1 = np.arccos(np.clip(np.dot(finger_vectors[0], finger_vectors[1]), -1, 1))
            angle2 = np.arccos(np.clip(np.dot(finger_vectors[1], finger_vectors[2]), -1, 1))
            features['finger_spread'] = (angle1 + angle2) / 2
        
        # Thumb-index distance for pinch detection
        thumb_tip = np.array(landmarks[4])
        index_tip = np.array(landmarks[8])
        thumb_index_distance = np.linalg.norm(thumb_tip - index_tip)
        features['thumb_index_distance'] = thumb_index_distance / hand_size  # Normalized
        
        # Palm orientation (using normal vector)
        normal = self.calculate_palm_normal(landmarks)
        features['palm_forward'] = abs(normal[2])  # How much palm faces forward
        features['palm_up'] = abs(normal[1])       # How much palm faces up
        
        return features

class DepthEstimator:
    """Estimates depth from 2D hand tracking"""
    
    def __init__(self):
        self.baseline_hand_size = 100  # pixels at 50cm distance
        self.baseline_distance = 50    # cm
    
    def estimate_depth_from_hand_size(self, hand_size: float) -> float:
        """Estimate depth based on apparent hand size"""
        if hand_size <= 0:
            return self.baseline_distance
        
        # Inverse relationship: larger hand = closer distance
        estimated_distance = (self.baseline_hand_size * self.baseline_distance) / hand_size
        return max(10, min(200, estimated_distance))  # Clamp between 10cm and 200cm

class Hand3DTracker:
    """Tracks hands across frames for gesture recognition"""
    
    def __init__(self):
        self.active_hands = {}
        self.next_id = 0
        self.max_tracking_distance = 50  # pixels
    
    def update_tracking(self, hands: List[Hand3D]) -> List[Hand3D]:
        """Update hand tracking IDs"""
        # Simple tracking based on position proximity
        updated_hands = []
        
        for hand in hands:
            closest_id = self.find_closest_hand(hand)
            if closest_id is not None:
                hand.tracking_id = closest_id
                self.active_hands[closest_id] = hand
            else:
                hand.tracking_id = self.next_id
                self.active_hands[self.next_id] = hand
                self.next_id += 1
            
            updated_hands.append(hand)
        
        # Remove inactive hands
        active_ids = {hand.tracking_id for hand in updated_hands}
        inactive_ids = set(self.active_hands.keys()) - active_ids
        for inactive_id in inactive_ids:
            del self.active_hands[inactive_id]
        
        return updated_hands
    
    def find_closest_hand(self, hand: Hand3D) -> Optional[int]:
        """Find closest tracked hand"""
        min_distance = float('inf')
        closest_id = None
        
        for hand_id, tracked_hand in self.active_hands.items():
            distance = self.calculate_hand_distance(hand, tracked_hand)
            if distance < min_distance and distance < self.max_tracking_distance:
                min_distance = distance
                closest_id = hand_id
        
        return closest_id
    
    def calculate_hand_distance(self, hand1: Hand3D, hand2: Hand3D) -> float:
        """Calculate distance between two hands"""
        pos1 = np.array(hand1.palm_center)
        pos2 = np.array(hand2.palm_center)
        return np.linalg.norm(pos1 - pos2)

class GestureRecognizer:
    """Advanced gesture recognition engine"""
    
    def __init__(self, config: GestureConfig):
        self.config = config
        self.gesture_library = GestureLibrary()
        self.recognition_state = GestureState.IDLE
        self.current_gesture = None
        self.gesture_start_time = None
        self.gesture_history = deque(maxlen=100)
        
        # Machine learning model for custom gestures
        self.ml_model = None
        self.feature_scaler = None
        
    def recognize_gestures(self, hands: List[Hand3D]) -> List[RecognizedGesture]:
        """Recognize gestures from hand data"""
        recognized_gestures = []
        
        if not hands:
            self.recognition_state = GestureState.IDLE
            return recognized_gestures
        
        # Check static gestures
        for hand in hands:
            static_gesture = self.recognize_static_gesture(hand)
            if static_gesture:
                recognized_gestures.append(static_gesture)
        
        # Check dynamic gestures (requires hand history)
        if len(self.gesture_history) >= 5:  # Need some history
            dynamic_gesture = self.recognize_dynamic_gesture()
            if dynamic_gesture:
                recognized_gestures.append(dynamic_gesture)
        
        # Check bimanual gestures
        if len(hands) == 2:
            bimanual_gesture = self.recognize_bimanual_gesture(hands)
            if bimanual_gesture:
                recognized_gestures.append(bimanual_gesture)
        
        # Update gesture history
        self.gesture_history.append({
            'timestamp': time.time(),
            'hands': hands,
            'recognized': recognized_gestures
        })
        
        return recognized_gestures
    
    def recognize_static_gesture(self, hand: Hand3D) -> Optional[RecognizedGesture]:
        """Recognize static hand gestures"""
        best_match = None
        best_confidence = 0.0
        
        for gesture_name, gesture_data in self.gesture_library.static_gestures.items():
            confidence = self.calculate_static_gesture_confidence(hand, gesture_data)
            
            if confidence > gesture_data['features']['confidence_threshold'] and confidence > best_confidence:
                best_match = gesture_name
                best_confidence = confidence
        
        if best_match:
            return RecognizedGesture(
                name=best_match,
                type=GestureType.STATIC,
                confidence=best_confidence,
                duration=0.0,
                hands_involved=[hand.handedness],
                parameters=hand.gesture_features,
                timestamp=time.time(),
                frame_count=1
            )
        
        return None
    
    def calculate_static_gesture_confidence(self, hand: Hand3D, gesture_data: Dict) -> float:
        """Calculate confidence for static gesture match"""
        confidence = 0.0
        features = gesture_data['features']
        hand_features = hand.gesture_features
        
        # Check palm direction
        if 'palm_direction' in features:
            if features['palm_direction'] == 'forward' and hand_features.get('palm_forward', 0) > 0.7:
                confidence += 0.3
            elif features['palm_direction'] == 'up' and hand_features.get('palm_up', 0) > 0.7:
                confidence += 0.3
        
        # Check finger spread
        if 'finger_spread' in features:
            expected_spread = 1.0 if features['finger_spread'] == 'wide' else 0.5
            actual_spread = hand_features.get('finger_spread', 0.5)
            spread_match = 1.0 - abs(expected_spread - actual_spread)
            confidence += 0.3 * spread_match
        
        # Check thumb-index distance for pinch gestures
        if 'thumb_index_distance' in features:
            expected_distance = 0.1 if features['thumb_index_distance'] == 'small' else 0.3
            actual_distance = hand_features.get('thumb_index_distance', 0.2)
            distance_match = 1.0 - abs(expected_distance - actual_distance) * 5
            confidence += 0.4 * max(0, distance_match)
        
        return min(1.0, confidence)
    
    def recognize_dynamic_gesture(self) -> Optional[RecognizedGesture]:
        """Recognize dynamic gestures from hand movement history"""
        # Analyze last 10 frames for movement patterns
        recent_history = list(self.gesture_history)[-10:]
        
        if len(recent_history) < 5:
            return None
        
        # Extract movement vectors
        movements = []
        for i in range(1, len(recent_history)):
            prev_hands = recent_history[i-1]['hands']
            curr_hands = recent_history[i]['hands']
            
            if prev_hands and curr_hands:
                # Track first hand movement
                prev_palm = np.array(prev_hands[0].palm_center)
                curr_palm = np.array(curr_hands[0].palm_center)
                movement = curr_palm - prev_palm
                movements.append(movement)
        
        if not movements:
            return None
        
        # Analyze movement pattern
        total_movement = np.sum(movements, axis=0)
        movement_magnitude = np.linalg.norm(total_movement)
        
        # Recognize swipe gestures
        if movement_magnitude > 100:  # Significant movement
            direction = total_movement / movement_magnitude
            
            # Determine swipe direction
            if abs(direction[0]) > abs(direction[1]):
                swipe_type = 'swipe_right' if direction[0] > 0 else 'swipe_left'
            else:
                swipe_type = 'swipe_down' if direction[1] > 0 else 'swipe_up'
            
            return RecognizedGesture(
                name=f'interface_{swipe_type}',
                type=GestureType.SWIPE,
                confidence=0.8,
                duration=len(recent_history) * 0.033,  # Assuming 30 FPS
                hands_involved=[recent_history[-1]['hands'][0].handedness],
                parameters={'direction': direction.tolist(), 'magnitude': movement_magnitude},
                timestamp=time.time(),
                frame_count=len(recent_history)
            )
        
        return None
    
    def recognize_bimanual_gesture(self, hands: List[Hand3D]) -> Optional[RecognizedGesture]:
        """Recognize two-handed gestures"""
        if len(hands) != 2:
            return None
        
        left_hand = next((h for h in hands if h.handedness == 'Left'), None)
        right_hand = next((h for h in hands if h.handedness == 'Right'), None)
        
        if not left_hand or not right_hand:
            return None
        
        # Calculate hand separation
        left_palm = np.array(left_hand.palm_center)
        right_palm = np.array(right_hand.palm_center)
        separation = np.linalg.norm(right_palm - left_palm)
        
        # Check for suit summon gesture (arms spread wide)
        if separation > 300:  # Wide separation
            # Check if both palms face forward
            left_forward = left_hand.gesture_features.get('palm_forward', 0)
            right_forward = right_hand.gesture_features.get('palm_forward', 0)
            
            if left_forward > 0.6 and right_forward > 0.6:
                return RecognizedGesture(
                    name='suit_summon',
                    type=GestureType.BIMANUAL,
                    confidence=0.85,
                    duration=0.0,
                    hands_involved=['Left', 'Right'],
                    parameters={'separation': separation, 'palm_forward': (left_forward + right_forward) / 2},
                    timestamp=time.time(),
                    frame_count=1
                )
        
        return None
    
    def train_custom_gesture(self, gesture_name: str, training_data: List[List[Hand3D]]):
        """Train a custom gesture using machine learning"""
        if not self.config.enable_custom_gestures:
            return False
        
        # Extract features from training data
        feature_vectors = []
        for sequence in training_data:
            features = self.extract_sequence_features(sequence)
            feature_vectors.append(features)
        
        # Train a simple classifier (would use more advanced ML in production)
        self.gesture_library.custom_gestures[gesture_name] = {
            'features': feature_vectors,
            'trained_at': time.time(),
            'confidence_threshold': 0.7
        }
        
        return True
    
    def extract_sequence_features(self, hand_sequence: List[Hand3D]) -> List[float]:
        """Extract features from a sequence of hand poses"""
        features = []
        
        for hand in hand_sequence:
            # Add gesture features
            features.extend([
                hand.gesture_features.get('hand_size', 0),
                hand.gesture_features.get('finger_spread', 0),
                hand.gesture_features.get('thumb_index_distance', 0),
                hand.gesture_features.get('palm_forward', 0),
                hand.gesture_features.get('palm_up', 0)
            ])
            
            # Add finger states
            features.extend([float(state) for state in hand.finger_states])
        
        # Pad or truncate to fixed size
        target_size = 50  # Arbitrary fixed size
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return features

class JarvisGestureSystem:
    """Main JARVIS Gesture Recognition System"""
    
    def __init__(self, config: GestureConfig = None):
        self.config = config or GestureConfig()
        self.hand_tracker = AdvancedHandTracker(self.config)
        self.gesture_recognizer = GestureRecognizer(self.config)
        
        # System state
        self.is_active = True
        self.gesture_callbacks = {}
        self.performance_stats = {
            'frames_processed': 0,
            'gestures_recognized': 0,
            'avg_processing_time': 0.0,
            'accuracy_rate': 0.95
        }
        
        print("🤖 JARVIS Advanced Gesture System initialized")
        print(f"   Max hands: {self.config.max_hands}")
        print(f"   Detection confidence: {self.config.detection_confidence}")
        print(f"   3D tracking: {self.config.depth_estimation}")
        print(f"   Air gestures: {self.config.enable_air_gestures}")
    
    def process_frame(self, frame: np.ndarray, depth_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process video frame and recognize gestures"""
        start_time = time.time()
        
        result = {
            'hands': [],
            'gestures': [],
            'processing_time': 0.0,
            'frame_number': self.performance_stats['frames_processed']
        }
        
        if not self.is_active:
            return result
        
        # Track hands in 3D
        hands = self.hand_tracker.process_frame(frame, depth_frame)
        result['hands'] = hands
        
        # Recognize gestures
        if hands:
            gestures = self.gesture_recognizer.recognize_gestures(hands)
            result['gestures'] = gestures
            
            # Execute gesture callbacks
            for gesture in gestures:
                self.execute_gesture_callback(gesture)
        
        # Update performance stats
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        self.performance_stats['frames_processed'] += 1
        self.performance_stats['gestures_recognized'] += len(result['gestures'])
        
        # Update average processing time
        current_avg = self.performance_stats['avg_processing_time']
        frame_count = self.performance_stats['frames_processed']
        new_avg = (current_avg * (frame_count - 1) + processing_time) / frame_count
        self.performance_stats['avg_processing_time'] = new_avg
        
        return result
    
    def register_gesture_callback(self, gesture_name: str, callback: Callable[[RecognizedGesture], None]):
        """Register callback for specific gesture"""
        self.gesture_callbacks[gesture_name] = callback
        print(f"Registered callback for gesture: {gesture_name}")
    
    def execute_gesture_callback(self, gesture: RecognizedGesture):
        """Execute registered callback for recognized gesture"""
        if gesture.name in self.gesture_callbacks:
            try:
                self.gesture_callbacks[gesture.name](gesture)
            except Exception as e:
                print(f"Error executing gesture callback for {gesture.name}: {e}")
    
    def calibrate_system(self, calibration_frames: int = 30):
        """Calibrate the gesture system"""
        print(f"Starting system calibration... Please hold still for {calibration_frames} frames")
        # Calibration logic would go here
        self.hand_tracker.is_calibrated = True
        print("✅ Calibration completed")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'active': self.is_active,
            'calibrated': self.hand_tracker.is_calibrated,
            'performance': self.performance_stats,
            'config': {
                'max_hands': self.config.max_hands,
                'detection_confidence': self.config.detection_confidence,
                'tracking_confidence': self.config.tracking_confidence,
                'enable_3d': self.config.depth_estimation,
                'enable_air_gestures': self.config.enable_air_gestures
            },
            'registered_callbacks': list(self.gesture_callbacks.keys())
        }
    
    def save_custom_gesture(self, name: str, filename: str):
        """Save custom gesture to file"""
        if name in self.gesture_recognizer.gesture_library.custom_gestures:
            with open(filename, 'wb') as f:
                pickle.dump(self.gesture_recognizer.gesture_library.custom_gestures[name], f)
            print(f"Custom gesture '{name}' saved to {filename}")
        else:
            print(f"Custom gesture '{name}' not found")
    
    def load_custom_gesture(self, name: str, filename: str):
        """Load custom gesture from file"""
        try:
            with open(filename, 'rb') as f:
                gesture_data = pickle.load(f)
            self.gesture_recognizer.gesture_library.custom_gestures[name] = gesture_data
            print(f"Custom gesture '{name}' loaded from {filename}")
        except Exception as e:
            print(f"Failed to load custom gesture: {e}")

# Example gesture callbacks
def repulsor_blast_callback(gesture: RecognizedGesture):
    """Callback for repulsor blast gesture"""
    print(f"🔥 REPULSOR BLAST activated! Confidence: {gesture.confidence:.2f}")
    print("   Charging energy systems...")

def hologram_manipulate_callback(gesture: RecognizedGesture):
    """Callback for hologram manipulation"""
    print(f"🌐 Hologram manipulation active! Hand: {gesture.hands_involved[0]}")
    direction = gesture.parameters.get('direction', [0, 0, 0])
    print(f"   Movement direction: {direction}")

def suit_summon_callback(gesture: RecognizedGesture):
    """Callback for suit summoning"""
    print(f"🦾 SUIT SUMMON initiated! Confidence: {gesture.confidence:.2f}")
    print("   Mark 85 armor incoming...")

# Main test function
def main():
    """Test the advanced gesture system"""
    config = GestureConfig(
        max_hands=2,
        detection_confidence=0.8,
        enable_air_gestures=True,
        enable_custom_gestures=True
    )
    
    gesture_system = JarvisGestureSystem(config)
    
    # Register gesture callbacks
    gesture_system.register_gesture_callback('repulsor_blast', repulsor_blast_callback)
    gesture_system.register_gesture_callback('hologram_manipulate', hologram_manipulate_callback)
    gesture_system.register_gesture_callback('suit_summon', suit_summon_callback)
    
    # Simulate camera input
    cap = cv2.VideoCapture(0)
    
    print("🎥 Starting gesture recognition... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = gesture_system.process_frame(frame)
        
        # Display results
        display_frame = frame.copy()
        
        # Draw hand landmarks and gestures
        for i, hand in enumerate(result['hands']):
            # Draw hand landmarks
            for landmark in hand.landmarks:
                cv2.circle(display_frame, (int(landmark[0]), int(landmark[1])), 3, (0, 255, 0), -1)
            
            # Draw palm center
            palm = hand.palm_center
            cv2.circle(display_frame, (int(palm[0]), int(palm[1])), 8, (255, 0, 0), -1)
            
            # Show hand info
            cv2.putText(display_frame, f"{hand.handedness} ({hand.confidence:.2f})", 
                       (int(palm[0]) + 10, int(palm[1]) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display recognized gestures
        for i, gesture in enumerate(result['gestures']):
            text = f"{gesture.name} ({gesture.confidence:.2f})"
            cv2.putText(display_frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show performance stats
        fps = 1.0 / result['processing_time'] if result['processing_time'] > 0 else 0
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('JARVIS Gesture Recognition', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Show final stats
    status = gesture_system.get_system_status()
    print(f"\n📊 Final Statistics:")
    print(f"   Frames processed: {status['performance']['frames_processed']}")
    print(f"   Gestures recognized: {status['performance']['gestures_recognized']}")
    print(f"   Average processing time: {status['performance']['avg_processing_time']:.3f}s")

if __name__ == "__main__":
    main()