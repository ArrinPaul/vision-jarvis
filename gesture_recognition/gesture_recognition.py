import cv2
import mediapipe as mp
import numpy as np
import json
import threading
import time
from datetime import datetime
from .custom_gestures import CustomGestureTrainer

class GestureRecognizer:
    """
    Advanced Gesture Recognition supporting multi-hand, dynamic, and complex gestures using MediaPipe.
    Features: Custom gesture training, gesture-to-action mapping, and advanced hand tracking.
    """
    def __init__(self, config_file="gesture_config.json"):
        # Initialize MediaPipe
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        
        # Custom gesture trainer
        self.custom_trainer = CustomGestureTrainer()
        self.custom_trainer.load_model()
        
        # Configuration
        self.config_file = config_file
        self.gesture_actions = {}
        self.gesture_history = []
        self.gesture_sequences = {}
        
        # State tracking
        self.is_recognizing = False
        self.last_gesture = None
        self.gesture_stability_count = 0
        self.required_stability = 5  # frames
        
        # Load configuration
        self.load_configuration()
        
        # Built-in gestures
        self.builtin_gestures = {
            'peace': self._detect_peace_sign,
            'thumbs_up': self._detect_thumbs_up,
            'thumbs_down': self._detect_thumbs_down,
            'ok_sign': self._detect_ok_sign,
            'pointing': self._detect_pointing,
            'fist': self._detect_fist,
            'open_palm': self._detect_open_palm,
            'rock': self._detect_rock,
            'paper': self._detect_paper,
            'scissors': self._detect_scissors
        }
        
    def load_configuration(self):
        """Load gesture configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.gesture_actions = config.get('gesture_actions', {})
                self.gesture_sequences = config.get('gesture_sequences', {})
        except FileNotFoundError:
            print("No gesture configuration found. Starting with defaults.")
            self._create_default_config()
        except Exception as e:
            print(f"Error loading gesture configuration: {e}")
            
    def save_configuration(self):
        """Save gesture configuration to file."""
        try:
            config = {
                'gesture_actions': self.gesture_actions,
                'gesture_sequences': self.gesture_sequences,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving gesture configuration: {e}")
            
    def _create_default_config(self):
        """Create default gesture-to-action mappings."""
        self.gesture_actions = {
            'thumbs_up': {'action': 'volume_up', 'description': 'Increase volume'},
            'thumbs_down': {'action': 'volume_down', 'description': 'Decrease volume'},
            'peace': {'action': 'take_screenshot', 'description': 'Take screenshot'},
            'ok_sign': {'action': 'confirm_action', 'description': 'Confirm action'},
            'pointing': {'action': 'select_item', 'description': 'Select pointed item'},
            'fist': {'action': 'stop_action', 'description': 'Stop current action'},
            'open_palm': {'action': 'pause_action', 'description': 'Pause current action'}
        }
        self.save_configuration()
        
    def start_recognition(self, callback=None):
        """Start gesture recognition in a separate thread."""
        if self.is_recognizing:
            return
            
        self.is_recognizing = True
        threading.Thread(target=self._recognition_loop, args=(callback,), daemon=True).start()
        
    def stop_recognition(self):
        """Stop gesture recognition."""
        self.is_recognizing = False
        
    def _recognition_loop(self, callback=None):
        """Main gesture recognition loop."""
        cap = cv2.VideoCapture(0)
        
        try:
            while self.is_recognizing:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                gestures_detected = self.process_frame(frame)
                
                # Execute callback if provided
                if callback:
                    callback(frame, gestures_detected)
                    
                # Display frame
                cv2.imshow("JARVIS Gesture Recognition", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self._enter_training_mode()
                    
        except Exception as e:
            print(f"Gesture recognition error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.is_recognizing = False
            
    def process_frame(self, frame):
        """Process a single frame for gesture detection."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected_gestures = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get hand classification (left/right)
                hand_label = "Unknown"
                if results.multi_handedness:
                    hand_label = results.multi_handedness[idx].classification[0].label
                    
                # Detect built-in gestures
                for gesture_name, detector_func in self.builtin_gestures.items():
                    if detector_func(hand_landmarks):
                        detected_gestures.append({
                            'name': gesture_name,
                            'hand': hand_label,
                            'confidence': 0.9,
                            'landmarks': hand_landmarks,
                            'type': 'builtin'
                        })
                        
                # Detect custom gestures
                custom_gesture, confidence = self.custom_trainer.predict_gesture(hand_landmarks)
                if custom_gesture and confidence > 0.7:
                    detected_gestures.append({
                        'name': custom_gesture,
                        'hand': hand_label,
                        'confidence': confidence,
                        'landmarks': hand_landmarks,
                        'type': 'custom'
                    })
                    
        # Process detected gestures
        self._process_detected_gestures(detected_gestures)
        
        return detected_gestures
        
    def _process_detected_gestures(self, gestures):
        """Process detected gestures and execute actions."""
        for gesture in gestures:
            gesture_name = gesture['name']
            
            # Check for stability (same gesture detected multiple times)
            if gesture_name == self.last_gesture:
                self.gesture_stability_count += 1
            else:
                self.last_gesture = gesture_name
                self.gesture_stability_count = 1
                
            # Execute action if gesture is stable
            if self.gesture_stability_count >= self.required_stability:
                self._execute_gesture_action(gesture)
                self.gesture_stability_count = 0  # Reset to prevent repeated execution
                
    def _execute_gesture_action(self, gesture):
        """Execute action associated with gesture."""
        gesture_name = gesture['name']
        
        if gesture_name in self.gesture_actions:
            action_info = self.gesture_actions[gesture_name]
            action = action_info['action']
            
            print(f"Executing action '{action}' for gesture '{gesture_name}'")
            
            # Log gesture
            self.gesture_history.append({
                'gesture': gesture_name,
                'action': action,
                'timestamp': datetime.now().isoformat(),
                'confidence': gesture['confidence']
            })
            
            # Execute the actual action (this would interface with other systems)
            self._perform_action(action, gesture)
            
    def _perform_action(self, action, gesture):
        """Perform the actual action (interface with other systems)."""
        # This would interface with other JARVIS modules
        if action == 'volume_up':
            print("Increasing volume...")
        elif action == 'volume_down':
            print("Decreasing volume...")
        elif action == 'take_screenshot':
            print("Taking screenshot...")
        elif action == 'confirm_action':
            print("Action confirmed...")
        elif action == 'select_item':
            print("Item selected...")
        elif action == 'stop_action':
            print("Stopping current action...")
        elif action == 'pause_action':
            print("Pausing current action...")
        else:
            print(f"Unknown action: {action}")
            
    # Built-in gesture detection methods
    def _detect_thumbs_up(self, landmarks):
        """Detect thumbs up gesture."""
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        index_mcp = landmarks.landmark[5]
        
        # Thumb should be extended upward
        thumb_up = thumb_tip.y < thumb_ip.y < index_mcp.y
        
        # Other fingers should be folded
        fingers_folded = all(
            landmarks.landmark[tip].y > landmarks.landmark[tip-2].y
            for tip in [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        )
        
        return thumb_up and fingers_folded
        
    def _detect_thumbs_down(self, landmarks):
        """Detect thumbs down gesture."""
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        index_mcp = landmarks.landmark[5]
        
        # Thumb should be extended downward
        thumb_down = thumb_tip.y > thumb_ip.y > index_mcp.y
        
        # Other fingers should be folded
        fingers_folded = all(
            landmarks.landmark[tip].y > landmarks.landmark[tip-2].y
            for tip in [8, 12, 16, 20]
        )
        
        return thumb_down and fingers_folded
        
    def _detect_peace_sign(self, landmarks):
        """Detect peace sign (V) gesture."""
        # Index and middle fingers should be extended
        index_extended = landmarks.landmark[8].y < landmarks.landmark[6].y
        middle_extended = landmarks.landmark[12].y < landmarks.landmark[10].y
        
        # Ring and pinky should be folded
        ring_folded = landmarks.landmark[16].y > landmarks.landmark[14].y
        pinky_folded = landmarks.landmark[20].y > landmarks.landmark[18].y
        
        # Thumb should be folded
        thumb_folded = landmarks.landmark[4].x < landmarks.landmark[3].x
        
        return index_extended and middle_extended and ring_folded and pinky_folded and thumb_folded
        
    def _detect_ok_sign(self, landmarks):
        """Detect OK sign gesture."""
        # Thumb and index finger should be close (forming circle)
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        
        distance = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        
        fingers_circle = distance < 0.05
        
        # Other fingers should be extended
        middle_extended = landmarks.landmark[12].y < landmarks.landmark[10].y
        ring_extended = landmarks.landmark[16].y < landmarks.landmark[14].y
        pinky_extended = landmarks.landmark[20].y < landmarks.landmark[18].y
        
        return fingers_circle and middle_extended and ring_extended and pinky_extended
        
    def _detect_pointing(self, landmarks):
        """Detect pointing gesture."""
        # Only index finger should be extended
        index_extended = landmarks.landmark[8].y < landmarks.landmark[6].y
        
        # Other fingers should be folded
        middle_folded = landmarks.landmark[12].y > landmarks.landmark[10].y
        ring_folded = landmarks.landmark[16].y > landmarks.landmark[14].y
        pinky_folded = landmarks.landmark[20].y > landmarks.landmark[18].y
        thumb_folded = landmarks.landmark[4].y > landmarks.landmark[3].y
        
        return index_extended and middle_folded and ring_folded and pinky_folded and thumb_folded
        
    def _detect_fist(self, landmarks):
        """Detect closed fist gesture."""
        # All fingers should be folded
        fingers_folded = all(
            landmarks.landmark[tip].y > landmarks.landmark[tip-2].y
            for tip in [8, 12, 16, 20]
        )
        
        thumb_folded = landmarks.landmark[4].x < landmarks.landmark[3].x
        
        return fingers_folded and thumb_folded
        
    def _detect_open_palm(self, landmarks):
        """Detect open palm gesture."""
        # All fingers should be extended
        fingers_extended = all(
            landmarks.landmark[tip].y < landmarks.landmark[tip-2].y
            for tip in [8, 12, 16, 20]
        )
        
        thumb_extended = landmarks.landmark[4].x > landmarks.landmark[3].x
        
        return fingers_extended and thumb_extended
        
    def _detect_rock(self, landmarks):
        """Detect rock gesture (same as fist)."""
        return self._detect_fist(landmarks)
        
    def _detect_paper(self, landmarks):
        """Detect paper gesture (same as open palm)."""
        return self._detect_open_palm(landmarks)
        
    def _detect_scissors(self, landmarks):
        """Detect scissors gesture (same as peace sign)."""
        return self._detect_peace_sign(landmarks)
        
    def _enter_training_mode(self):
        """Enter custom gesture training mode."""
        print("Entering gesture training mode...")
        print("Hold the gesture you want to train and press 'c' to capture")
        print("Press 'q' to exit training mode")
        
        gesture_name = input("Enter gesture name: ")
        samples_collected = 0
        target_samples = 10
        
        cap = cv2.VideoCapture(0)
        
        while samples_collected < target_samples:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Display instructions
            cv2.putText(frame, f"Training: {gesture_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {samples_collected}/{target_samples}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
            cv2.imshow("Gesture Training", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and results.multi_hand_landmarks:
                # Capture training sample
                self.custom_trainer.add_training_sample(results.multi_hand_landmarks[0], gesture_name)
                samples_collected += 1
                print(f"Captured sample {samples_collected}/{target_samples}")
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyWindow("Gesture Training")
        
        if samples_collected >= target_samples:
            print("Training model...")
            if self.custom_trainer.train_model():
                print(f"Successfully trained gesture: {gesture_name}")
            else:
                print("Training failed")
        else:
            print("Training cancelled - not enough samples")
            
    def add_gesture_action(self, gesture_name, action, description=""):
        """Add gesture-to-action mapping."""
        self.gesture_actions[gesture_name] = {
            'action': action,
            'description': description
        }
        self.save_configuration()
        print(f"Added action mapping: {gesture_name} -> {action}")
        
    def get_gesture_stats(self):
        """Get gesture recognition statistics."""
        total_gestures = len(self.gesture_history)
        recent_gestures = [g for g in self.gesture_history 
                          if (datetime.now() - datetime.fromisoformat(g['timestamp'])).seconds < 3600]
        
        return {
            'total_gestures': total_gestures,
            'recent_gestures': len(recent_gestures),
            'available_gestures': list(self.builtin_gestures.keys()) + list(self.custom_trainer.gesture_names.keys()),
            'gesture_actions': len(self.gesture_actions)
        }
        
    def detect_gesture(self):
        """Legacy method for backward compatibility."""
        self.start_recognition()
        
    def train_custom_gesture(self, gesture_name):
        """Legacy method for backward compatibility."""
        self._enter_training_mode()
