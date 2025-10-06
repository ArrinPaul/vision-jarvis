"""
Enhanced Biometric Authentication and Security Features for JARVIS User Profiles.
Provides advanced face recognition, voice print authentication, behavioral biometrics,
and comprehensive security features with fallback methods.
"""

import numpy as np
import cv2
import hashlib
import pickle
import os
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from collections import defaultdict

# Optional imports for enhanced features
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    face_recognition = None

try:
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    Fernet = None

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SVC = None
    StandardScaler = None

class BiometricAuthenticator:
    """
    Advanced multi-modal biometric authentication system for secure user identification.
    Supports face recognition, voice authentication, behavioral patterns, and more.
    """
    
    def __init__(self, data_dir="biometric_data"):
        self.data_dir = data_dir
        self.face_encodings = {}
        self.voice_prints = {}
        self.behavioral_models = {}
        
        # Authentication settings
        self.face_threshold = 0.6
        self.voice_threshold = 0.8
        self.behavior_threshold = 0.7
        
        # Security settings
        self.max_attempts = 3
        self.lockout_duration = 300  # 5 minutes
        self.failed_attempts = {}
        self.locked_users = {}
        
        # Initialize directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/faces", exist_ok=True)
        os.makedirs(f"{data_dir}/voices", exist_ok=True)
        os.makedirs(f"{data_dir}/behaviors", exist_ok=True)
        
        # Load existing data
        self._load_biometric_data()
        
        logging.info("Enhanced Biometric Authenticator initialized")
        
    def enroll_user_biometrics(self, user_id, face_image=None, voice_sample=None, behavior_data=None):
        """Enroll user biometrics for authentication."""
        enrollment_results = {
            "user_id": user_id,
            "face_enrolled": False,
            "voice_enrolled": False,
            "behavior_enrolled": False,
            "enrollment_timestamp": datetime.now().isoformat()
        }
        
        # Enroll face
        if face_image is not None:
            if self._enroll_face(user_id, face_image):
                enrollment_results["face_enrolled"] = True
                logging.info(f"Face enrolled for user {user_id}")
            else:
                logging.warning(f"Face enrollment failed for user {user_id}")
                
        # Enroll voice
        if voice_sample is not None:
            if self._enroll_voice(user_id, voice_sample):
                enrollment_results["voice_enrolled"] = True
                logging.info(f"Voice enrolled for user {user_id}")
            else:
                logging.warning(f"Voice enrollment failed for user {user_id}")
                
        # Enroll behavioral patterns
        if behavior_data is not None:
            if self._enroll_behavior(user_id, behavior_data):
                enrollment_results["behavior_enrolled"] = True
                logging.info(f"Behavioral patterns enrolled for user {user_id}")
                
        self._save_biometric_data()
        return enrollment_results
        
    def authenticate_user(self, user_id=None, face_image=None, voice_sample=None, behavior_data=None):
        """
        Multi-modal biometric authentication.
        Returns authentication result and confidence scores.
        """
        # Check if user is locked out
        if user_id and self._is_user_locked(user_id):
            return {
                "authenticated": False,
                "reason": "user_locked",
                "lockout_remaining": self._get_lockout_remaining(user_id)
            }
            
        authentication_results = {
            "authenticated": False,
            "user_id": user_id,
            "confidence_scores": {},
            "method_results": {},
            "overall_confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        method_scores = []
        
        # Face authentication
        if face_image is not None:
            face_result = self._authenticate_face(user_id, face_image)
            authentication_results["method_results"]["face"] = face_result
            authentication_results["confidence_scores"]["face"] = face_result["confidence"]
            
            if face_result["success"]:
                method_scores.append(face_result["confidence"])
                
        # Voice authentication
        if voice_sample is not None:
            voice_result = self._authenticate_voice(user_id, voice_sample)
            authentication_results["method_results"]["voice"] = voice_result
            authentication_results["confidence_scores"]["voice"] = voice_result["confidence"]
            
            if voice_result["success"]:
                method_scores.append(voice_result["confidence"])
                
        # Behavioral authentication
        if behavior_data is not None:
            behavior_result = self._authenticate_behavior(user_id, behavior_data)
            authentication_results["method_results"]["behavior"] = behavior_result
            authentication_results["confidence_scores"]["behavior"] = behavior_result["confidence"]
            
            if behavior_result["success"]:
                method_scores.append(behavior_result["confidence"])
                
        # Calculate overall authentication
        if method_scores:
            authentication_results["overall_confidence"] = np.mean(method_scores)
            
            # Multi-modal fusion - require at least one strong authentication
            if authentication_results["overall_confidence"] > 0.7 or any(score > 0.8 for score in method_scores):
                authentication_results["authenticated"] = True
                self._reset_failed_attempts(user_id)
            else:
                self._record_failed_attempt(user_id)
                authentication_results["reason"] = "insufficient_confidence"
        else:
            authentication_results["reason"] = "no_biometric_data"
            
        return authentication_results
        
    def _enroll_face(self, user_id, face_image):
        """Enroll face biometric with enhanced fallback support."""
        try:
            if FACE_RECOGNITION_AVAILABLE:
                return self._enroll_face_advanced(user_id, face_image)
            else:
                logging.warning("Face recognition library not available - using enhanced fallback method")
                return self._enroll_face_fallback(user_id, face_image)
        except Exception as e:
            logging.error(f"Face enrollment error: {e}")
            return False
            
    def _enroll_face_advanced(self, user_id, face_image):
        """Advanced face enrollment using face_recognition library."""
        try:
            # Convert to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
                
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return False
                
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                return False
                
            # Store multiple samples for better accuracy
            if user_id not in self.face_encodings:
                self.face_encodings[user_id] = {
                    "encodings": [],
                    "enrollment_date": datetime.now().isoformat(),
                    "face_locations": []
                }
            
            self.face_encodings[user_id]["encodings"].append(face_encodings[0])
            self.face_encodings[user_id]["face_locations"].append(face_locations[0])
            
            # Limit to 5 samples per user
            if len(self.face_encodings[user_id]["encodings"]) > 5:
                self.face_encodings[user_id]["encodings"].pop(0)
                self.face_encodings[user_id]["face_locations"].pop(0)
            
            # Save face image
            face_path = f"{self.data_dir}/faces/{user_id}_face_{len(self.face_encodings[user_id]['encodings'])}.jpg"
            cv2.imwrite(face_path, face_image)
            
            return True
            
        except Exception as e:
            logging.error(f"Advanced face enrollment error: {e}")
            return False
    
    def _enroll_face_fallback(self, user_id, face_image):
        """Enhanced fallback face enrollment using OpenCV with multiple feature extraction methods."""
        try:
            # Initialize cascade classifiers
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return False
            
            # Extract multiple features for better recognition
            x, y, w, h = faces[0]
            face_region = gray[y:y+h, x:x+w]
            
            # Feature 1: Resized face region
            face_features = cv2.resize(face_region, (100, 100)).flatten()
            
            # Feature 2: LBP (Local Binary Pattern) features
            lbp_features = self._calculate_lbp_features(face_region)
            
            # Feature 3: Histogram of oriented gradients (HOG)
            hog_features = self._calculate_hog_features(face_region)
            
            # Feature 4: Eye detection for additional verification
            eyes = eye_cascade.detectMultiScale(face_region)
            eye_features = np.array([len(eyes), eyes[0][2] if len(eyes) > 0 else 0, eyes[0][3] if len(eyes) > 0 else 0])
            
            # Combine all features
            combined_features = np.concatenate([face_features, lbp_features, hog_features, eye_features])
            
            # Store multiple enrollment samples for better matching
            if user_id not in self.face_encodings:
                self.face_encodings[user_id] = {
                    "samples": [],
                    "enrollment_date": datetime.now().isoformat(),
                    "face_dimensions": (w, h)
                }
            
            self.face_encodings[user_id]["samples"].append(combined_features)
            
            # Limit to 5 samples per user
            if len(self.face_encodings[user_id]["samples"]) > 5:
                self.face_encodings[user_id]["samples"].pop(0)
            
            return True
            
        except Exception as e:
            logging.error(f"Enhanced face enrollment fallback error: {e}")
            return False
    
    def _authenticate_face(self, user_id, face_image):
        """Authenticate using face recognition with fallback support."""
        if user_id not in self.face_encodings:
            return {"success": False, "confidence": 0.0, "reason": "no_enrollment"}
            
        try:
            if FACE_RECOGNITION_AVAILABLE and "encodings" in self.face_encodings[user_id]:
                return self._authenticate_face_advanced(user_id, face_image)
            else:
                return self._authenticate_face_fallback(user_id, face_image)
        except Exception as e:
            logging.error(f"Face authentication error: {e}")
            return {"success": False, "confidence": 0.0, "reason": "authentication_error"}
    
    def _authenticate_face_advanced(self, user_id, face_image):
        """Advanced face authentication using face_recognition library."""
        try:
            # Convert to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
                
            # Find faces
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return {"success": False, "confidence": 0.0, "reason": "no_face_detected"}
                
            # Get encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                return {"success": False, "confidence": 0.0, "reason": "no_face_encoding"}
                
            # Compare with all enrolled encodings
            enrolled_encodings = self.face_encodings[user_id]["encodings"]
            face_distances = face_recognition.face_distance(enrolled_encodings, face_encodings[0])
            
            # Use best match
            best_distance = min(face_distances)
            confidence = max(0.0, 1.0 - best_distance)
            success = confidence >= self.face_threshold
            
            return {
                "success": success,
                "confidence": confidence,
                "distance": best_distance,
                "threshold": self.face_threshold,
                "samples_compared": len(enrolled_encodings)
            }
            
        except Exception as e:
            logging.error(f"Advanced face authentication error: {e}")
            return {"success": False, "confidence": 0.0, "reason": "authentication_error"}
    
    def _authenticate_face_fallback(self, user_id, face_image):
        """Enhanced fallback face verification using multiple feature matching."""
        try:
            if "samples" not in self.face_encodings[user_id]:
                return {"success": False, "confidence": 0.0, "reason": "no_fallback_enrollment"}
            
            # Extract face features using same method as enrollment
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {"success": False, "confidence": 0.0, "reason": "no_face_detected"}
            
            x, y, w, h = faces[0]
            face_region = gray[y:y+h, x:x+w]
            
            # Extract same features as enrollment
            face_features = cv2.resize(face_region, (100, 100)).flatten()
            lbp_features = self._calculate_lbp_features(face_region)
            hog_features = self._calculate_hog_features(face_region)
            
            eyes = eye_cascade.detectMultiScale(face_region)
            eye_features = np.array([len(eyes), eyes[0][2] if len(eyes) > 0 else 0, eyes[0][3] if len(eyes) > 0 else 0])
            
            combined_features = np.concatenate([face_features, lbp_features, hog_features, eye_features])
            
            # Compare with all stored samples
            confidences = []
            stored_samples = self.face_encodings[user_id]["samples"]
            
            for stored_features in stored_samples:
                if len(combined_features) != len(stored_features):
                    continue
                    
                # Multiple similarity metrics
                correlation = np.corrcoef(combined_features, stored_features)[0, 1]
                cosine_sim = np.dot(combined_features, stored_features) / (np.linalg.norm(combined_features) * np.linalg.norm(stored_features) + 1e-8)
                euclidean_dist = np.linalg.norm(combined_features - stored_features)
                
                # Normalize and combine metrics
                correlation_score = max(0, (correlation + 1) / 2) if not np.isnan(correlation) else 0
                cosine_score = max(0, (cosine_sim + 1) / 2)
                euclidean_score = max(0, 1 - (euclidean_dist / (np.linalg.norm(combined_features) + 1e-8)))
                
                # Weighted average of metrics
                combined_confidence = (correlation_score * 0.4 + cosine_score * 0.4 + euclidean_score * 0.2)
                confidences.append(combined_confidence)
            
            # Use best match from all samples
            best_confidence = max(confidences) if confidences else 0.0
            success = best_confidence > 0.7
            
            # Additional security check - require at least 2 samples to match reasonably well
            good_matches = sum(1 for conf in confidences if conf > 0.6)
            if len(stored_samples) >= 2 and good_matches < 2:
                success = False
                best_confidence *= 0.5  # Reduce confidence for security
            
            return {"success": success, "confidence": best_confidence, "samples_matched": good_matches}
            
        except Exception as e:
            logging.error(f"Enhanced face verification fallback error: {e}")
            return {"success": False, "confidence": 0.0, "reason": str(e)}
    
    def _calculate_lbp_features(self, image):
        """Calculate Local Binary Pattern features."""
        try:
            # Simple LBP implementation
            rows, cols = image.shape
            if rows < 3 or cols < 3:
                return np.zeros(256, dtype=np.float32)
                
            lbp_image = np.zeros((rows-2, cols-2), dtype=np.uint8)
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = image[i, j]
                    binary_string = ''
                    
                    # Check 8 neighbors
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += '1' if neighbor >= center else '0'
                    
                    lbp_image[i-1, j-1] = int(binary_string, 2)
            
            # Calculate histogram of LBP values
            hist, _ = np.histogram(lbp_image.flatten(), bins=256, range=(0, 256))
            return hist.astype(np.float32) / (np.sum(hist) + 1e-8)  # Normalize
            
        except Exception as e:
            logging.error(f"LBP calculation error: {e}")
            return np.zeros(256, dtype=np.float32)
    
    def _calculate_hog_features(self, image):
        """Calculate Histogram of Oriented Gradients features."""
        try:
            # Simple HOG implementation
            gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            magnitude = np.sqrt(gx**2 + gy**2)
            orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
            
            # Create HOG descriptor (simplified)
            cell_size = 8
            n_bins = 9
            rows, cols = image.shape
            
            if rows < cell_size or cols < cell_size:
                return np.zeros(n_bins, dtype=np.float32)
            
            hog_features = []
            for i in range(0, rows - cell_size, cell_size):
                for j in range(0, cols - cell_size, cell_size):
                    cell_mag = magnitude[i:i+cell_size, j:j+cell_size]
                    cell_ori = orientation[i:i+cell_size, j:j+cell_size]
                    
                    hist, _ = np.histogram(cell_ori.flatten(), bins=n_bins, 
                                         range=(0, 180), weights=cell_mag.flatten())
                    hog_features.extend(hist)
            
            if not hog_features:
                return np.zeros(n_bins, dtype=np.float32)
                
            hog_array = np.array(hog_features, dtype=np.float32)
            return hog_array / (np.linalg.norm(hog_array) + 1e-8)  # Normalize
            
        except Exception as e:
            logging.error(f"HOG calculation error: {e}")
            return np.zeros(100, dtype=np.float32)  # Default size
            
    def _enroll_voice(self, user_id, voice_sample):
        """Enhanced voice enrollment with feature extraction."""
        try:
            # Extract voice features (MFCC, pitch, etc.)
            voice_features = self._extract_voice_features(voice_sample)
            
            if voice_features is None:
                return False
                
            if user_id not in self.voice_prints:
                self.voice_prints[user_id] = {
                    "samples": [],
                    "enrollment_date": datetime.now().isoformat()
                }
            
            self.voice_prints[user_id]["samples"].append({
                "features": voice_features,
                "sample_duration": len(voice_sample) / 16000  # Assuming 16kHz sample rate
            })
            
            # Limit to 3 voice samples per user
            if len(self.voice_prints[user_id]["samples"]) > 3:
                self.voice_prints[user_id]["samples"].pop(0)
            
            return True
            
        except Exception as e:
            logging.error(f"Voice enrollment error: {e}")
            return False
    
    def _authenticate_voice(self, user_id, voice_sample):
        """Enhanced voice authentication."""
        if user_id not in self.voice_prints:
            return {"success": False, "confidence": 0.0, "reason": "no_enrollment"}
            
        try:
            # Extract features from voice sample
            sample_features = self._extract_voice_features(voice_sample)
            
            if sample_features is None:
                return {"success": False, "confidence": 0.0, "reason": "feature_extraction_failed"}
            
            # Compare with all enrolled voice samples
            confidences = []
            for voice_sample_data in self.voice_prints[user_id]["samples"]:
                enrolled_features = voice_sample_data["features"]
                confidence = self._calculate_voice_similarity(enrolled_features, sample_features)
                confidences.append(confidence)
            
            # Use best match
            best_confidence = max(confidences) if confidences else 0.0
            success = best_confidence >= self.voice_threshold
            
            return {
                "success": success,
                "confidence": best_confidence,
                "threshold": self.voice_threshold,
                "samples_compared": len(confidences)
            }
            
        except Exception as e:
            logging.error(f"Voice authentication error: {e}")
            return {"success": False, "confidence": 0.0, "reason": "authentication_error"}
    
    def _extract_voice_features(self, voice_sample):
        """Extract voice features for authentication."""
        try:
            # Placeholder implementation - in a real system, you'd use librosa or similar
            # to extract MFCC, pitch, spectral features, etc.
            
            # For demonstration, we'll create pseudo-features based on sample statistics
            if len(voice_sample) == 0:
                return None
            
            # Basic statistical features
            mean_val = np.mean(voice_sample)
            std_val = np.std(voice_sample)
            max_val = np.max(voice_sample)
            min_val = np.min(voice_sample)
            
            # Frequency domain features (simplified)
            fft = np.fft.fft(voice_sample)
            magnitude_spectrum = np.abs(fft)
            spectral_centroid = np.sum(magnitude_spectrum * np.arange(len(magnitude_spectrum))) / np.sum(magnitude_spectrum)
            
            # Zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.sign(voice_sample)))) / 2
            
            # Combine features
            features = np.array([
                mean_val, std_val, max_val, min_val,
                spectral_centroid, zero_crossings,
                np.mean(magnitude_spectrum[:len(magnitude_spectrum)//4]),  # Low freq energy
                np.mean(magnitude_spectrum[len(magnitude_spectrum)//4:len(magnitude_spectrum)//2]),  # Mid freq energy
                np.mean(magnitude_spectrum[len(magnitude_spectrum)//2:])  # High freq energy
            ])
            
            return features
            
        except Exception as e:
            logging.error(f"Voice feature extraction error: {e}")
            return None
    
    def _calculate_voice_similarity(self, enrolled_features, sample_features):
        """Calculate similarity between voice feature vectors."""
        try:
            # Normalize features
            enrolled_norm = enrolled_features / (np.linalg.norm(enrolled_features) + 1e-8)
            sample_norm = sample_features / (np.linalg.norm(sample_features) + 1e-8)
            
            # Calculate cosine similarity
            similarity = np.dot(enrolled_norm, sample_norm)
            
            # Convert to confidence score (0-1)
            confidence = (similarity + 1) / 2  # Map from [-1,1] to [0,1]
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logging.error(f"Voice similarity calculation error: {e}")
            return 0.0
    
    def _enroll_behavior(self, user_id, behavior_data):
        """Enhanced behavioral biometric enrollment."""
        try:
            # Extract behavioral features
            behavioral_features = self._extract_behavioral_features(behavior_data)
            
            if behavioral_features is None:
                return False
            
            if user_id not in self.behavioral_models:
                self.behavioral_models[user_id] = {
                    "samples": [],
                    "enrollment_date": datetime.now().isoformat()
                }
            
            self.behavioral_models[user_id]["samples"].append({
                "features": behavioral_features,
                "data_points": len(behavior_data),
                "timestamp": datetime.now().isoformat()
            })
            
            # Limit to 10 behavioral samples per user
            if len(self.behavioral_models[user_id]["samples"]) > 10:
                self.behavioral_models[user_id]["samples"].pop(0)
            
            return True
            
        except Exception as e:
            logging.error(f"Behavioral enrollment error: {e}")
            return False
    
    def _authenticate_behavior(self, user_id, behavior_data):
        """Enhanced behavioral authentication."""
        if user_id not in self.behavioral_models:
            return {"success": False, "confidence": 0.0, "reason": "no_enrollment"}
            
        try:
            # Extract behavioral features
            sample_features = self._extract_behavioral_features(behavior_data)
            
            if sample_features is None:
                return {"success": False, "confidence": 0.0, "reason": "feature_extraction_failed"}
            
            # Compare with all enrolled behavioral samples
            confidences = []
            for behavior_sample in self.behavioral_models[user_id]["samples"]:
                enrolled_features = behavior_sample["features"]
                confidence = self._calculate_behavioral_similarity(enrolled_features, sample_features)
                confidences.append(confidence)
            
            # Use average of top matches for stability
            if len(confidences) >= 3:
                top_confidences = sorted(confidences, reverse=True)[:3]
                average_confidence = np.mean(top_confidences)
            else:
                average_confidence = max(confidences) if confidences else 0.0
            
            success = average_confidence >= self.behavior_threshold
            
            return {
                "success": success,
                "confidence": average_confidence,
                "threshold": self.behavior_threshold,
                "samples_compared": len(confidences)
            }
            
        except Exception as e:
            logging.error(f"Behavioral authentication error: {e}")
            return {"success": False, "confidence": 0.0, "reason": "authentication_error"}
    
    def _extract_behavioral_features(self, behavior_data):
        """Extract comprehensive behavioral features from user interaction data."""
        try:
            features = []
            
            # Typing patterns
            if "typing_patterns" in behavior_data:
                typing_data = behavior_data["typing_patterns"]
                features.extend([
                    typing_data.get("avg_keystroke_time", 0),
                    typing_data.get("keystroke_variance", 0),
                    typing_data.get("pause_patterns", 0),
                    typing_data.get("typing_speed", 0),
                    typing_data.get("error_rate", 0)
                ])
                
            # Mouse movement patterns
            if "mouse_patterns" in behavior_data:
                mouse_data = behavior_data["mouse_patterns"]
                features.extend([
                    mouse_data.get("avg_velocity", 0),
                    mouse_data.get("click_frequency", 0),
                    mouse_data.get("movement_smoothness", 0),
                    mouse_data.get("scroll_patterns", 0),
                    mouse_data.get("drag_patterns", 0)
                ])
                
            # Interaction timing patterns
            if "interaction_timing" in behavior_data:
                timing_data = behavior_data["interaction_timing"]
                features.extend([
                    timing_data.get("avg_response_time", 0),
                    timing_data.get("command_frequency", 0),
                    timing_data.get("session_duration_preference", 0),
                    timing_data.get("break_patterns", 0)
                ])
                
            # Navigation patterns
            if "navigation_patterns" in behavior_data:
                nav_data = behavior_data["navigation_patterns"]
                features.extend([
                    nav_data.get("preferred_shortcuts", 0),
                    nav_data.get("menu_usage_pattern", 0),
                    nav_data.get("feature_discovery_rate", 0),
                    nav_data.get("workflow_consistency", 0)
                ])
            
            # Voice interaction patterns
            if "voice_patterns" in behavior_data:
                voice_data = behavior_data["voice_patterns"]
                features.extend([
                    voice_data.get("command_complexity", 0),
                    voice_data.get("speech_pace", 0),
                    voice_data.get("retry_patterns", 0)
                ])
            
            return np.array(features) if features else None
            
        except Exception as e:
            logging.error(f"Behavioral feature extraction error: {e}")
            return None
    
    def _calculate_behavioral_similarity(self, enrolled_features, sample_features):
        """Calculate similarity between behavioral feature vectors."""
        try:
            # Ensure same length
            min_len = min(len(enrolled_features), len(sample_features))
            if min_len == 0:
                return 0.0
                
            enrolled_trim = enrolled_features[:min_len]
            sample_trim = sample_features[:min_len]
            
            # Calculate multiple similarity metrics
            # 1. Cosine similarity
            cosine_sim = np.dot(enrolled_trim, sample_trim) / (np.linalg.norm(enrolled_trim) * np.linalg.norm(sample_trim) + 1e-8)
            
            # 2. Normalized correlation
            correlation = np.corrcoef(enrolled_trim, sample_trim)[0, 1] if not np.isnan(np.corrcoef(enrolled_trim, sample_trim)[0, 1]) else 0
            
            # 3. Inverse normalized Euclidean distance
            euclidean_dist = np.linalg.norm(enrolled_trim - sample_trim)
            max_distance = np.linalg.norm(enrolled_trim) + np.linalg.norm(sample_trim)
            euclidean_sim = 1.0 - (euclidean_dist / (max_distance + 1e-8))
            
            # Combine similarities with weights
            combined_similarity = (
                0.4 * max(0, (cosine_sim + 1) / 2) +  # Normalize cosine to [0,1]
                0.4 * max(0, (correlation + 1) / 2) +  # Normalize correlation to [0,1]
                0.2 * max(0, euclidean_sim)
            )
            
            return max(0.0, min(1.0, combined_similarity))
            
        except Exception as e:
            logging.error(f"Behavioral similarity calculation error: {e}")
            return 0.0
    
    # Security and user management methods
    def _is_user_locked(self, user_id):
        """Check if user is currently locked out."""
        if user_id not in self.locked_users:
            return False
            
        lockout_time = datetime.fromisoformat(self.locked_users[user_id])
        return datetime.now() < lockout_time + timedelta(seconds=self.lockout_duration)
        
    def _get_lockout_remaining(self, user_id):
        """Get remaining lockout time in seconds."""
        if user_id not in self.locked_users:
            return 0
            
        lockout_time = datetime.fromisoformat(self.locked_users[user_id])
        remaining = (lockout_time + timedelta(seconds=self.lockout_duration) - datetime.now()).total_seconds()
        
        return max(0, remaining)
        
    def _record_failed_attempt(self, user_id):
        """Record failed authentication attempt."""
        if user_id is None:
            return
            
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = 0
            
        self.failed_attempts[user_id] += 1
        
        # Lock user if too many failed attempts
        if self.failed_attempts[user_id] >= self.max_attempts:
            self.locked_users[user_id] = datetime.now().isoformat()
            logging.warning(f"User {user_id} locked due to failed authentication attempts")
            
    def _reset_failed_attempts(self, user_id):
        """Reset failed attempt counter for successful authentication."""
        if user_id is None:
            return
            
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
        if user_id in self.locked_users:
            del self.locked_users[user_id]
    
    def _load_biometric_data(self):
        """Load existing biometric data from storage."""
        try:
            # Load face encodings
            face_file = f"{self.data_dir}/face_encodings.pkl"
            if os.path.exists(face_file):
                with open(face_file, 'rb') as f:
                    self.face_encodings = pickle.load(f)
                    
            # Load voice prints
            voice_file = f"{self.data_dir}/voice_prints.pkl"
            if os.path.exists(voice_file):
                with open(voice_file, 'rb') as f:
                    self.voice_prints = pickle.load(f)
                    
            # Load behavioral models
            behavior_file = f"{self.data_dir}/behavioral_models.pkl"
            if os.path.exists(behavior_file):
                with open(behavior_file, 'rb') as f:
                    self.behavioral_models = pickle.load(f)
                    
            logging.info("Biometric data loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading biometric data: {e}")
            
    def _save_biometric_data(self):
        """Save biometric data to storage."""
        try:
            # Save face encodings
            face_file = f"{self.data_dir}/face_encodings.pkl"
            with open(face_file, 'wb') as f:
                pickle.dump(self.face_encodings, f)
                
            # Save voice prints
            voice_file = f"{self.data_dir}/voice_prints.pkl"
            with open(voice_file, 'wb') as f:
                pickle.dump(self.voice_prints, f)
                
            # Save behavioral models
            behavior_file = f"{self.data_dir}/behavioral_models.pkl"
            with open(behavior_file, 'wb') as f:
                pickle.dump(self.behavioral_models, f)
                
            logging.info("Biometric data saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving biometric data: {e}")
    
    def get_user_biometric_status(self, user_id):
        """Get comprehensive biometric enrollment status for user."""
        return {
            "user_id": user_id,
            "face_enrolled": user_id in self.face_encodings,
            "voice_enrolled": user_id in self.voice_prints,
            "behavior_enrolled": user_id in self.behavioral_models,
            "is_locked": self._is_user_locked(user_id),
            "failed_attempts": self.failed_attempts.get(user_id, 0),
            "lockout_remaining": self._get_lockout_remaining(user_id),
            "enrollment_details": {
                "face_samples": len(self.face_encodings[user_id].get("samples", [])) if user_id in self.face_encodings else 0,
                "voice_samples": len(self.voice_prints[user_id].get("samples", [])) if user_id in self.voice_prints else 0,
                "behavior_samples": len(self.behavioral_models[user_id].get("samples", [])) if user_id in self.behavioral_models else 0
            }
        }
        
    def remove_user_biometrics(self, user_id):
        """Remove all biometric data for a user."""
        removed = []
        
        if user_id in self.face_encodings:
            del self.face_encodings[user_id]
            removed.append("face")
            
        if user_id in self.voice_prints:
            del self.voice_prints[user_id]
            removed.append("voice")
            
        if user_id in self.behavioral_models:
            del self.behavioral_models[user_id]
            removed.append("behavior")
            
        # Clean up security data
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
        if user_id in self.locked_users:
            del self.locked_users[user_id]
            
        self._save_biometric_data()
        
        return {
            "user_id": user_id,
            "removed_biometrics": removed,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_security_settings(self, settings):
        """Update security thresholds and settings."""
        if "face_threshold" in settings:
            self.face_threshold = max(0.0, min(1.0, settings["face_threshold"]))
        if "voice_threshold" in settings:
            self.voice_threshold = max(0.0, min(1.0, settings["voice_threshold"]))
        if "behavior_threshold" in settings:
            self.behavior_threshold = max(0.0, min(1.0, settings["behavior_threshold"]))
        if "max_attempts" in settings:
            self.max_attempts = max(1, settings["max_attempts"])
        if "lockout_duration" in settings:
            self.lockout_duration = max(60, settings["lockout_duration"])  # Minimum 1 minute
        
        logging.info(f"Security settings updated: {settings}")
        return {"success": True, "updated_settings": settings}


# Additional privacy and security classes
class PrivacyManager:
    """Enhanced privacy and data protection manager for user profiles."""
    
    def __init__(self, encryption_key=None):
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.privacy_levels = {
            "minimal": {
                "data_collection": "basic_only",
                "personalization": "limited",
                "data_sharing": "none",
                "retention_days": 30,
                "encryption_level": "basic"
            },
            "balanced": {
                "data_collection": "standard",
                "personalization": "moderate", 
                "data_sharing": "anonymized",
                "retention_days": 90,
                "encryption_level": "standard"
            },
            "full": {
                "data_collection": "comprehensive",
                "personalization": "full",
                "data_sharing": "with_consent",
                "retention_days": 365,
                "encryption_level": "advanced"
            }
        }
        
    def apply_privacy_settings(self, user_data, privacy_level):
        """Apply comprehensive privacy settings to user data."""
        if privacy_level not in self.privacy_levels:
            privacy_level = "balanced"
            
        settings = self.privacy_levels[privacy_level]
        protected_data = user_data.copy()
        
        # Apply data collection limits
        if settings["data_collection"] == "basic_only":
            protected_data = self._limit_to_basic_data(protected_data)
        elif settings["data_collection"] == "standard":
            protected_data = self._apply_standard_limits(protected_data)
            
        # Apply anonymization if needed
        if settings["data_sharing"] == "anonymized":
            protected_data = self._anonymize_data(protected_data)
        elif settings["data_sharing"] == "none":
            protected_data = self._strip_shareable_data(protected_data)
            
        # Apply encryption based on level
        if settings["encryption_level"] == "advanced":
            protected_data = self._apply_advanced_encryption(protected_data)
        elif settings["encryption_level"] == "standard":
            protected_data = self._apply_standard_encryption(protected_data)
            
        return protected_data
        
    def encrypt_sensitive_data(self, data):
        """Enhanced encryption for sensitive user data."""
        try:
            if CRYPTOGRAPHY_AVAILABLE:
                f = Fernet(self.encryption_key)
                data_str = json.dumps(data, default=str)
                encrypted_data = f.encrypt(data_str.encode())
                return encrypted_data
            else:
                logging.warning("Cryptography library not available - using fallback encryption")
                return self._encrypt_data_fallback(data)
                
        except Exception as e:
            logging.error(f"Encryption error: {e}")
            return data
            
    def decrypt_sensitive_data(self, encrypted_data):
        """Enhanced decryption for sensitive user data."""
        try:
            if CRYPTOGRAPHY_AVAILABLE:
                f = Fernet(self.encryption_key)
                decrypted_str = f.decrypt(encrypted_data).decode()
                return json.loads(decrypted_str)
            else:
                logging.warning("Cryptography library not available - using fallback decryption")
                return self._decrypt_data_fallback(encrypted_data)
                
        except Exception as e:
            logging.error(f"Decryption error: {e}")
            return {}
    
    def _generate_encryption_key(self):
        """Generate secure encryption key."""
        try:
            if CRYPTOGRAPHY_AVAILABLE:
                return Fernet.generate_key()
            else:
                # Fallback key generation
                return hashlib.sha256(f"jarvis_default_key_{time.time()}".encode()).digest()
        except:
            return b"default_key_placeholder"
    
    def _encrypt_data_fallback(self, data):
        """Fallback encryption using base64 encoding."""
        try:
            import base64
            data_str = json.dumps(data, default=str)
            encoded = base64.b64encode(data_str.encode()).decode()
            return encoded
        except Exception as e:
            logging.error(f"Fallback encryption error: {e}")
            return data
    
    def _decrypt_data_fallback(self, encrypted_data):
        """Fallback decryption using base64 decoding."""
        try:
            import base64
            decoded = base64.b64decode(encrypted_data.encode()).decode()
            return json.loads(decoded)
        except Exception as e:
            logging.error(f"Fallback decryption error: {e}")
            return encrypted_data
    
    def _limit_to_basic_data(self, user_data):
        """Limit data to basic profile information only."""
        basic_fields = [
            "id", "personal_info.name", "personal_info.language", 
            "preferences.communication_style", "preferences.privacy_level"
        ]
        return self._filter_data_fields(user_data, basic_fields)
    
    def _apply_standard_limits(self, user_data):
        """Apply standard privacy limits."""
        sensitive_fields = [
            "security.biometric_data", "personal_info.precise_location",
            "behavioral_patterns.detailed_tracking", "voice_patterns.raw_audio"
        ]
        return self._remove_data_fields(user_data, sensitive_fields)
    
    def _anonymize_data(self, user_data):
        """Anonymize user data for sharing."""
        anonymized = user_data.copy()
        
        if "personal_info" in anonymized:
            if "name" in anonymized["personal_info"]:
                anonymized["personal_info"]["name"] = self._hash_value(anonymized["personal_info"]["name"])
            if "location" in anonymized["personal_info"]:
                anonymized["personal_info"]["location"] = "anonymized"
                
        if "id" in anonymized:
            anonymized["id"] = self._hash_value(anonymized["id"])
            
        return anonymized
    
    def _hash_value(self, value):
        """Create secure hash of sensitive value."""
        return hashlib.sha256(str(value).encode()).hexdigest()[:16]
    
    def _filter_data_fields(self, data, allowed_fields):
        """Filter data to only include specified fields."""
        filtered = {}
        
        for field in allowed_fields:
            parts = field.split('.')
            current_data = data
            current_filtered = filtered
            
            for i, part in enumerate(parts):
                if part in current_data:
                    if i == len(parts) - 1:
                        current_filtered[part] = current_data[part]
                    else:
                        if part not in current_filtered:
                            current_filtered[part] = {}
                        current_filtered = current_filtered[part]
                        current_data = current_data[part]
                        
        return filtered
    
    def _remove_data_fields(self, data, removed_fields):
        """Remove specific fields from data."""
        filtered = data.copy()
        
        for field in removed_fields:
            parts = field.split('.')
            current = filtered
            
            for i, part in enumerate(parts):
                if part in current:
                    if i == len(parts) - 1:
                        del current[part]
                    else:
                        current = current[part]
                        
        return filtered
    
    def _strip_shareable_data(self, user_data):
        """Remove all data that could be shared externally."""
        shareable_fields = [
            "analytics", "usage_patterns", "performance_metrics",
            "error_reports", "feature_usage"
        ]
        return self._remove_data_fields(user_data, shareable_fields)
    
    def _apply_advanced_encryption(self, data):
        """Apply advanced encryption to sensitive data fields."""
        sensitive_fields = ["biometric_data", "voice_patterns", "behavioral_patterns"]
        
        for field in sensitive_fields:
            if field in data:
                data[field] = self.encrypt_sensitive_data(data[field])
        
        return data
    
    def _apply_standard_encryption(self, data):
        """Apply standard encryption to moderately sensitive fields."""
        moderate_fields = ["personal_info", "preferences.detailed"]
        
        for field in moderate_fields:
            if field in data:
                data[field] = self.encrypt_sensitive_data(data[field])
        
        return data


if __name__ == "__main__":
    # Test the enhanced biometric authentication system
    print("Testing Enhanced Biometric Authentication System...")
    
    # Initialize authenticator
    auth = BiometricAuthenticator()
    
    # Test data
    test_user_id = "test_user_1"
    
    # Create dummy test data
    test_face = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_voice = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
    test_behavior = {
        "typing_patterns": {"avg_keystroke_time": 0.15, "keystroke_variance": 0.05},
        "mouse_patterns": {"avg_velocity": 100, "click_frequency": 2.5},
        "interaction_timing": {"avg_response_time": 1.2}
    }
    
    # Test enrollment
    print("Testing biometric enrollment...")
    enrollment_result = auth.enroll_user_biometrics(
        test_user_id, 
        face_image=test_face,
        voice_sample=test_voice,
        behavior_data=test_behavior
    )
    print(f"Enrollment result: {enrollment_result}")
    
    # Test authentication
    print("Testing biometric authentication...")
    auth_result = auth.authenticate_user(
        test_user_id,
        face_image=test_face,
        voice_sample=test_voice,
        behavior_data=test_behavior
    )
    print(f"Authentication result: {auth_result}")
    
    # Test status
    status = auth.get_user_biometric_status(test_user_id)
    print(f"User status: {status}")
    
    print("Enhanced Biometric Authentication System test completed!")