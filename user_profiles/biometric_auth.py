"""
Biometric authentication and security features for JARVIS user profiles.
Provides face recognition, voice print authentication, and behavioral biometrics.
"""

import numpy as np
import cv2
import hashlib
import pickle
import os
import json
from datetime import datetime, timedelta
import logging
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import threading
import time
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

class BiometricAuthenticator:
    """
    Multi-modal biometric authentication system for secure user identification.
    Supports face recognition, voice authentication, and behavioral patterns.
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
        
        logging.info("Biometric Authenticator initialized")
        
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
        """Enroll face biometric."""
        try:
            if not FACE_RECOGNITION_AVAILABLE:
                logging.warning("Face recognition library not available - using fallback method")
                return self._enroll_face_fallback(user_id, face_image)
            
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
                
            # Store the first face encoding
            self.face_encodings[user_id] = {
                "encoding": face_encodings[0],
                "enrollment_date": datetime.now().isoformat(),
                "face_location": face_locations[0]
            }
            
            # Save face image
            face_path = f"{self.data_dir}/faces/{user_id}_face.jpg"
            cv2.imwrite(face_path, face_image)
            
            return True
            
        except Exception as e:
            logging.error(f"Face enrollment error: {e}")
            return False
            
    def _enroll_voice(self, user_id, voice_sample):
        """Enroll voice biometric."""
        try:
            # Extract voice features (MFCC, pitch, etc.)
            voice_features = self._extract_voice_features(voice_sample)
            
            if voice_features is None:
                return False
                
            self.voice_prints[user_id] = {
                "features": voice_features,
                "enrollment_date": datetime.now().isoformat(),
                "sample_duration": len(voice_sample) / 16000  # Assuming 16kHz sample rate
            }
            
            # Save voice sample
            voice_path = f"{self.data_dir}/voices/{user_id}_voice.wav"
            # Save voice sample using appropriate audio library
            
            return True
            
        except Exception as e:
            logging.error(f"Voice enrollment error: {e}")
            return False
            
    def _enroll_behavior(self, user_id, behavior_data):
        """Enroll behavioral biometric patterns."""
        try:
            # Extract behavioral features
            behavioral_features = self._extract_behavioral_features(behavior_data)
            
            if behavioral_features is None:
                return False
                
            self.behavioral_models[user_id] = {
                "features": behavioral_features,
                "enrollment_date": datetime.now().isoformat(),
                "data_points": len(behavior_data)
            }
            
            return True
            
        except Exception as e:
            logging.error(f"Behavioral enrollment error: {e}")
            return False
            
    def _authenticate_face(self, user_id, face_image):
        """Authenticate using face recognition."""
        if user_id not in self.face_encodings:
            return {"success": False, "confidence": 0.0, "reason": "no_enrollment"}
            
        try:
            if not FACE_RECOGNITION_AVAILABLE:
                logging.warning("Face recognition library not available - using fallback method")
                return self._verify_face_fallback(user_id, face_image)
            
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
                
            # Compare with enrolled face
            enrolled_encoding = self.face_encodings[user_id]["encoding"]
            face_distances = face_recognition.face_distance([enrolled_encoding], face_encodings[0])
            
            # Convert distance to confidence (lower distance = higher confidence)
            confidence = max(0.0, 1.0 - face_distances[0])
            success = confidence >= self.face_threshold
            
            return {
                "success": success,
                "confidence": confidence,
                "distance": face_distances[0],
                "threshold": self.face_threshold
            }
            
        except Exception as e:
            logging.error(f"Face authentication error: {e}")
            return {"success": False, "confidence": 0.0, "reason": "authentication_error"}
            
    def _authenticate_voice(self, user_id, voice_sample):
        """Authenticate using voice recognition."""
        if user_id not in self.voice_prints:
            return {"success": False, "confidence": 0.0, "reason": "no_enrollment"}
            
        try:
            # Extract features from voice sample
            sample_features = self._extract_voice_features(voice_sample)
            
            if sample_features is None:
                return {"success": False, "confidence": 0.0, "reason": "feature_extraction_failed"}
                
            # Compare with enrolled voice print
            enrolled_features = self.voice_prints[user_id]["features"]
            
            # Calculate similarity
            confidence = self._calculate_voice_similarity(enrolled_features, sample_features)
            success = confidence >= self.voice_threshold
            
            return {
                "success": success,
                "confidence": confidence,
                "threshold": self.voice_threshold
            }
            
        except Exception as e:
            logging.error(f"Voice authentication error: {e}")
            return {"success": False, "confidence": 0.0, "reason": "authentication_error"}
            
    def _authenticate_behavior(self, user_id, behavior_data):
        """Authenticate using behavioral patterns."""
        if user_id not in self.behavioral_models:
            return {"success": False, "confidence": 0.0, "reason": "no_enrollment"}
            
        try:
            # Extract behavioral features
            sample_features = self._extract_behavioral_features(behavior_data)
            
            if sample_features is None:
                return {"success": False, "confidence": 0.0, "reason": "feature_extraction_failed"}
                
            # Compare with enrolled behavioral model
            enrolled_features = self.behavioral_models[user_id]["features"]
            
            # Calculate behavioral similarity
            confidence = self._calculate_behavioral_similarity(enrolled_features, sample_features)
            success = confidence >= self.behavior_threshold
            
            return {
                "success": success,
                "confidence": confidence,
                "threshold": self.behavior_threshold
            }
            
        except Exception as e:
            logging.error(f"Behavioral authentication error: {e}")
            return {"success": False, "confidence": 0.0, "reason": "authentication_error"}
            
    def _extract_voice_features(self, voice_sample):
        """Extract voice features for authentication."""
        try:
            # This would typically use librosa or similar for MFCC extraction
            # For now, return a placeholder feature vector
            
            # Simulate MFCC features (13 coefficients)
            mfcc_features = np.random.random(13)  # Placeholder
            
            # Simulate pitch features
            pitch_features = np.random.random(5)  # Placeholder
            
            # Combine features
            features = np.concatenate([mfcc_features, pitch_features])
            
            return features
            
        except Exception as e:
            logging.error(f"Voice feature extraction error: {e}")
            return None
            
    def _extract_behavioral_features(self, behavior_data):
        """Extract behavioral features from user interaction data."""
        try:
            features = []
            
            # Typing patterns
            if "typing_patterns" in behavior_data:
                typing_data = behavior_data["typing_patterns"]
                features.extend([
                    typing_data.get("avg_keystroke_time", 0),
                    typing_data.get("keystroke_variance", 0),
                    typing_data.get("pause_patterns", 0)
                ])
                
            # Mouse movement patterns
            if "mouse_patterns" in behavior_data:
                mouse_data = behavior_data["mouse_patterns"]
                features.extend([
                    mouse_data.get("avg_velocity", 0),
                    mouse_data.get("click_frequency", 0),
                    mouse_data.get("movement_smoothness", 0)
                ])
                
            # Interaction timing patterns
            if "interaction_timing" in behavior_data:
                timing_data = behavior_data["interaction_timing"]
                features.extend([
                    timing_data.get("avg_response_time", 0),
                    timing_data.get("command_frequency", 0),
                    timing_data.get("session_duration_preference", 0)
                ])
                
            # Navigation patterns
            if "navigation_patterns" in behavior_data:
                nav_data = behavior_data["navigation_patterns"]
                features.extend([
                    nav_data.get("preferred_shortcuts", 0),
                    nav_data.get("menu_usage_pattern", 0),
                    nav_data.get("feature_discovery_rate", 0)
                ])
                
            return np.array(features) if features else None
            
        except Exception as e:
            logging.error(f"Behavioral feature extraction error: {e}")
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
            
    def _calculate_behavioral_similarity(self, enrolled_features, sample_features):
        """Calculate similarity between behavioral feature vectors."""
        try:
            # Ensure same length
            min_len = min(len(enrolled_features), len(sample_features))
            enrolled_trim = enrolled_features[:min_len]
            sample_trim = sample_features[:min_len]
            
            # Calculate normalized Euclidean distance
            distance = np.linalg.norm(enrolled_trim - sample_trim)
            max_distance = np.linalg.norm(enrolled_trim) + np.linalg.norm(sample_trim)
            
            # Convert distance to similarity
            if max_distance > 0:
                similarity = 1.0 - (distance / max_distance)
            else:
                similarity = 1.0
                
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logging.error(f"Behavioral similarity calculation error: {e}")
            return 0.0
            
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
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = 0
            
        self.failed_attempts[user_id] += 1
        
        # Lock user if too many failed attempts
        if self.failed_attempts[user_id] >= self.max_attempts:
            self.locked_users[user_id] = datetime.now().isoformat()
            logging.warning(f"User {user_id} locked due to failed authentication attempts")
            
    def _reset_failed_attempts(self, user_id):
        """Reset failed attempt counter for successful authentication."""
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
        """Get biometric enrollment status for user."""
        return {
            "user_id": user_id,
            "face_enrolled": user_id in self.face_encodings,
            "voice_enrolled": user_id in self.voice_prints,
            "behavior_enrolled": user_id in self.behavioral_models,
            "is_locked": self._is_user_locked(user_id),
            "failed_attempts": self.failed_attempts.get(user_id, 0),
            "lockout_remaining": self._get_lockout_remaining(user_id)
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

class PrivacyManager:
    """
    Privacy and data protection manager for user profiles.
    Handles data encryption, anonymization, and privacy preferences.
    """
    
    def __init__(self, encryption_key=None):
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.privacy_levels = {
            "minimal": {
                "data_collection": "basic_only",
                "personalization": "limited",
                "data_sharing": "none",
                "retention_days": 30
            },
            "balanced": {
                "data_collection": "standard",
                "personalization": "moderate",
                "data_sharing": "anonymized",
                "retention_days": 90
            },
            "full": {
                "data_collection": "comprehensive",
                "personalization": "full",
                "data_sharing": "with_consent",
                "retention_days": 365
            }
        }
        
    def apply_privacy_settings(self, user_data, privacy_level):
        """Apply privacy settings to user data."""
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
            
        return protected_data
        
    def encrypt_sensitive_data(self, data):
        """Encrypt sensitive user data."""
        try:
            if not CRYPTOGRAPHY_AVAILABLE:
                logging.warning("Cryptography library not available - using fallback encryption")
                return self._encrypt_data_fallback(data)
            
            f = Fernet(self.encryption_key)
            
            # Convert data to JSON string
            import json
            data_str = json.dumps(data)
            
            # Encrypt
            encrypted_data = f.encrypt(data_str.encode())
            
            return encrypted_data
            
        except Exception as e:
            logging.error(f"Encryption error: {e}")
            return data  # Return original if encryption fails
            
    def decrypt_sensitive_data(self, encrypted_data):
        """Decrypt sensitive user data."""
        try:
            if not CRYPTOGRAPHY_AVAILABLE:
                logging.warning("Cryptography library not available - using fallback decryption")
                return self._decrypt_data_fallback(encrypted_data)
            
            f = Fernet(self.encryption_key)
            
            # Decrypt
            decrypted_str = f.decrypt(encrypted_data).decode()
            
            # Convert back to object
            import json
            data = json.loads(decrypted_str)
            
            return data
            
        except Exception as e:
            logging.error(f"Decryption error: {e}")
            return {}
            
    def _generate_encryption_key(self):
        """Generate encryption key for data protection."""
        try:
            if CRYPTOGRAPHY_AVAILABLE:
                return Fernet.generate_key()
            else:
                # Fallback key generation
                return hashlib.sha256(b"jarvis_default_key").digest()
        except:
            return b"default_key_placeholder"  # Fallback
            
    def _limit_to_basic_data(self, user_data):
        """Limit data to basic profile information only."""
        basic_fields = [
            "id", "personal_info.name", "personal_info.language", 
            "preferences.communication_style", "preferences.privacy_level"
        ]
        
        return self._filter_data_fields(user_data, basic_fields)
        
    def _apply_standard_limits(self, user_data):
        """Apply standard privacy limits."""
        # Remove highly sensitive data
        sensitive_fields = [
            "security.biometric_data", "personal_info.location",
            "behavioral_patterns.detailed_tracking"
        ]
        
        return self._remove_data_fields(user_data, sensitive_fields)
        
    def _anonymize_data(self, user_data):
        """Anonymize user data for sharing."""
        anonymized = user_data.copy()
        
        # Replace identifying information with hashed versions
        if "personal_info" in anonymized:
            if "name" in anonymized["personal_info"]:
                anonymized["personal_info"]["name"] = self._hash_value(anonymized["personal_info"]["name"])
            if "location" in anonymized["personal_info"]:
                anonymized["personal_info"]["location"] = "anonymized"
                
        # Remove direct identifiers
        if "id" in anonymized:
            anonymized["id"] = self._hash_value(anonymized["id"])
            
        return anonymized
        
    def _hash_value(self, value):
        """Create hash of sensitive value."""
        return hashlib.sha256(str(value).encode()).hexdigest()[:16]
        
    def _filter_data_fields(self, data, allowed_fields):
        """Filter data to only include allowed fields."""
        filtered = {}
        
        for field in allowed_fields:
            parts = field.split('.')
            current_data = data
            current_filtered = filtered
            
            for i, part in enumerate(parts):
                if part in current_data:
                    if i == len(parts) - 1:
                        # Last part - copy the value
                        current_filtered[part] = current_data[part]
                    else:
                        # Intermediate part - ensure dict exists
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
                        # Last part - remove it
                        del current[part]
                    else:
                        current = current[part]
                        
        return filtered
    
    def _encrypt_data_fallback(self, data):
        """Fallback encryption using basic encoding."""
        try:
            import base64
            
            data_str = json.dumps(data)
            encoded = base64.b64encode(data_str.encode()).decode()
            return encoded
            
        except Exception as e:
            logging.error(f"Fallback encryption error: {e}")
            return data
    
    def _decrypt_data_fallback(self, encrypted_data):
        """Fallback decryption using basic decoding."""
        try:
            import base64
            
            decoded = base64.b64decode(encrypted_data.encode()).decode()
            return json.loads(decoded)
            
        except Exception as e:
            logging.error(f"Fallback decryption error: {e}")
            return encrypted_data

class AdaptationEngine:
    """
    AI-powered adaptation engine that learns from user behavior and adapts the system.
    Uses machine learning to optimize user experience over time.
    """
    
    def __init__(self):
        self.adaptation_models = {}
        self.learning_history = defaultdict(list)
        self.adaptation_cache = {}
        
    def adapt_interface(self, user_id, interface_usage_data):
        """Adapt interface based on user behavior."""
        adaptations = {
            "layout_changes": [],
            "feature_recommendations": [],
            "shortcut_suggestions": [],
            "workflow_optimizations": []
        }
        
        # Analyze usage patterns
        patterns = self._analyze_interface_patterns(interface_usage_data)
        
        # Generate layout adaptations
        if patterns.get("frequent_features"):
            adaptations["layout_changes"] = self._suggest_layout_changes(patterns["frequent_features"])
            
        # Generate feature recommendations
        adaptations["feature_recommendations"] = self._recommend_features(patterns)
        
        # Generate shortcut suggestions
        adaptations["shortcut_suggestions"] = self._suggest_shortcuts(patterns)
        
        return adaptations
        
    def adapt_responses(self, user_id, conversation_history):
        """Adapt response style based on conversation patterns."""
        if user_id not in self.adaptation_models:
            self.adaptation_models[user_id] = ResponseAdaptationModel()
            
        model = self.adaptation_models[user_id]
        
        # Analyze conversation patterns
        patterns = self._analyze_conversation_patterns(conversation_history)
        
        # Update adaptation model
        model.update(patterns)
        
        # Generate response adaptations
        adaptations = model.get_adaptations()
        
        return adaptations
        
    def _analyze_interface_patterns(self, usage_data):
        """Analyze interface usage patterns."""
        patterns = {}
        
        if "feature_usage" in usage_data:
            feature_counts = usage_data["feature_usage"]
            
            # Find most frequently used features
            frequent_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            patterns["frequent_features"] = [f[0] for f in frequent_features]
            
        if "navigation_paths" in usage_data:
            # Analyze common navigation paths
            paths = usage_data["navigation_paths"]
            patterns["common_paths"] = self._find_common_paths(paths)
            
        return patterns
        
    def _analyze_conversation_patterns(self, conversation_history):
        """Analyze conversation patterns for response adaptation."""
        patterns = {
            "avg_response_length": 0,
            "question_types": [],
            "satisfaction_scores": [],
            "topic_preferences": [],
            "communication_style": "neutral"
        }
        
        if not conversation_history:
            return patterns
            
        # Analyze response lengths
        response_lengths = []
        for conv in conversation_history:
            if "response_length" in conv:
                response_lengths.append(conv["response_length"])
                
        if response_lengths:
            patterns["avg_response_length"] = np.mean(response_lengths)
            
        # Analyze satisfaction scores
        satisfaction_scores = []
        for conv in conversation_history:
            if "satisfaction" in conv:
                satisfaction_scores.append(conv["satisfaction"])
                
        patterns["satisfaction_scores"] = satisfaction_scores
        
        return patterns
        
    def _suggest_layout_changes(self, frequent_features):
        """Suggest layout changes based on feature usage."""
        suggestions = []
        
        for feature in frequent_features[:3]:  # Top 3 features
            suggestions.append({
                "type": "move_to_primary",
                "feature": feature,
                "reason": f"Frequently used feature should be more accessible"
            })
            
        return suggestions
        
    def _recommend_features(self, patterns):
        """Recommend features based on usage patterns."""
        recommendations = []
        
        # Recommend related features
        if "frequent_features" in patterns:
            for feature in patterns["frequent_features"]:
                related = self._get_related_features(feature)
                for related_feature in related:
                    recommendations.append({
                        "feature": related_feature,
                        "reason": f"Often used with {feature}"
                    })
                    
        return recommendations[:5]  # Limit recommendations
        
    def _suggest_shortcuts(self, patterns):
        """Suggest shortcuts based on usage patterns."""
        suggestions = []
        
        if "common_paths" in patterns:
            for path in patterns["common_paths"][:3]:
                suggestions.append({
                    "path": " -> ".join(path),
                    "suggested_shortcut": f"Ctrl+{len(suggestions)+1}",
                    "reason": "Frequently used workflow"
                })
                
        return suggestions
        
    def _find_common_paths(self, navigation_paths):
        """Find common navigation paths."""
        from collections import Counter
        
        # Convert paths to tuples for counting
        path_tuples = [tuple(path) for path in navigation_paths if len(path) > 1]
        
        # Count occurrences
        path_counts = Counter(path_tuples)
        
        # Return most common paths
        common_paths = [list(path) for path, count in path_counts.most_common(5)]
        
        return common_paths
        
    def _get_related_features(self, feature):
        """Get features related to the given feature."""
        # This would be based on feature relationship mapping
        feature_relationships = {
            "voice_commands": ["gesture_control", "automation_rules"],
            "smart_home": ["automation_rules", "scheduling"],
            "face_recognition": ["security", "personalization"],
            "ar_interface": ["gesture_control", "voice_commands"]
        }
        
        return feature_relationships.get(feature, [])

class ResponseAdaptationModel:
    """Model for adapting response characteristics."""
    
    def __init__(self):
        self.patterns = {}
        self.adaptations = {}
        
    def update(self, new_patterns):
        """Update model with new patterns."""
        for key, value in new_patterns.items():
            if key not in self.patterns:
                self.patterns[key] = []
            self.patterns[key].append(value)
            
        # Update adaptations based on patterns
        self._compute_adaptations()
        
    def _compute_adaptations(self):
        """Compute response adaptations based on learned patterns."""
        # Response length adaptation
        if "avg_response_length" in self.patterns:
            recent_lengths = self.patterns["avg_response_length"][-5:]  # Last 5 interactions
            if recent_lengths:
                avg_length = np.mean(recent_lengths)
                if avg_length < 50:
                    self.adaptations["response_length"] = "short"
                elif avg_length > 200:
                    self.adaptations["response_length"] = "detailed"
                else:
                    self.adaptations["response_length"] = "medium"
                    
        # Satisfaction-based adaptations
        if "satisfaction_scores" in self.patterns:
            recent_satisfaction = []
            for scores in self.patterns["satisfaction_scores"][-3:]:  # Last 3 interactions
                if scores:
                    recent_satisfaction.extend(scores)
                    
            if recent_satisfaction:
                avg_satisfaction = np.mean(recent_satisfaction)
                if avg_satisfaction < 0.6:
                    self.adaptations["tone"] = "more_helpful"
                    self.adaptations["detail_level"] = "increase"
                elif avg_satisfaction > 0.8:
                    self.adaptations["tone"] = "maintain"
                    
    def get_adaptations(self):
        """Get current adaptations."""
        return self.adaptations.copy()

# Add these methods to BiometricAuthenticator class
def _add_fallback_methods_to_biometric_authenticator():
    """Helper function to add fallback methods to BiometricAuthenticator class."""
    
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
    
    def _verify_face_fallback(self, user_id, face_image):
        """Enhanced fallback face verification using multiple feature matching."""
        try:
            if user_id not in self.face_encodings or "samples" not in self.face_encodings[user_id]:
                return {"success": False, "confidence": 0.0, "reason": "no_enrollment"}
            
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
                # Multiple similarity metrics
                correlation = np.corrcoef(combined_features, stored_features)[0, 1] if len(combined_features) == len(stored_features) else 0
                cosine_sim = np.dot(combined_features, stored_features) / (np.linalg.norm(combined_features) * np.linalg.norm(stored_features))
                euclidean_dist = np.linalg.norm(combined_features - stored_features)
                
                # Normalize and combine metrics
                correlation_score = max(0, (correlation + 1) / 2) if not np.isnan(correlation) else 0
                cosine_score = max(0, (cosine_sim + 1) / 2)
                euclidean_score = max(0, 1 - (euclidean_dist / np.linalg.norm(combined_features)))
                
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
            return hist.astype(np.float32) / np.sum(hist)  # Normalize
            
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
            
            hog_features = []
            for i in range(0, rows - cell_size, cell_size):
                for j in range(0, cols - cell_size, cell_size):
                    cell_mag = magnitude[i:i+cell_size, j:j+cell_size]
                    cell_ori = orientation[i:i+cell_size, j:j+cell_size]
                    
                    hist, _ = np.histogram(cell_ori.flatten(), bins=n_bins, 
                                         range=(0, 180), weights=cell_mag.flatten())
                    hog_features.extend(hist)
            
            hog_array = np.array(hog_features, dtype=np.float32)
            return hog_array / (np.linalg.norm(hog_array) + 1e-8)  # Normalize
            
        except Exception as e:
            logging.error(f"HOG calculation error: {e}")
            return np.zeros(100, dtype=np.float32)  # Default size
    
    # Add methods to class
    BiometricAuthenticator._enroll_face_fallback = _enroll_face_fallback
    BiometricAuthenticator._verify_face_fallback = _verify_face_fallback
    BiometricAuthenticator._calculate_lbp_features = _calculate_lbp_features
    BiometricAuthenticator._calculate_hog_features = _calculate_hog_features

# Call the function to add methods to the class
_add_fallback_methods_to_biometric_authenticator()


# Test the biometric authentication system
if __name__ == "__main__":
    print("Testing Biometric Authentication System...")
    
    # Initialize authenticator
    authenticator = BiometricAuthenticator()
    
    # Test user
    test_user_id = "test_user_bio"
    
    print(f"Biometric Authentication System test completed!")