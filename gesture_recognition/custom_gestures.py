import cv2
import numpy as np
import json
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class CustomGestureTrainer:
    """
    Train custom gestures using machine learning.
    """
    def __init__(self, model_file="custom_gestures.pkl"):
        self.model_file = model_file
        self.gesture_data = []
        self.gesture_labels = []
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.gesture_names = {}
        
    def extract_features(self, landmarks):
        """Extract features from hand landmarks."""
        if not landmarks:
            return None
            
        # Convert landmarks to numpy array
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Normalize to wrist position
        wrist = points[0]
        normalized_points = points - wrist
        
        # Calculate distances between key points
        features = []
        
        # Finger tip positions relative to wrist
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        for tip in fingertips:
            features.extend(normalized_points[tip])
            
        # Distances between fingertips
        for i, tip1 in enumerate(fingertips):
            for tip2 in fingertips[i+1:]:
                dist = np.linalg.norm(normalized_points[tip1] - normalized_points[tip2])
                features.append(dist)
                
        # Angles between fingers
        for i in range(len(fingertips)-1):
            tip1, tip2 = fingertips[i], fingertips[i+1]
            v1 = normalized_points[tip1]
            v2 = normalized_points[tip2]
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
            features.append(angle)
            
        # Hand orientation
        palm_normal = self._calculate_palm_normal(normalized_points)
        features.extend(palm_normal)
        
        return np.array(features)
        
    def _calculate_palm_normal(self, points):
        """Calculate palm normal vector."""
        # Use three points to define palm plane
        p1, p2, p3 = points[0], points[5], points[17]  # Wrist, Index base, Pinky base
        
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
            
        return normal
        
    def add_training_sample(self, landmarks, gesture_name):
        """Add training sample for gesture."""
        features = self.extract_features(landmarks)
        if features is not None:
            self.gesture_data.append(features)
            
            # Convert gesture name to label
            if gesture_name not in self.gesture_names:
                self.gesture_names[gesture_name] = len(self.gesture_names)
                
            self.gesture_labels.append(self.gesture_names[gesture_name])
            print(f"Added training sample for gesture: {gesture_name}")
            
    def train_model(self):
        """Train the gesture recognition model."""
        if len(self.gesture_data) < 2:
            print("Not enough training data")
            return False
            
        X = np.array(self.gesture_data)
        y = np.array(self.gesture_labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Save model
        self.save_model()
        
        print(f"Model trained with {len(X)} samples for {len(self.gesture_names)} gestures")
        return True
        
    def predict_gesture(self, landmarks):
        """Predict gesture from landmarks."""
        features = self.extract_features(landmarks)
        if features is None:
            return None, 0.0
            
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        confidence = np.max(self.model.predict_proba(features_scaled))
        
        # Convert label back to gesture name
        gesture_name = None
        for name, label in self.gesture_names.items():
            if label == prediction:
                gesture_name = name
                break
                
        return gesture_name, confidence
        
    def save_model(self):
        """Save trained model to file."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'gesture_names': self.gesture_names,
            'training_samples': len(self.gesture_data),
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self):
        """Load trained model from file."""
        try:
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.gesture_names = model_data['gesture_names']
            
            print(f"Loaded model with {len(self.gesture_names)} gestures")
            return True
        except FileNotFoundError:
            print("No saved model found")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def get_training_stats(self):
        """Get training statistics."""
        stats = {}
        for gesture_name, label in self.gesture_names.items():
            count = sum(1 for l in self.gesture_labels if l == label)
            stats[gesture_name] = count
            
        return stats