import cv2
import numpy as np
import face_recognition
import pickle
import os
from datetime import datetime

class FaceRecognitionEngine:
    """
    Advanced face recognition with enrollment and identification capabilities.
    """
    def __init__(self, encodings_file="face_encodings.pkl"):
        self.encodings_file = encodings_file
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        
        # Load existing encodings
        self.load_encodings()
        
    def load_encodings(self):
        """Load face encodings from file."""
        try:
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print(f"Loaded {len(self.known_face_names)} known faces")
        except FileNotFoundError:
            print("No existing face encodings found")
        except Exception as e:
            print(f"Error loading face encodings: {e}")
            
    def save_encodings(self):
        """Save face encodings to file."""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            print("Face encodings saved successfully")
        except Exception as e:
            print(f"Error saving face encodings: {e}")
            
    def enroll_face(self, image, person_name):
        """Enroll a new face for recognition."""
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            print("No face found in the image")
            return False
            
        # Use the first face found
        face_encoding = face_encodings[0]
        
        # Check if person already exists
        if person_name in self.known_face_names:
            # Update existing encoding
            index = self.known_face_names.index(person_name)
            self.known_face_encodings[index] = face_encoding
            print(f"Updated face encoding for {person_name}")
        else:
            # Add new face
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(person_name)
            print(f"Enrolled new face: {person_name}")
            
        self.save_encodings()
        return True
        
    def recognize_faces(self, frame):
        """Recognize faces in a frame."""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find faces in the frame
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
        
        self.face_names = []
        for face_encoding in self.face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            
            # Use the known face with the smallest distance
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                name = self.known_face_names[best_match_index]
                
            self.face_names.append(name)
            
        # Scale back up face locations
        self.face_locations = [(top*4, right*4, bottom*4, left*4) 
                              for (top, right, bottom, left) in self.face_locations]
        
        return list(zip(self.face_locations, self.face_names))
        
    def draw_face_boxes(self, frame, face_data):
        """Draw bounding boxes and names on faces."""
        for (top, right, bottom, left), name in face_data:
            # Draw rectangle around face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
        return frame
        
    def get_known_faces(self):
        """Get list of known faces."""
        return self.known_face_names.copy()
        
    def remove_face(self, person_name):
        """Remove a face from the database."""
        if person_name in self.known_face_names:
            index = self.known_face_names.index(person_name)
            del self.known_face_encodings[index]
            del self.known_face_names[index]
            self.save_encodings()
            print(f"Removed face: {person_name}")
            return True
        else:
            print(f"Face not found: {person_name}")
            return False