import cv2
import numpy as np
import threading
import time
from datetime import datetime
from ultralytics import YOLO
from .face_recognition import FaceRecognitionEngine
from .person_tracker import PersonTracker

class ObjectDetector:
    """
    Advanced Real-Time Object/Person Detection with face recognition, tracking, and contextual actions.
    Features: YOLO detection, face recognition, person tracking, and intelligent responses.
    """
    def __init__(self, model_path="yolov8n.pt", config_file="detection_config.json"):
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Initialize face recognition
        self.face_recognition = FaceRecognitionEngine()
        
        # Initialize person tracker
        self.person_tracker = PersonTracker(max_disappeared=30, max_distance=100)
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.is_detecting = False
        self.detection_callbacks = []
        
        # Object categories of interest
        self.person_classes = [0]  # COCO class ID for person
        self.object_classes = list(range(80))  # All COCO classes
        
        # Detection history
        self.detection_history = []
        self.max_history = 1000
        
        # Context analysis
        self.scene_context = {}
        self.alert_conditions = {}
        
    def start_detection(self, callback=None):
        """Start object detection in a separate thread."""
        if self.is_detecting:
            return
            
        self.is_detecting = True
        if callback:
            self.detection_callbacks.append(callback)
            
        threading.Thread(target=self._detection_loop, daemon=True).start()
        
    def stop_detection(self):
        """Stop object detection."""
        self.is_detecting = False
        
    def _detection_loop(self):
        """Main detection loop."""
        cap = cv2.VideoCapture(0)
        
        try:
            while self.is_detecting:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Perform detection
                detection_results = self.process_frame(frame)
                
                # Call callbacks
                for callback in self.detection_callbacks:
                    callback(frame, detection_results)
                    
                # Display frame
                self.draw_detections(frame, detection_results)
                cv2.imshow("JARVIS Object Detection", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('e'):
                    self._enter_face_enrollment_mode()
                elif key == ord('s'):
                    self._save_detection_snapshot(frame, detection_results)
                    
        except Exception as e:
            print(f"Detection error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.is_detecting = False
            
    def process_frame(self, frame):
        """Process a single frame for object and face detection."""
        # YOLO object detection
        yolo_results = self.model(frame, conf=self.confidence_threshold)
        
        # Extract detections
        detections = []
        person_detections = []
        
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get class name
                    class_name = self.model.names[class_id]
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'centroid': ((x1 + x2) // 2, (y1 + y2) // 2)
                    }
                    
                    detections.append(detection)
                    
                    # Track persons separately
                    if class_id in self.person_classes:
                        person_detections.append(detection)
                        
        # Face recognition on detected persons
        face_results = []
        if person_detections:
            face_data = self.face_recognition.recognize_faces(frame)
            face_results = face_data
            
            # Associate faces with person detections
            for face_location, face_name in face_data:
                # Find closest person detection
                face_centroid = (
                    (face_location[3] + face_location[1]) // 2,  # left + right / 2
                    (face_location[0] + face_location[2]) // 2   # top + bottom / 2
                )
                
                closest_person = None
                min_distance = float('inf')
                
                for person in person_detections:
                    distance = np.sqrt(
                        (person['centroid'][0] - face_centroid[0]) ** 2 +
                        (person['centroid'][1] - face_centroid[1]) ** 2
                    )
                    if distance < min_distance:
                        min_distance = distance
                        closest_person = person
                        
                if closest_person and min_distance < 100:
                    closest_person['face_name'] = face_name
                    closest_person['face_location'] = face_location
                    
        # Update person tracker
        tracker_inputs = []
        for person in person_detections:
            attributes = {
                'class_name': person['class_name'],
                'confidence': person['confidence'],
                'face_name': person.get('face_name', 'Unknown')
            }
            tracker_inputs.append((person['centroid'], person['bbox'], attributes))
            
        tracked_objects = self.person_tracker.update(tracker_inputs)
        
        # Analyze scene context
        scene_analysis = self._analyze_scene_context(detections, tracked_objects)
        
        # Store detection history
        self._store_detection_history(detections, face_results, tracked_objects)
        
        return {
            'objects': detections,
            'faces': face_results,
            'tracked_persons': tracked_objects,
            'scene_analysis': scene_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
    def draw_detections(self, frame, detection_results):
        """Draw detection results on frame."""
        # Draw object detections
        for obj in detection_results['objects']:
            x1, y1, x2, y2 = obj['bbox']
            confidence = obj['confidence']
            class_name = obj['class_name']
            
            # Choose color based on class
            if obj['class_id'] in self.person_classes:
                color = (0, 255, 0)  # Green for persons
            else:
                color = (255, 0, 0)  # Blue for other objects
                
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            if 'face_name' in obj and obj['face_name'] != 'Unknown':
                label += f" ({obj['face_name']})"
                
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        # Draw face recognition results
        for face_location, face_name in detection_results['faces']:
            top, right, bottom, left = face_location
            color = (0, 255, 0) if face_name != "Unknown" else (0, 0, 255)
            
            # Draw face rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw name label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, face_name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                       
        # Draw tracking IDs
        for track_id, track_data in detection_results['tracked_persons'].items():
            centroid = track_data['centroid']
            face_name = track_data['attributes'].get('face_name', 'Unknown')
            
            # Draw tracking ID
            cv2.circle(frame, centroid, 5, (255, 255, 0), -1)
            cv2.putText(frame, f"ID:{track_id}", (centroid[0] - 20, centroid[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                       
        # Draw scene analysis
        analysis = detection_results['scene_analysis']
        y_offset = 30
        for key, value in analysis.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
    def _analyze_scene_context(self, detections, tracked_objects):
        """Analyze scene context and generate insights."""
        analysis = {}
        
        # Count objects by type
        object_counts = {}
        for obj in detections:
            class_name = obj['class_name']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
        analysis['object_counts'] = object_counts
        analysis['total_objects'] = len(detections)
        analysis['active_tracks'] = len(tracked_objects)
        
        # Analyze person behavior
        known_persons = sum(1 for track in tracked_objects.values() 
                          if track['attributes'].get('face_name', 'Unknown') != 'Unknown')
        unknown_persons = len(tracked_objects) - known_persons
        
        analysis['known_persons'] = known_persons
        analysis['unknown_persons'] = unknown_persons
        
        # Check for alert conditions
        alerts = []
        if unknown_persons > 0:
            alerts.append(f"{unknown_persons} unknown person(s) detected")
        if len(tracked_objects) > 5:
            alerts.append("Crowded scene detected")
            
        analysis['alerts'] = alerts
        
        return analysis
        
    def _store_detection_history(self, detections, faces, tracked_objects):
        """Store detection results in history."""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'objects': len(detections),
            'faces': len(faces),
            'tracked_persons': len(tracked_objects),
            'scene_summary': self._generate_scene_summary(detections, faces, tracked_objects)
        }
        
        self.detection_history.append(history_entry)
        
        # Limit history size
        if len(self.detection_history) > self.max_history:
            self.detection_history = self.detection_history[-self.max_history:]
            
    def _generate_scene_summary(self, detections, faces, tracked_objects):
        """Generate a text summary of the current scene."""
        object_types = set(obj['class_name'] for obj in detections)
        known_faces = [name for _, name in faces if name != 'Unknown']
        
        summary = f"{len(detections)} objects detected"
        if object_types:
            summary += f" ({', '.join(list(object_types)[:3])})"
        if known_faces:
            summary += f", known persons: {', '.join(known_faces[:3])}"
        if len(tracked_objects) > 0:
            summary += f", {len(tracked_objects)} person(s) tracked"
            
        return summary
        
    def _enter_face_enrollment_mode(self):
        """Enter face enrollment mode."""
        print("Entering face enrollment mode...")
        person_name = input("Enter person's name: ")
        
        cap = cv2.VideoCapture(0)
        enrolled = False
        
        print("Position face in frame and press 'c' to capture, 'q' to cancel")
        
        while not enrolled:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            
            # Show instructions
            cv2.putText(frame, f"Enrolling: {person_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture, 'q' to cancel", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Face Enrollment", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Enroll face
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.face_recognition.enroll_face(rgb_frame, person_name):
                    print(f"Successfully enrolled {person_name}")
                    enrolled = True
                else:
                    print("Failed to enroll face. Try again.")
            elif key == ord('q'):
                print("Face enrollment cancelled")
                break
                
        cap.release()
        cv2.destroyWindow("Face Enrollment")
        
    def _save_detection_snapshot(self, frame, detection_results):
        """Save current frame and detection data as snapshot."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save image
        image_filename = f"detection_snapshot_{timestamp}.jpg"
        cv2.imwrite(image_filename, frame)
        
        # Save detection data
        data_filename = f"detection_data_{timestamp}.json"
        import json
        with open(data_filename, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in detection_results.items():
                if key == 'objects':
                    serializable_results[key] = [
                        {k: v.tolist() if isinstance(v, np.ndarray) else v 
                         for k, v in obj.items()} for obj in value
                    ]
                else:
                    serializable_results[key] = value
                    
            json.dump(serializable_results, f, indent=2)
            
        print(f"Snapshot saved: {image_filename}, {data_filename}")
        
    def add_detection_callback(self, callback):
        """Add callback function for detection events."""
        self.detection_callbacks.append(callback)
        
    def get_detection_stats(self):
        """Get detection statistics."""
        if not self.detection_history:
            return {}
            
        recent_history = self.detection_history[-100:]  # Last 100 detections
        
        avg_objects = sum(h['objects'] for h in recent_history) / len(recent_history)
        avg_faces = sum(h['faces'] for h in recent_history) / len(recent_history)
        avg_tracked = sum(h['tracked_persons'] for h in recent_history) / len(recent_history)
        
        return {
            'total_detections': len(self.detection_history),
            'avg_objects_per_frame': avg_objects,
            'avg_faces_per_frame': avg_faces,
            'avg_tracked_persons': avg_tracked,
            'known_faces': len(self.face_recognition.get_known_faces()),
            'active_tracks': len(self.person_tracker.objects)
        }
        
    def detect_objects(self):
        """Legacy method for backward compatibility."""
        self.start_detection()
        
        # Keep running until stopped
        try:
            while self.is_detecting:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_detection()
            
    def track_object(self, object_id):
        """Get tracking information for specific object."""
        if object_id in self.person_tracker.objects:
            return self.person_tracker.objects[object_id]
        else:
            print(f"Object {object_id} not currently tracked")
            return None
