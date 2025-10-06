"""
Detection utilities for JARVIS object detection system.
Provides helper functions for detection processing, filtering, and analysis.
"""

import cv2
import numpy as np
import time
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque

class DetectionFilter:
    """Filter and process detection results based on various criteria."""
    
    def __init__(self, config_path="detection_config.json"):
        self.config = self._load_config(config_path)
        self.detection_buffer = deque(maxlen=30)  # Buffer for smoothing
        
    def _load_config(self, config_path):
        """Load detection configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_config()
            
    def _get_default_config(self):
        """Get default configuration."""
        return {
            "detection_settings": {
                "confidence_threshold": 0.5,
                "max_detections": 100
            },
            "tracking": {
                "max_disappeared": 30
            }
        }
        
    def filter_by_confidence(self, detections, min_confidence=None):
        """Filter detections by confidence threshold."""
        if min_confidence is None:
            min_confidence = self.config["detection_settings"]["confidence_threshold"]
            
        return [d for d in detections if d['confidence'] >= min_confidence]
        
    def filter_by_class(self, detections, allowed_classes):
        """Filter detections by class names or IDs."""
        if isinstance(allowed_classes[0], str):
            # Filter by class names
            return [d for d in detections if d['class_name'] in allowed_classes]
        else:
            # Filter by class IDs
            return [d for d in detections if d['class_id'] in allowed_classes]
            
    def filter_by_size(self, detections, min_area=100, max_area=None):
        """Filter detections by bounding box area."""
        filtered = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            area = (x2 - x1) * (y2 - y1)
            
            if area >= min_area:
                if max_area is None or area <= max_area:
                    filtered.append(detection)
                    
        return filtered
        
    def non_max_suppression(self, detections, overlap_threshold=0.5):
        """Apply non-maximum suppression to remove overlapping detections."""
        if not detections:
            return []
            
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            # Keep the highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [d for d in detections 
                         if self._calculate_iou(current['bbox'], d['bbox']) < overlap_threshold]
                         
        return keep
        
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def smooth_detections(self, current_detections):
        """Smooth detections over time to reduce jitter."""
        self.detection_buffer.append(current_detections)
        
        if len(self.detection_buffer) < 3:
            return current_detections
            
        # Average detection positions over last few frames
        smoothed = []
        for current in current_detections:
            # Find similar detections in previous frames
            similar_detections = []
            for frame_detections in list(self.detection_buffer)[-3:]:
                for prev_detection in frame_detections:
                    if (prev_detection['class_id'] == current['class_id'] and
                        self._calculate_iou(current['bbox'], prev_detection['bbox']) > 0.3):
                        similar_detections.append(prev_detection)
                        
            if similar_detections:
                # Average the bounding boxes
                avg_bbox = self._average_bboxes([current] + similar_detections)
                smoothed_detection = current.copy()
                smoothed_detection['bbox'] = avg_bbox
                smoothed_detection['centroid'] = (
                    (avg_bbox[0] + avg_bbox[2]) // 2,
                    (avg_bbox[1] + avg_bbox[3]) // 2
                )
                smoothed.append(smoothed_detection)
            else:
                smoothed.append(current)
                
        return smoothed
        
    def _average_bboxes(self, detections):
        """Average multiple bounding boxes."""
        x1_sum = sum(d['bbox'][0] for d in detections)
        y1_sum = sum(d['bbox'][1] for d in detections)
        x2_sum = sum(d['bbox'][2] for d in detections)
        y2_sum = sum(d['bbox'][3] for d in detections)
        
        count = len(detections)
        return (
            x1_sum // count,
            y1_sum // count,
            x2_sum // count,
            y2_sum // count
        )

class DetectionAnalyzer:
    """Analyze detection patterns and provide insights."""
    
    def __init__(self):
        self.detection_history = []
        self.behavior_patterns = defaultdict(list)
        self.anomaly_detector = AnomalyDetector()
        
    def analyze_scene(self, detections):
        """Analyze current scene for patterns and anomalies."""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'object_count': len(detections),
            'object_types': {},
            'spatial_distribution': self._analyze_spatial_distribution(detections),
            'density': self._calculate_density(detections),
            'anomalies': []
        }
        
        # Count object types
        for detection in detections:
            class_name = detection['class_name']
            analysis['object_types'][class_name] = analysis['object_types'].get(class_name, 0) + 1
            
        # Detect anomalies
        analysis['anomalies'] = self.anomaly_detector.detect_anomalies(detections)
        
        # Store in history
        self.detection_history.append(analysis)
        
        return analysis
        
    def _analyze_spatial_distribution(self, detections):
        """Analyze how objects are distributed in the frame."""
        if not detections:
            return {'distribution': 'empty'}
            
        centroids = [d['centroid'] for d in detections]
        
        # Calculate center of mass
        center_x = sum(c[0] for c in centroids) / len(centroids)
        center_y = sum(c[1] for c in centroids) / len(centroids)
        
        # Calculate spread
        distances = [np.sqrt((c[0] - center_x)**2 + (c[1] - center_y)**2) for c in centroids]
        avg_distance = sum(distances) / len(distances)
        
        # Classify distribution
        if avg_distance < 50:
            distribution_type = 'clustered'
        elif avg_distance > 200:
            distribution_type = 'scattered'
        else:
            distribution_type = 'normal'
            
        return {
            'distribution': distribution_type,
            'center_of_mass': (int(center_x), int(center_y)),
            'average_spread': avg_distance
        }
        
    def _calculate_density(self, detections):
        """Calculate object density in the frame."""
        if not detections:
            return 0.0
            
        # Assume frame size (could be passed as parameter)
        frame_area = 640 * 480
        
        # Calculate total object area
        total_object_area = 0
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            area = (x2 - x1) * (y2 - y1)
            total_object_area += area
            
        return total_object_area / frame_area
        
    def get_behavior_patterns(self, object_id):
        """Get behavior patterns for a specific tracked object."""
        return self.behavior_patterns.get(object_id, [])
        
    def update_behavior_pattern(self, object_id, position, timestamp):
        """Update behavior pattern for a tracked object."""
        pattern_entry = {
            'position': position,
            'timestamp': timestamp,
            'velocity': self._calculate_velocity(object_id, position, timestamp)
        }
        
        self.behavior_patterns[object_id].append(pattern_entry)
        
        # Keep only recent patterns
        cutoff_time = datetime.now() - timedelta(minutes=10)
        self.behavior_patterns[object_id] = [
            p for p in self.behavior_patterns[object_id]
            if datetime.fromisoformat(p['timestamp']) > cutoff_time
        ]
        
    def _calculate_velocity(self, object_id, current_position, timestamp):
        """Calculate velocity based on position history."""
        if object_id not in self.behavior_patterns or not self.behavior_patterns[object_id]:
            return (0.0, 0.0)
            
        previous = self.behavior_patterns[object_id][-1]
        prev_pos = previous['position']
        prev_time = datetime.fromisoformat(previous['timestamp'])
        curr_time = datetime.fromisoformat(timestamp)
        
        time_diff = (curr_time - prev_time).total_seconds()
        if time_diff == 0:
            return (0.0, 0.0)
            
        velocity_x = (current_position[0] - prev_pos[0]) / time_diff
        velocity_y = (current_position[1] - prev_pos[1]) / time_diff
        
        return (velocity_x, velocity_y)

class AnomalyDetector:
    """Detect anomalous patterns in object detection."""
    
    def __init__(self):
        self.baseline_patterns = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        
    def detect_anomalies(self, detections):
        """Detect anomalies in current detections."""
        anomalies = []
        
        # Check for unusual object counts
        object_count_anomaly = self._check_object_count_anomaly(len(detections))
        if object_count_anomaly:
            anomalies.append(object_count_anomaly)
            
        # Check for unusual object types
        object_types = [d['class_name'] for d in detections]
        type_anomalies = self._check_object_type_anomalies(object_types)
        anomalies.extend(type_anomalies)
        
        # Check for unusual spatial patterns
        spatial_anomaly = self._check_spatial_anomalies(detections)
        if spatial_anomaly:
            anomalies.append(spatial_anomaly)
            
        return anomalies
        
    def _check_object_count_anomaly(self, count):
        """Check if object count is anomalous."""
        if 'object_count' not in self.baseline_patterns:
            return None
            
        baseline = self.baseline_patterns['object_count']
        if abs(count - baseline['mean']) > self.anomaly_threshold * baseline['std']:
            return {
                'type': 'object_count',
                'severity': 'high' if count > baseline['mean'] + 3 * baseline['std'] else 'medium',
                'description': f"Unusual object count: {count} (normal: {baseline['mean']:.1f}±{baseline['std']:.1f})"
            }
        return None
        
    def _check_object_type_anomalies(self, object_types):
        """Check for unusual object types."""
        anomalies = []
        
        for obj_type in set(object_types):
            if obj_type not in self.baseline_patterns.get('object_types', {}):
                anomalies.append({
                    'type': 'unusual_object',
                    'severity': 'medium',
                    'description': f"Unusual object type detected: {obj_type}"
                })
                
        return anomalies
        
    def _check_spatial_anomalies(self, detections):
        """Check for unusual spatial distributions."""
        if not detections or 'spatial' not in self.baseline_patterns:
            return None
            
        # Calculate current density
        centroids = [d['centroid'] for d in detections]
        center_x = sum(c[0] for c in centroids) / len(centroids)
        center_y = sum(c[1] for c in centroids) / len(centroids)
        
        baseline_center = self.baseline_patterns['spatial']['center']
        distance_from_baseline = np.sqrt(
            (center_x - baseline_center[0])**2 + 
            (center_y - baseline_center[1])**2
        )
        
        if distance_from_baseline > 100:  # Arbitrary threshold
            return {
                'type': 'spatial_anomaly',
                'severity': 'low',
                'description': f"Objects concentrated in unusual area"
            }
            
        return None
        
    def update_baseline(self, detections):
        """Update baseline patterns with new data."""
        # Update object count baseline
        if 'object_count' not in self.baseline_patterns:
            self.baseline_patterns['object_count'] = {'values': [], 'mean': 0, 'std': 1}
            
        count_data = self.baseline_patterns['object_count']
        count_data['values'].append(len(detections))
        count_data['values'] = count_data['values'][-100:]  # Keep last 100 values
        
        if len(count_data['values']) > 10:
            count_data['mean'] = np.mean(count_data['values'])
            count_data['std'] = np.std(count_data['values']) or 1
            
        # Update object types baseline
        if 'object_types' not in self.baseline_patterns:
            self.baseline_patterns['object_types'] = set()
            
        for detection in detections:
            self.baseline_patterns['object_types'].add(detection['class_name'])
            
        # Update spatial baseline
        if detections:
            centroids = [d['centroid'] for d in detections]
            center_x = sum(c[0] for c in centroids) / len(centroids)
            center_y = sum(c[1] for c in centroids) / len(centroids)
            
            if 'spatial' not in self.baseline_patterns:
                self.baseline_patterns['spatial'] = {'centers': [], 'center': (center_x, center_y)}
                
            spatial_data = self.baseline_patterns['spatial']
            spatial_data['centers'].append((center_x, center_y))
            spatial_data['centers'] = spatial_data['centers'][-50:]  # Keep last 50 centers
            
            # Update average center
            if len(spatial_data['centers']) > 5:
                avg_x = sum(c[0] for c in spatial_data['centers']) / len(spatial_data['centers'])
                avg_y = sum(c[1] for c in spatial_data['centers']) / len(spatial_data['centers'])
                spatial_data['center'] = (avg_x, avg_y)

class DetectionRecorder:
    """Record and replay detection sessions."""
    
    def __init__(self, output_dir="detection_recordings"):
        self.output_dir = output_dir
        self.is_recording = False
        self.current_session = None
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def start_recording(self, session_name=None):
        """Start recording detection session."""
        if session_name is None:
            session_name = f"detection_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.current_session = {
            'name': session_name,
            'start_time': datetime.now().isoformat(),
            'frames': [],
            'metadata': {}
        }
        
        self.is_recording = True
        print(f"Started recording session: {session_name}")
        
    def record_frame(self, frame, detections):
        """Record a frame with its detections."""
        if not self.is_recording or not self.current_session:
            return
            
        frame_data = {
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'frame_shape': frame.shape
        }
        
        self.current_session['frames'].append(frame_data)
        
        # Save frame image
        frame_filename = f"{self.current_session['name']}_frame_{len(self.current_session['frames']):06d}.jpg"
        frame_path = f"{self.output_dir}/{frame_filename}"
        cv2.imwrite(frame_path, frame)
        
        frame_data['frame_file'] = frame_filename
        
    def stop_recording(self):
        """Stop recording and save session data."""
        if not self.is_recording or not self.current_session:
            return
            
        self.current_session['end_time'] = datetime.now().isoformat()
        self.current_session['duration'] = len(self.current_session['frames'])
        
        # Save session metadata
        session_file = f"{self.output_dir}/{self.current_session['name']}_metadata.json"
        with open(session_file, 'w') as f:
            json.dump(self.current_session, f, indent=2)
            
        print(f"Recording saved: {session_file}")
        print(f"Recorded {len(self.current_session['frames'])} frames")
        
        self.is_recording = False
        self.current_session = None
        
    def load_session(self, session_file):
        """Load a recorded session."""
        with open(session_file, 'r') as f:
            return json.load(f)
            
    def replay_session(self, session_data, callback=None):
        """Replay a recorded session."""
        print(f"Replaying session: {session_data['name']}")
        
        for frame_data in session_data['frames']:
            # Load frame image
            frame_path = f"{self.output_dir}/{frame_data['frame_file']}"
            frame = cv2.imread(frame_path)
            
            if frame is not None and callback:
                callback(frame, frame_data['detections'])
                
            # Wait for natural playback speed
            time.sleep(0.033)  # ~30 FPS