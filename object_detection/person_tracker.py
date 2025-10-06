import cv2
import numpy as np
from collections import defaultdict
import math

class PersonTracker:
    """
    Advanced person tracking across video frames using centroid tracking.
    """
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = {}  # objectID -> centroid
        self.disappeared = {}  # objectID -> frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Person attributes
        self.person_attributes = {}  # objectID -> attributes dict
        
    def register(self, centroid, bbox=None, attributes=None):
        """Register a new object with given centroid."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        
        # Store additional attributes
        if attributes is None:
            attributes = {}
        if bbox is not None:
            attributes['bbox'] = bbox
        attributes['first_seen'] = cv2.getTickCount()
        
        self.person_attributes[self.next_object_id] = attributes
        
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Deregister an object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.person_attributes:
            del self.person_attributes[object_id]
            
    def update(self, detections):
        """
        Update tracker with new detections.
        detections: list of (centroid, bbox, attributes) tuples
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Remove if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_objects()
            
        # Initialize input centroids array
        input_centroids = []
        input_bboxes = []
        input_attributes = []
        
        for detection in detections:
            if len(detection) >= 2:
                input_centroids.append(detection[0])
                input_bboxes.append(detection[1] if len(detection) > 1 else None)
                input_attributes.append(detection[2] if len(detection) > 2 else {})
            
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                bbox = input_bboxes[i] if i < len(input_bboxes) else None
                attrs = input_attributes[i] if i < len(input_attributes) else {}
                self.register(centroid, bbox, attrs)
                
        else:
            # Match existing objects to detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance matrix
            D = self._compute_distance_matrix(object_centroids, input_centroids)
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # Track used row and column indices
            used_row_indices = set()
            used_col_indices = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                    
                # Check if distance is acceptable
                if D[row, col] > self.max_distance:
                    continue
                    
                # Update object
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Update attributes
                if col < len(input_bboxes) and input_bboxes[col] is not None:
                    self.person_attributes[object_id]['bbox'] = input_bboxes[col]
                if col < len(input_attributes):
                    self.person_attributes[object_id].update(input_attributes[col])
                
                used_row_indices.add(row)
                used_col_indices.add(col)
                
            # Handle unmatched detections and objects
            unused_rows = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_cols = set(range(0, D.shape[1])).difference(used_col_indices)
            
            # If more objects than detections, mark as disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
                        
            # If more detections than objects, register new objects
            else:
                for col in unused_cols:
                    bbox = input_bboxes[col] if col < len(input_bboxes) else None
                    attrs = input_attributes[col] if col < len(input_attributes) else {}
                    self.register(input_centroids[col], bbox, attrs)
                    
        return self.get_objects()
        
    def _compute_distance_matrix(self, object_centroids, input_centroids):
        """Compute distance matrix between object centroids and input centroids."""
        D = np.zeros((len(object_centroids), len(input_centroids)))
        
        for i, obj_centroid in enumerate(object_centroids):
            for j, input_centroid in enumerate(input_centroids):
                D[i, j] = math.sqrt(
                    (obj_centroid[0] - input_centroid[0]) ** 2 +
                    (obj_centroid[1] - input_centroid[1]) ** 2
                )
                
        return D
        
    def get_objects(self):
        """Get current tracked objects."""
        result = {}
        for object_id in self.objects:
            result[object_id] = {
                'centroid': self.objects[object_id],
                'attributes': self.person_attributes.get(object_id, {})
            }
        return result
        
    def get_object_path(self, object_id, max_points=50):
        """Get tracking path for an object."""
        # This would require storing historical positions
        # For now, return empty list
        return []
        
    def get_tracking_stats(self):
        """Get tracking statistics."""
        active_tracks = len(self.objects)
        total_tracks_created = self.next_object_id
        
        return {
            'active_tracks': active_tracks,
            'total_tracks_created': total_tracks_created,
            'max_disappeared': self.max_disappeared,
            'max_distance': self.max_distance
        }