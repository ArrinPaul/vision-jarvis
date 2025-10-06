# JARVIS Object Detection Module

## Overview

The Object Detection module is a comprehensive computer vision system that provides real-time object and person detection, face recognition, person tracking, and intelligent scene analysis. It serves as the visual perception layer for the JARVIS AI assistant.

## Features

### 🎯 Core Detection Capabilities
- **YOLO-based Object Detection**: Real-time detection of 80+ object classes
- **Advanced Face Recognition**: Face enrollment, identification, and tracking
- **Person Tracking**: Multi-person tracking with unique ID assignment
- **Scene Analysis**: Contextual understanding of detected objects and scenes

### 🔍 Advanced Analysis
- **Anomaly Detection**: Identifies unusual patterns or objects
- **Behavior Analysis**: Tracks movement patterns and behaviors
- **Density Analysis**: Calculates object density and spatial distribution
- **Historical Tracking**: Maintains detection history and statistics

### ⚡ Real-time Processing
- **Multi-threaded Architecture**: Non-blocking detection processing
- **Callback System**: Event-driven detection notifications
- **Performance Optimization**: Configurable detection parameters
- **Live Visualization**: Real-time detection overlay and annotations

## Architecture

```
ObjectDetector (Main Class)
├── YOLOv8 Model (ultralytics)
├── FaceRecognitionEngine
│   ├── Face Enrollment
│   ├── Face Identification
│   └── Encoding Storage
├── PersonTracker
│   ├── Centroid Tracking
│   ├── ID Assignment
│   └── Disappearance Handling
└── DetectionUtils
    ├── DetectionFilter
    ├── DetectionAnalyzer
    ├── AnomalyDetector
    └── DetectionRecorder
```

## Installation

### Prerequisites
```bash
pip install ultralytics opencv-python face-recognition numpy scikit-learn
```

### Model Download
The module uses YOLOv8n model which will be automatically downloaded on first use:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically
```

## Quick Start

### Basic Object Detection
```python
from object_detection import ObjectDetector

# Initialize detector
detector = ObjectDetector()

# Start detection (blocking)
detector.detect_objects()
```

### Advanced Usage with Callbacks
```python
from object_detection import ObjectDetector

def detection_callback(frame, results):
    print(f"Detected {len(results['objects'])} objects")
    print(f"Tracked {len(results['tracked_persons'])} persons")

# Initialize and configure
detector = ObjectDetector()
detector.add_detection_callback(detection_callback)

# Start non-blocking detection
detector.start_detection()

# Your code here...

# Stop detection
detector.stop_detection()
```

### Face Recognition Setup
```python
# Enroll faces during runtime
# Press 'e' during detection to enter enrollment mode
# Follow prompts to capture and name faces

# Or programmatically:
import cv2
detector = ObjectDetector()
frame = cv2.imread("person_photo.jpg")
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
detector.face_recognition.enroll_face(rgb_frame, "John Doe")
```

## Configuration

### Main Configuration (detection_config.json)
```json
{
  "detection_settings": {
    "confidence_threshold": 0.5,
    "model_path": "yolov8n.pt",
    "max_detections": 100
  },
  "face_recognition": {
    "tolerance": 0.6,
    "model": "hog"
  },
  "tracking": {
    "max_disappeared": 30,
    "max_distance": 100
  },
  "alerts": {
    "unknown_person_alert": true,
    "crowd_threshold": 5
  }
}
```

### Key Parameters
- **confidence_threshold**: Minimum confidence for object detection (0.0-1.0)
- **tolerance**: Face recognition tolerance (lower = stricter matching)
- **max_disappeared**: Frames before removing lost tracks
- **max_distance**: Maximum pixel distance for track association

## API Reference

### ObjectDetector Class

#### Methods
```python
# Core detection
start_detection(callback=None)           # Start detection in background
stop_detection()                         # Stop detection
process_frame(frame)                     # Process single frame
detect_objects()                         # Start blocking detection

# Callbacks and events
add_detection_callback(callback)         # Add detection event callback

# Statistics and analysis
get_detection_stats()                    # Get detection statistics
track_object(object_id)                  # Get track information

# Face recognition
# Access via detector.face_recognition
enroll_face(rgb_image, name)            # Enroll new face
recognize_faces(rgb_image)              # Recognize faces in image
get_known_faces()                       # Get list of enrolled faces
```

#### Detection Results Format
```python
{
    'objects': [
        {
            'bbox': (x1, y1, x2, y2),
            'confidence': 0.85,
            'class_id': 0,
            'class_name': 'person',
            'centroid': (x_center, y_center),
            'face_name': 'John Doe'  # If face recognized
        }
    ],
    'faces': [
        ((top, right, bottom, left), 'John Doe')
    ],
    'tracked_persons': {
        0: {
            'centroid': (x, y),
            'bbox': (x1, y1, x2, y2),
            'attributes': {'face_name': 'John Doe'}
        }
    },
    'scene_analysis': {
        'object_counts': {'person': 2, 'car': 1},
        'total_objects': 3,
        'known_persons': 1,
        'unknown_persons': 1,
        'alerts': ['Unknown person detected']
    },
    'timestamp': '2024-01-01T12:00:00'
}
```

### FaceRecognitionEngine Class

#### Methods
```python
enroll_face(rgb_image, name)            # Enroll new face
recognize_faces(rgb_image)              # Recognize faces
get_known_faces()                       # List enrolled faces
remove_face(name)                       # Remove enrolled face
update_face(old_name, new_name)         # Update face name
```

### PersonTracker Class

#### Methods
```python
update(input_objects)                   # Update tracker with detections
get_track_info(track_id)               # Get track information
reset()                                # Reset all tracks
```

## Detection Utilities

### DetectionFilter
```python
from detection_utils import DetectionFilter

filter = DetectionFilter()

# Filter by confidence
high_conf = filter.filter_by_confidence(detections, 0.8)

# Filter by object class
persons_only = filter.filter_by_class(detections, ['person'])

# Filter by size
large_objects = filter.filter_by_size(detections, min_area=1000)

# Apply non-maximum suppression
filtered = filter.non_max_suppression(detections, 0.5)

# Smooth detections over time
smoothed = filter.smooth_detections(detections)
```

### DetectionAnalyzer
```python
from detection_utils import DetectionAnalyzer

analyzer = DetectionAnalyzer()

# Analyze scene
analysis = analyzer.analyze_scene(detections)

# Update behavior patterns
analyzer.update_behavior_pattern(track_id, position, timestamp)

# Get behavior history
patterns = analyzer.get_behavior_patterns(track_id)
```

### DetectionRecorder
```python
from detection_utils import DetectionRecorder

recorder = DetectionRecorder("recordings/")

# Record session
recorder.start_recording("session_name")
recorder.record_frame(frame, detections)
recorder.stop_recording()

# Replay session
session_data = recorder.load_session("session_metadata.json")
recorder.replay_session(session_data, callback_function)
```

## Interactive Controls

During detection, the following keyboard controls are available:

- **Q**: Quit detection
- **E**: Enter face enrollment mode
- **S**: Save detection snapshot
- **C**: Capture face (during enrollment)
- **ESC**: Cancel current operation

## Performance Optimization

### Hardware Acceleration
```python
# Use GPU if available (requires CUDA)
detector = ObjectDetector()
detector.model.to('cuda')  # Enable GPU processing
```

### Frame Processing
```python
# Reduce frame size for better performance
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### Detection Filtering
```python
# Filter detections to reduce processing
detector.confidence_threshold = 0.7  # Higher threshold
detector.object_classes = [0]  # Only detect persons
```

## Integration Examples

### With Voice Assistant
```python
def detection_callback(frame, results):
    if results['scene_analysis']['unknown_persons'] > 0:
        voice_assistant.speak("Unknown person detected")
    
    known_persons = [
        track['attributes']['face_name'] 
        for track in results['tracked_persons'].values()
        if track['attributes'].get('face_name', 'Unknown') != 'Unknown'
    ]
    
    if known_persons:
        voice_assistant.speak(f"Hello {', '.join(known_persons)}")

detector.add_detection_callback(detection_callback)
```

### With Smart Home
```python
def security_callback(frame, results):
    alerts = results['scene_analysis']['alerts']
    
    if any('unknown person' in alert.lower() for alert in alerts):
        smart_home.trigger_security_alert()
        smart_home.turn_on_lights()
        smart_home.send_notification("Security Alert: Unknown person detected")

detector.add_detection_callback(security_callback)
```

### With AR Interface
```python
def ar_callback(frame, results):
    for obj in results['objects']:
        if obj['class_name'] == 'person':
            name = obj.get('face_name', 'Unknown Person')
            ar_interface.add_overlay(
                position=obj['centroid'],
                text=name,
                style='person_label'
            )

detector.add_detection_callback(ar_callback)
```

## Troubleshooting

### Common Issues

1. **Face Recognition Not Working**
   ```bash
   # Install face_recognition properly
   pip uninstall face-recognition
   pip install face-recognition
   ```

2. **YOLO Model Download Issues**
   ```python
   # Manually download model
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')  # Will download to ~/.ultralytics/
   ```

3. **Camera Access Issues**
   ```python
   # Try different camera indices
   cap = cv2.VideoCapture(1)  # Try 1, 2, etc.
   ```

4. **Performance Issues**
   - Reduce frame resolution
   - Increase confidence threshold
   - Limit detection classes
   - Use smaller YOLO model (yolov8n vs yolov8x)

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose detection
detector = ObjectDetector()
detector.debug_mode = True
```

## File Structure

```
object_detection/
├── __init__.py                 # Module initialization
├── object_detection.py         # Main ObjectDetector class
├── face_recognition.py         # Face recognition engine
├── person_tracker.py          # Person tracking system
├── detection_utils.py         # Utility classes and functions
├── detection_config.json      # Configuration file
├── test_object_detection.py   # Comprehensive test suite
└── README.md                  # This documentation
```

## Testing

Run the comprehensive test suite:
```bash
cd object_detection/
python test_object_detection.py
```

The test suite covers:
- Object detection functionality
- Face recognition engine
- Person tracking system
- Detection utilities
- Integration scenarios
- Performance benchmarks

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update configuration schema for new parameters
4. Document all public methods and classes
5. Ensure backward compatibility

## License

This module is part of the JARVIS AI Assistant project and follows the same licensing terms.

---

*For more information about the complete JARVIS system, see the main project documentation.*