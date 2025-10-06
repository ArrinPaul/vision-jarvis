import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
import json
import os
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Try to import environment config
try:
    from env_config import get_config
    ENV_CONFIG_AVAILABLE = True
except ImportError:
    ENV_CONFIG_AVAILABLE = False


class ObjectDetector:
    """Enhanced ONNX-based object detection with improved accuracy and performance"""

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4, class_names: Optional[List[str]] = None):
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        self.class_names = class_names or []

        # Load configuration if available
        if ENV_CONFIG_AVAILABLE:
            config = get_config()
            vision_config = config.get_config('vision')
            if not model_path:
                model_path = vision_config.get('model_path')
            self.confidence_threshold = vision_config.get('confidence_threshold', confidence_threshold)

        # Load class names from file if not provided
        if not self.class_names:
            self._load_class_names()

        if model_path and ONNX_AVAILABLE:
            self._load_model(model_path)
    
    def _load_class_names(self):
        """Load class names from JSON file"""
        try:
            class_file = Path("models/coco_classes.json")
            if class_file.exists():
                with open(class_file, 'r', encoding='utf-8') as f:
                    self.class_names = json.load(f)
                self.logger.info(f"Loaded {len(self.class_names)} class names")
            else:
                # Fallback to default COCO classes
                self.class_names = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                    'toothbrush'
                ]
                self.logger.warning("Using default COCO class names")
        except Exception as e:
            self.logger.error(f"Failed to load class names: {e}")
            self.class_names = []

    def _load_model(self, model_path: str):
        """Load ONNX model with enhanced error handling"""
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                self.logger.info("Run 'python download_models.py' to download required models")
                return

            # Set up ONNX Runtime session with optimizations
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers():
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers.insert(0, 'CUDAExecutionProvider')
                elif 'DmlExecutionProvider' in available_providers:
                    providers.insert(0, 'DmlExecutionProvider')

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.input_shape = self.session.get_inputs()[0].shape

            self.logger.info(f"✅ ONNX model loaded: {model_path}")
            self.logger.info(f"Input shape: {self.input_shape}")
            self.logger.info(f"Providers: {self.session.get_providers()}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load ONNX model: {e}")
            self.session = None
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """Detect objects in image and return list of detections with enhanced accuracy"""
        if not self.session:
            return []

        try:
            # Preprocess image
            input_image, scale_factor, pad_info = self._preprocess_image(image)

            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_image})

            # Post-process results
            detections = self._postprocess_outputs(outputs, image.shape, scale_factor, pad_info)

            # Filter by confidence and apply NMS
            filtered_detections = self._filter_and_nms(detections)

            return filtered_detections

        except Exception as e:
            self.logger.error(f"Object detection error: {e}")
            return []

    def detect_objects_with_context(self, image: np.ndarray, context: Optional[Dict] = None) -> Dict:
        """Detect objects with additional context information"""
        detections = self.detect_objects(image)

        result = {
            'detections': detections,
            'count': len(detections),
            'classes_detected': list(set([d['class_name'] for d in detections])),
            'confidence_stats': {
                'max': max([d['confidence'] for d in detections]) if detections else 0,
                'min': min([d['confidence'] for d in detections]) if detections else 0,
                'avg': sum([d['confidence'] for d in detections]) / len(detections) if detections else 0
            },
            'timestamp': None  # Can be set by caller
        }

        # Add context-specific analysis
        if context:
            result['context'] = self._analyze_context(detections, context)

        return result

    def _analyze_context(self, detections: List[Dict], context: Dict) -> Dict:
        """Analyze detections in context (e.g., for voice assistant responses)"""
        analysis = {
            'scene_type': 'unknown',
            'notable_objects': [],
            'people_count': 0,
            'vehicle_count': 0,
            'animal_count': 0
        }

        # Count different types of objects
        for detection in detections:
            class_name = detection['class_name'].lower()

            if class_name == 'person':
                analysis['people_count'] += 1
            elif class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'airplane', 'boat', 'train']:
                analysis['vehicle_count'] += 1
            elif class_name in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:
                analysis['animal_count'] += 1

            # Mark high-confidence detections as notable
            if detection['confidence'] > 0.8:
                analysis['notable_objects'].append(class_name)

        # Determine scene type
        if analysis['people_count'] > 0:
            if analysis['vehicle_count'] > 0:
                analysis['scene_type'] = 'street_or_parking'
            elif any(obj in analysis['notable_objects'] for obj in ['chair', 'table', 'couch', 'bed']):
                analysis['scene_type'] = 'indoor'
            else:
                analysis['scene_type'] = 'outdoor_people'
        elif analysis['vehicle_count'] > 0:
            analysis['scene_type'] = 'traffic_or_parking'
        elif analysis['animal_count'] > 0:
            analysis['scene_type'] = 'nature_or_pets'

        return analysis
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Enhanced preprocessing for YOLOv8 with proper scaling and padding"""
        # Determine model input dimensions (width, height)
        if self.input_shape and len(self.input_shape) >= 4:
            model_h = int(self.input_shape[2])
            model_w = int(self.input_shape[3])
        else:
            model_w, model_h = 640, 640

        # Calculate scaling factor while maintaining aspect ratio
        orig_h, orig_w = image.shape[:2]
        scale = min(model_w / orig_w, model_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calculate padding (left/top)
        pad_x = (model_w - new_w) // 2
        pad_y = (model_h - new_h) // 2

        # Create padded image (letterbox)
        padded = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Convert to model input format (NCHW, float32, normalized)
        input_image = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        return input_image, scale, (pad_x, pad_y)
    
    def _postprocess_outputs(self, outputs: List[np.ndarray], original_shape: Tuple,
                             scale: float, pad: Tuple[int, int]) -> List[Dict]:
        """Post-process YOLOv8 outputs and map boxes back to original image"""
        detections = []
        if not outputs:
            return detections

        preds = outputs[0]
        if isinstance(preds, list):
            preds = preds[0]

        # YOLOv8 ONNX output commonly has shape (1, num, 85)
        if len(preds.shape) == 3:
            preds = preds[0]

        for det in preds:
            # Box and scores
            x_center, y_center, w, h = det[:4]
            obj_conf = det[4]
            class_scores = det[5:]
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            conf = float(obj_conf) * class_conf

            if conf < self.confidence_threshold:
                continue

            # Convert from center-width-height to x1y1x2y2 in model space
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            # Map back to original image coordinates: reverse letterbox
            pad_x, pad_y = pad
            # Remove padding then scale back
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # Clamp to image bounds
            H, W = original_shape[:2]
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))

            detections.append({
                "class_id": class_id,
                "class_name": self._get_class_name(class_id),
                "confidence": conf,
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

        return detections
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from loaded class list or fallback COCO names"""
        if self.class_names and 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        # Fallback minimal list
        fallback = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe"
        ]
        if 0 <= class_id < len(fallback):
            return fallback[class_id]
        return f"class_{class_id}"

    def _filter_and_nms(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections by confidence and apply Non-Maximum Suppression"""
        if not detections:
            return []

        # Group by class for class-wise NMS
        final_dets: List[Dict] = []
        by_class: Dict[int, List[Dict]] = {}
        for det in detections:
            by_class.setdefault(det['class_id'], []).append(det)

        for class_id, dets in by_class.items():
            boxes = np.array([d['bbox'] for d in dets], dtype=np.float32)
            scores = np.array([d['confidence'] for d in dets], dtype=np.float32)

            # Convert to x, y, w, h for OpenCV NMSBoxes
            x = boxes[:, 0]
            y = boxes[:, 1]
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            rects = np.stack([x, y, w, h], axis=1).tolist()

            idxs = cv2.dnn.NMSBoxes(rects, scores.tolist(), self.confidence_threshold, self.nms_threshold)

            # idxs is a list of indices; handle different OpenCV return types
            keep_indices = []
            if isinstance(idxs, (list, tuple)):
                keep_indices = [int(i) for i in (idxs if np.ndim(idxs) == 1 else np.array(idxs).flatten())]
            elif hasattr(idxs, 'flatten'):
                keep_indices = [int(i) for i in idxs.flatten().tolist()]

            for i in keep_indices:
                final_dets.append(dets[i])

        return final_dets


class VisionManager:
    """Manages computer vision tools for situational awareness"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.object_detector = None
        
        # Initialize components based on config
        if config.get("features", {}).get("vision_object_detection", False):
            self._init_object_detection()
    
    def _init_object_detection(self):
        """Initialize object detection if enabled"""
        model_path = self.config.get("vision_model_path")
        if model_path:
            self.object_detector = ObjectDetector(model_path)
        else:
            self.logger.info("Object detection enabled but no model path specified")
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze a video frame and return insights"""
        results = {
            "timestamp": cv2.getTickCount() / cv2.getTickFrequency(),
            "objects": [],
            "scene_description": "",
            "context": {}
        }
        
        # Object detection
        if self.object_detector:
            objects = self.object_detector.detect_objects(frame)
            results["objects"] = objects
            
            # Generate scene description
            if objects:
                object_names = [obj["class_name"] for obj in objects]
                unique_objects = list(set(object_names))
                results["scene_description"] = f"I can see: {', '.join(unique_objects)}"
            else:
                results["scene_description"] = "No objects detected"
        
        # Add basic frame info
        h, w = frame.shape[:2]
        results["context"] = {
            "frame_size": (w, h),
            "brightness": float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))),
            "object_count": len(results["objects"])
        }
        
        return results
    
    def get_situational_context(self, frame: np.ndarray) -> str:
        """Get a brief situational context description"""
        if not self.config.get("features", {}).get("vision_object_detection", False):
            return ""
        
        analysis = self.analyze_frame(frame)
        objects = analysis["objects"]
        
        if not objects:
            return ""
        
        # Focus on relevant objects for Jarvis-like awareness
        relevant_objects = []
        for obj in objects:
            if obj["confidence"] > 0.7 and obj["class_name"] in [
                "person", "laptop", "tv", "cell phone", "book", "cup", "bottle"
            ]:
                relevant_objects.append(obj["class_name"])
        
        if relevant_objects:
            unique_objects = list(set(relevant_objects))
            return f"I notice: {', '.join(unique_objects)}"
        
        return ""


def create_vision_manager(config: Dict):
    """Factory function to create vision manager"""
    return VisionManager(config)


class MockVisionManager:
    """Mock vision manager when ONNX not available"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("Using mock vision manager (ONNX not available)")
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        return {
            "timestamp": cv2.getTickCount() / cv2.getTickFrequency(),
            "objects": [],
            "scene_description": "Vision analysis not available",
            "context": {"frame_size": frame.shape[:2][::-1], "object_count": 0}
        }
    
    def get_situational_context(self, frame: np.ndarray) -> str:
        return ""


def create_vision_manager_safe(config: Dict):
    """Create vision manager with fallback to mock"""
    if ONNX_AVAILABLE and config.get("features", {}).get("vision_object_detection", False):
        return VisionManager(config)
    else:
        return MockVisionManager(config)
