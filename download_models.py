#!/usr/bin/env python3
"""
Enhanced Model Download Script for Jarvis AI Assistant

This script downloads and sets up required AI models for the Jarvis system
with improved error handling, progress tracking, and model verification.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COCO class names for object detection
COCO_CLASSES = [
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


def check_dependencies() -> bool:
    """Check if required dependencies are available"""
    logger.info("Checking dependencies...")

    try:
        import ultralytics
        logger.info(f"✅ ultralytics version: {ultralytics.__version__}")
    except ImportError:
        logger.error("❌ ultralytics not installed. Installing...")
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"],
                              capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to install ultralytics: {result.stderr}")
            return False
        logger.info("✅ ultralytics installed successfully")

    return True


def setup_models_directory() -> Path:
    """Create models directory structure"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Create subdirectories for different model types
    (models_dir / "object_detection").mkdir(exist_ok=True)
    (models_dir / "sketch_recognition").mkdir(exist_ok=True)
    (models_dir / "cache").mkdir(exist_ok=True)

    logger.info(f"Models directory: {models_dir.absolute()}")
    return models_dir


def save_model_metadata(models_dir: Path, model_name: str, metadata: Dict):
    """Save model metadata for tracking"""
    metadata_file = models_dir / f"{model_name}_metadata.json"
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model metadata to {metadata_file}")
    except Exception as e:
        logger.warning(f"Failed to save metadata: {e}")


def download_yolov8_model(model_variant: str = "yolov8n") -> bool:
    """Download and setup YOLOv8 model with enhanced error handling"""
    from ultralytics import YOLO

    models_dir = setup_models_directory()
    onnx_path = models_dir / f"{model_variant}.onnx"

    # Check if ONNX model already exists
    if onnx_path.exists():
        file_size = os.path.getsize(onnx_path)
        if file_size > 1000000:  # At least 1MB
            logger.info(f"✅ {model_variant} ONNX model already exists: {onnx_path}")
            logger.info(f"File size: {file_size / 1024 / 1024:.1f} MB")
            return True
        else:
            logger.warning(f"Existing model file seems corrupted (size: {file_size} bytes), re-downloading...")
            os.remove(onnx_path)

    logger.info(f"Setting up {model_variant} model...")

    try:
        # Download PyTorch model first
        logger.info(f"Downloading {model_variant}.pt model...")
        model = YOLO(f"{model_variant}.pt")  # This will download if not present

        # Export to ONNX with optimized settings
        logger.info("Converting to ONNX format...")
        export_path = model.export(
            format="onnx",
            opset=12,
            dynamic=False,
            imgsz=640,
            simplify=True,
            verbose=False
        )

        # Move the exported file to models directory
        exported = Path(export_path) if isinstance(export_path, str) else Path(f"{model_variant}.onnx")

        if exported.exists():
            if exported != onnx_path:
                exported.rename(onnx_path)

            # Verify the exported model
            file_size = os.path.getsize(onnx_path)
            logger.info(f"✅ Successfully exported ONNX model to: {onnx_path}")
            logger.info(f"File size: {file_size / 1024 / 1024:.1f} MB")

            # Clean up PyTorch model to save space
            pt_files = [f"{model_variant}.pt", f"yolov8n.pt"]
            for pt_file in pt_files:
                if os.path.exists(pt_file):
                    try:
                        os.remove(pt_file)
                        logger.info(f"Cleaned up {pt_file}")
                    except:
                        pass

            return True
        else:
            logger.error("❌ Exported ONNX file not found")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to download/export {model_variant}: {e}")
        return False


def save_class_names(models_dir: Path):
    """Save COCO class names for easy access"""
    classes_file = models_dir / "coco_classes.json"
    try:
        with open(classes_file, 'w', encoding='utf-8') as f:
            json.dump(COCO_CLASSES, f, indent=2)
        logger.info(f"Saved class names to {classes_file}")
    except Exception as e:
        logger.warning(f"Failed to save class names: {e}")


def create_model_config(models_dir: Path):
    """Create a configuration file for the vision system"""
    config = {
        "object_detection": {
            "enabled": True,
            "model_path": "models/yolov8n.onnx",
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "input_size": [640, 640],
            "classes": len(COCO_CLASSES),
            "class_names": COCO_CLASSES
        },
        "sketch_recognition": {
            "enabled": False,
            "model_path": "models/sketch_classifier.onnx",
            "confidence_threshold": 0.7
        }
    }

    config_file = models_dir / "vision_config.json"
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created vision configuration: {config_file}")
    except Exception as e:
        logger.warning(f"Failed to create vision config: {e}")


def main():
    """Main function to download all required models"""
    logger.info("=== Jarvis AI Model Setup ===")

    # Check dependencies first
    if not check_dependencies():
        return 1

    models_dir = setup_models_directory()
    success = True

    # Download YOLOv8n model (default)
    logger.info("Setting up YOLOv8n object detection model...")
    if download_yolov8_model("yolov8n"):
        # Save model metadata
        metadata = {
            "model_name": "yolov8n",
            "model_type": "object_detection",
            "framework": "onnx",
            "input_size": [640, 640],
            "classes": len(COCO_CLASSES),
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": "models/yolov8n.onnx"
        }
        save_model_metadata(models_dir, "yolov8n", metadata)

        # Save class names and create config
        save_class_names(models_dir)
        create_model_config(models_dir)

    else:
        logger.error("❌ Failed to setup YOLOv8n model")
        success = False

    if success:
        logger.info("\n✅ Model setup completed successfully!")
        logger.info("\nTo enable object detection, add these to your .env file:")
        logger.info("ENABLE_OBJECT_DETECTION=true")
        logger.info("VISION_MODEL_PATH=models/yolov8n.onnx")
        logger.info("VISION_CONFIDENCE_THRESHOLD=0.5")

        logger.info(f"\nAvailable object classes: {len(COCO_CLASSES)}")
        logger.info("Including: person, car, bicycle, dog, cat, chair, laptop, etc.")

        # Test the model
        test_model = input("\nTest the downloaded model? [y/N]: ").lower().strip()
        if test_model == 'y':
            test_object_detection(models_dir / "yolov8n.onnx")

    else:
        logger.error("❌ Model setup failed")
        logger.info("You can try running this script again or check the error messages above")
        return 1

    return 0


def test_object_detection(model_path: Path):
    """Test the object detection model"""
    try:
        import onnxruntime as ort
        import numpy as np

        logger.info(f"Testing model: {model_path}")

        # Load the model
        session = ort.InferenceSession(str(model_path))

        # Get input details
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        logger.info(f"✅ Model loaded successfully")
        logger.info(f"Input name: {input_name}")
        logger.info(f"Input shape: {input_shape}")

        # Create dummy input
        dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

        # Run inference
        outputs = session.run(None, {input_name: dummy_input})

        logger.info(f"✅ Model inference test passed")
        logger.info(f"Output shapes: {[output.shape for output in outputs]}")

    except ImportError:
        logger.warning("onnxruntime not installed, skipping model test")
        logger.info("Install with: pip install onnxruntime")
    except Exception as e:
        logger.error(f"Model test failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
