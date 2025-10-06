# Models Directory

Place your ONNX models here.

## YOLOv8n (Object Detection)

- Recommended file: `yolov8n.onnx`
- Source: Download from Ultralytics (export YOLOv8n to ONNX)
  - GitHub: https://github.com/ultralytics/ultralytics
  - Docs: https://docs.ultralytics.com/modes/export/
- After download, set this path in `voice_config.json`:

```json
{
  "features": { "vision_object_detection": true },
  "vision_model_path": "models/yolov8n.onnx"
}
```

Notes:
- ONNXRuntime must be installed (already included in requirements)
- The example preprocessing in vision_tools.py assumes YOLO-like inputs (640x640)
- You may need to adjust threshold or class mapping for different ONNX models
