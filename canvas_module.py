import cv2
import numpy as np
import time
import torch
import torchvision
from utils import load_icon, overlay_image, is_point_in_rect
from PIL import Image
from torchvision import transforms
import os


class CanvasModule:
    def __init__(self):
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.prev_point = None
        self.drawing = False
        self.color = (0, 0, 255)  # Red
        self.mode = "DRAW"
        self.hover_start = None
        self.recognition_start = 0
        self.return_icon = load_icon("return_icon.png", (80, 80))
        self.brush_icon = load_icon("paint_icon.png", (120, 120))

        # Smoothing for drawing
        self.smooth_points = []
        self.smooth_factor = 0.7

        # Hand tracking persistence
        self.last_hand_time = time.time()
        self.hand_lost_threshold = 2.0  # Increased even more for maximum persistence
        self.drawing_lost_threshold = 0.5  # Slightly increased for drawing state
        self.last_valid_position = None  # Store last known good position
        self.position_stability_buffer = []  # Buffer for position stability

        # Sensitivity settings
        self.pinch_threshold = 50  # Increased slightly for more stable detection
        self.ui_hover_time = 0.6  # Further reduced for faster UI response
        self.return_hover_time = 0.8  # Further reduced
        self.mode_hover_time = 0.8  # Further reduced

        # Initialize hover tracking variables
        self.color_hover_start = None
        self.mode_hover_start = None

        # Load lightweight sketch recognition model
        self.model = None
        self.sketch_classifier = None
        self.recognition_cache = {}  # Cache for recognition results
        self.last_canvas_hash = None
        self.continuous_mode = False  # For optional real-time recognition

        try:
            # Use MobileNetV2 for much better efficiency
            self.model = torchvision.models.mobilenet_v2(pretrained=True)
            self.model.eval()

            # Simpler transform for sketches - keep 3 channels for MobileNet
            self.transform = transforms.Compose(
                [
                    transforms.Resize((128, 128)),  # Smaller size for efficiency
                    transforms.ToTensor(),
                    # Keep 3 channels but normalize for sketches
                    transforms.Lambda(
                        lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)
                    ),
                    # Light normalization optimized for sketches (not ImageNet)
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

            # Dynamic class system with confidence scores
            self.base_classes = {
                "circle": ["round", "ball", "sun", "wheel", "face"],
                "rectangle": ["square", "box", "house", "building", "window"],
                "triangle": ["arrow", "mountain", "roof", "tree"],
                "lines": ["stick", "fence", "hair", "grass"],
                "curves": ["wave", "snake", "river", "path"],
                "star": ["star", "flower", "explosion"],
                "animal": ["cat", "dog", "bird", "fish"],
                "vehicle": ["car", "truck", "plane", "boat"],
            }

            # Flatten for quick lookup
            self.all_classes = []
            self.class_to_category = {}
            for category, items in self.base_classes.items():
                self.all_classes.extend(items)
                for item in items:
                    self.class_to_category[item] = category

            print(f"Loaded efficient recognition with {len(self.all_classes)} classes")

        except Exception as e:
            print(f"Lightweight recognition model not available: {e}")
            # Fallback to simple geometric recognition
            self.use_geometric_fallback = True

    def stabilize_position(self, position):
        """Add position to stability buffer and return stabilized position"""
        if not position:
            return None

        # Add to buffer
        self.position_stability_buffer.append(position)

        # Keep buffer size manageable
        if len(self.position_stability_buffer) > 5:
            self.position_stability_buffer.pop(0)

        # Return averaged position for stability
        if len(self.position_stability_buffer) >= 2:
            avg_x = sum(p[0] for p in self.position_stability_buffer) / len(
                self.position_stability_buffer
            )
            avg_y = sum(p[1] for p in self.position_stability_buffer) / len(
                self.position_stability_buffer
            )
            return (int(avg_x), int(avg_y))
        else:
            return position

    def get_safe_text_position(
        self, x, y, img_width, img_height, offset_x=20, offset_y=-20
    ):
        """Get a safe position for text that avoids UI elements"""
        # Avoid top area (color palette and return button)
        if y < 120:
            y = 120

        # Avoid left side (mode buttons)
        if x < 200 and y < 220:
            x = 200

        # Avoid right edge
        if x > img_width - 150:
            x = img_width - 150

        # Avoid bottom edge
        if y > img_height - 60:
            y = img_height - 60

        return (x + offset_x, y + offset_y)

    def smooth_point(self, point):
        """Apply smoothing to reduce jitter in drawing"""
        if not self.smooth_points:
            self.smooth_points.append(point)
            return point

        # Keep last few points for smoothing - reduced for more responsiveness
        self.smooth_points.append(point)
        if len(self.smooth_points) > 2:  # Reduced from 3 for less lag
            self.smooth_points.pop(0)

        # Weighted average for better responsiveness (more weight to recent points)
        if len(self.smooth_points) == 1:
            return self.smooth_points[0]
        elif len(self.smooth_points) == 2:
            # 70% current, 30% previous
            curr = self.smooth_points[-1]
            prev = self.smooth_points[-2]
            avg_x = int(curr[0] * 0.7 + prev[0] * 0.3)
            avg_y = int(curr[1] * 0.7 + prev[1] * 0.3)
            return (avg_x, avg_y)
        else:
            # Weighted average with more emphasis on recent points
            weights = [0.5, 0.3, 0.2]  # Most recent gets highest weight
            avg_x = sum(p[0] * w for p, w in zip(reversed(self.smooth_points), weights))
            avg_y = sum(p[1] * w for p, w in zip(reversed(self.smooth_points), weights))
            return (int(avg_x), int(avg_y))

    def draw_ui(self, img):
        # Draw return button (keep in top-left)
        img = overlay_image(img, self.return_icon, 20, 20)

        # Draw horizontal color palette at top center
        colors = [
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]

        color_rects = []
        # Center the color palette horizontally
        palette_width = len(colors) * 55  # 50 + 5 spacing
        start_x = (img.shape[1] - palette_width) // 2

        for i, color in enumerate(colors):
            x = start_x + i * 55
            y = 20  # Top of screen
            cv2.rectangle(img, (x, y), (x + 50, y + 40), color, -1)
            cv2.rectangle(img, (x, y), (x + 50, y + 40), (200, 200, 200), 2)
            color_rects.append((x, y, x + 50, y + 40))

            # Highlight selected color
            if color == self.color:
                cv2.rectangle(img, (x - 3, y - 3), (x + 53, y + 43), (255, 255, 255), 3)

        # Draw mode buttons - moved higher up (removed RECOGNIZE button)
        modes = ["DRAW", "CLEAR"]
        mode_rects = []
        center_y = img.shape[0] // 2
        for i, mode in enumerate(modes):
            x = 50
            y = 120 + i * 50  # Moved much higher (was img.shape[0] - 150 + i * 50)
            cv2.rectangle(
                img,
                (x, y),
                (x + 120, y + 35),  # Made slightly smaller
                (50, 50, 50) if self.mode != mode else (0, 150, 0),
                -1,
            )
            cv2.putText(
                img,
                mode,
                (x + 10, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # Slightly smaller font
                (255, 255, 255),
                2,
            )
            mode_rects.append((x, y, x + 120, y + 35))

        return color_rects, mode_rects

    def detect_geometric_shapes(self, canvas):
        """Lightweight geometric shape detection using OpenCV - very efficient"""
        # Convert to grayscale
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return "empty", 0.0

        # Get the largest contour (main drawing)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < 500:  # Too small
            return "small_shape", 0.5

        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        vertices = len(approx)

        # Calculate metrics for classification
        perimeter = cv2.arcLength(largest_contour, True)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)

        # Solidity (area/hull_area)
        solidity = area / hull_area if hull_area > 0 else 0

        # Aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1

        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # Classification logic with confidence
        if circularity > 0.7:
            return "circle", min(0.9, circularity)
        elif vertices == 3:
            return "triangle", 0.8
        elif vertices == 4:
            if 0.8 <= aspect_ratio <= 1.2:
                return "square", 0.85
            else:
                return "rectangle", 0.8
        elif vertices > 6 and circularity > 0.5:
            return "star", 0.7
        elif solidity < 0.8:
            return "complex_shape", 0.6
        else:
            return "polygon", 0.5

    def get_canvas_hash(self, canvas):
        """Quick hash of canvas for caching"""
        # Simple hash based on non-zero pixels
        return hash(
            tuple(np.nonzero(canvas.sum(axis=2))[0][:100])
        )  # Sample first 100 points

    def recognize_object(self):
        """Dynamic recognition with caching and fallback methods"""
        if np.sum(self.canvas) == 0:  # Empty canvas
            return

        # Check cache first (most efficient)
        canvas_hash = self.get_canvas_hash(self.canvas)
        if canvas_hash == self.last_canvas_hash and hasattr(self, "last_result"):
            # Canvas hasn't changed significantly, use cached result
            class_name, confidence = self.last_result
            self.display_recognition_result(class_name, confidence, cached=True)
            return

        self.last_canvas_hash = canvas_hash

        # Try geometric detection first (fastest)
        geometric_result, geo_confidence = self.detect_geometric_shapes(self.canvas)

        # If geometric detection is confident enough, use it
        if geo_confidence > 0.7:
            self.last_result = (geometric_result, geo_confidence)
            self.display_recognition_result(geometric_result, geo_confidence)
            return

        # If model is available and geometric detection isn't confident
        if self.model is not None:
            try:
                # Convert canvas to PIL image
                canvas_rgb = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(canvas_rgb)

                # Preprocess (now much lighter)
                input_tensor = self.transform(pil_img)
                input_batch = input_tensor.unsqueeze(0)

                with torch.no_grad():
                    # Use only a subset of the model for efficiency
                    features = self.model.features(input_batch)
                    pooled = features.mean([2, 3])  # Global average pooling

                    # Simple classification based on feature patterns
                    feature_vec = pooled.flatten().numpy()

                    # Use feature matching for dynamic classification
                    best_class, ml_confidence = self.classify_from_features(feature_vec)

                    # Combine geometric and ML results
                    if ml_confidence > geo_confidence:
                        result = (best_class, ml_confidence)
                    else:
                        result = (geometric_result, geo_confidence)

            except Exception as e:
                print(f"ML recognition failed: {e}")
                result = (geometric_result, geo_confidence)
        else:
            # Use geometric result
            result = (geometric_result, geo_confidence)

        self.last_result = result
        self.display_recognition_result(result[0], result[1])

    def classify_from_features(self, features):
        """Lightweight feature-based classification"""
        # Simple heuristics based on feature patterns
        feature_sum = np.sum(features)
        feature_std = np.std(features)
        feature_max = np.max(features)

        # Basic pattern matching (can be expanded)
        if feature_std < 0.1 and feature_sum > 10:
            return "simple_shape", 0.7
        elif feature_max > 5:
            return "complex_drawing", 0.6
        else:
            return "sketch", 0.5

    def display_recognition_result(self, class_name, confidence, cached=False):
        """Display recognition result with confidence and status"""
        # Clear a larger area to prevent overlaps
        self.canvas[60:180, 60:450] = 0

        # Display main result with better positioning
        display_text = f"{class_name.replace('_', ' ').title()}"
        cv2.putText(
            self.canvas,
            display_text,
            (80, 100),  # Moved slightly left and up
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,  # Slightly smaller font
            (0, 255, 0) if confidence > 0.7 else (0, 255, 255),
            2,
        )

        # Display confidence with proper spacing
        conf_text = f"Confidence: {confidence:.1%}"
        cv2.putText(
            self.canvas,
            conf_text,
            (80, 130),  # Better vertical spacing
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Display status with proper spacing
        status = "CACHED" if cached else "NEW"
        cv2.putText(
            self.canvas,
            status,
            (80, 155),  # Moved below confidence instead of to the right
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (128, 128, 128),
            1,
        )

    def add_dynamic_class(self, new_class, category="custom"):
        """Add new class dynamically - very lightweight"""
        if category not in self.base_classes:
            self.base_classes[category] = []

        if new_class not in self.all_classes:
            self.base_classes[category].append(new_class)
            self.all_classes.append(new_class)
            self.class_to_category[new_class] = category

            # Clear cache when classes change
            self.recognition_cache.clear()
            self.last_canvas_hash = None

            print(f"Added '{new_class}' to category '{category}'")

    def continuous_recognition(self, enable=True):
        """Enable/disable continuous recognition while drawing"""
        self.continuous_mode = enable
        if enable:
            print("Continuous recognition enabled - will recognize shapes as you draw")
        else:
            print("Continuous recognition disabled - recognition on demand only")

    def run(self, img, lm_list):
        should_exit = False
        img_height, img_width, _ = img.shape
        current_time = time.time()

        # Draw UI and get interactive areas
        img = overlay_image(
            img, self.brush_icon, img.shape[1] // 2 - 60, 70
        )  # Moved down to avoid color palette
        color_rects, mode_rects = self.draw_ui(img)

        # Process hand gestures
        if lm_list:
            index_tip = lm_list[8]
            x, y = index_tip[1], index_tip[2]

            # Calculate pinch distance (distance between thumb tip and index tip)
            thumb_tip = lm_list[4]
            pinch_distance = np.linalg.norm(
                np.array([x, y]) - np.array([thumb_tip[1], thumb_tip[2]])
            )

            # Initialize UI tracking variables
            in_ui_area = False
            mode_selected = False

            # Draw cursor
            cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)

            # Check return button
            if is_point_in_rect(x, y, (20, 20, 100, 100)):
                if self.hover_start is None:
                    self.hover_start = time.time()
                elif time.time() - self.hover_start >= 1.5:
                    should_exit = True
            else:
                self.hover_start = None

            # Check color selection
            for i, rect in enumerate(color_rects):
                if is_point_in_rect(x, y, rect):
                    in_ui_area = True
                    if self.color_hover_start is None:
                        self.color_hover_start = time.time()
                    elif time.time() - self.color_hover_start >= 1.0:
                        self.color = [
                            (0, 0, 255),
                            (0, 255, 0),
                            (255, 0, 0),
                            (0, 255, 255),
                            (255, 0, 255),
                            (255, 255, 0),
                        ][i]
                        self.color_hover_start = None
                    break

            if not any(is_point_in_rect(x, y, rect) for rect in color_rects):
                self.color_hover_start = None

            # Check mode selection
            for i, rect in enumerate(mode_rects):
                if is_point_in_rect(x, y, rect):
                    in_ui_area = True
                    mode_selected = True
                    if self.mode_hover_start is None:
                        self.mode_hover_start = current_time
                    elif current_time - self.mode_hover_start >= self.mode_hover_time:
                        mode = ["DRAW", "CLEAR"][i]  # Removed RECOGNIZE from list
                        if mode == "CLEAR":
                            self.canvas = np.zeros_like(self.canvas)
                        else:
                            self.mode = mode
                        self.mode_hover_start = None
                    break

            if not mode_selected:
                self.mode_hover_start = None

            # Check brush icon area (prevent drawing over it)
            brush_icon_x = img.shape[1] // 2 - 60
            brush_icon_y = 70
            if is_point_in_rect(
                x,
                y,
                (brush_icon_x, brush_icon_y, brush_icon_x + 120, brush_icon_y + 120),
            ):
                in_ui_area = True

            # Draw cursor based on state and add visual feedback
            cursor_color = (0, 255, 0)  # Default green
            cursor_size = 10

            # Show hover progress for UI elements - updated for new timings
            if self.hover_start and current_time - self.hover_start > 0:
                progress = min(
                    1.0, (current_time - self.hover_start) / self.return_hover_time
                )
                cv2.circle(
                    img, (x, y), int(15 + progress * 10), (255, 255, 0), 2
                )  # Yellow ring

            if self.color_hover_start and current_time - self.color_hover_start > 0:
                progress = min(
                    1.0, (current_time - self.color_hover_start) / self.ui_hover_time
                )
                cv2.circle(
                    img, (x, y), int(15 + progress * 10), (0, 255, 255), 2
                )  # Cyan ring

            if self.mode_hover_start and current_time - self.mode_hover_start > 0:
                progress = min(
                    1.0, (current_time - self.mode_hover_start) / self.mode_hover_time
                )
                cv2.circle(
                    img, (x, y), int(15 + progress * 10), (255, 0, 255), 2
                )  # Magenta ring

            # Special cursor for drawing mode - improved sensitivity
            if self.mode == "DRAW" and not in_ui_area:
                if (
                    pinch_distance < self.pinch_threshold
                ):  # More sensitive pinch detection
                    cursor_color = (0, 0, 255)  # Red for drawing
                    cursor_size = 15
                    # Position DRAWING text to avoid overlaps using safe positioning
                    text_x, text_y = self.get_safe_text_position(
                        x, y, img_width, img_height, -40, -40
                    )
                    cv2.putText(
                        img,
                        "DRAWING",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cursor_color = (0, 255, 0)  # Green for ready
                    cursor_size = 10
                    # Removed pinch distance display to avoid FPS-like overlapping numbers

            # Draw the main cursor
            cv2.circle(img, (x, y), cursor_size, cursor_color, cv2.FILLED)

            # Drawing logic - only if not in UI area and in DRAW mode
            if self.mode == "DRAW" and not in_ui_area:
                # Convert coordinates to canvas space first
                canvas_x = int(x * self.canvas.shape[1] / img_width)
                canvas_y = int(y * self.canvas.shape[0] / img_height)

                # Ensure coordinates are within canvas bounds
                canvas_x = max(0, min(canvas_x, self.canvas.shape[1] - 1))
                canvas_y = max(0, min(canvas_y, self.canvas.shape[0] - 1))

                # Apply smoothing
                smooth_point = self.smooth_point((canvas_x, canvas_y))

                if (
                    pinch_distance < self.pinch_threshold
                ):  # More sensitive pinch detection
                    if not self.drawing:
                        self.drawing = True
                        self.prev_point = None

                    # Convert coordinates to canvas space
                    canvas_x = int(x * self.canvas.shape[1] / img_width)
                    canvas_y = int(y * self.canvas.shape[0] / img_height)

                    if self.prev_point:
                        cv2.line(
                            self.canvas,
                            self.prev_point,
                            (canvas_x, canvas_y),
                            self.color,
                            5,
                        )
                    self.prev_point = (canvas_x, canvas_y)
                else:
                    # Stop drawing - fingers not pinched
                    if self.drawing:
                        self.drawing = False
                        self.smooth_points = []  # Clear smoothing points

                        # Optional: Auto-recognize when drawing stops (lightweight)
                        if hasattr(self, "continuous_mode") and self.continuous_mode:
                            # Only run geometric detection for performance
                            shape, confidence = self.detect_geometric_shapes(
                                self.canvas
                            )
                            if confidence > 0.6:  # Lower threshold for continuous mode
                                self.display_recognition_result(shape, confidence)

            else:
                # Reset drawing state when not in draw mode or in UI area
                self.drawing = False
                self.prev_point = None
                self.smooth_points = []
        else:
            # No hand detected - use enhanced persistence logic
            current_time = time.time()
            time_since_last_hand = current_time - self.last_hand_time

            # Use last valid position for brief tracking losses (if available)
            if time_since_last_hand <= 0.5 and self.last_valid_position:
                # Show phantom cursor for very brief losses
                x, y = self.last_valid_position
                cv2.circle(img, (x, y), 8, (128, 128, 128), 2)  # Gray phantom cursor
                # Position tracking text safely to avoid overlaps
                text_x, text_y = self.get_safe_text_position(
                    x, y, img_width, img_height, 20, -20
                )
                cv2.putText(
                    img,
                    "TRACKING...",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (128, 128, 128),
                    1,
                )

            # Different thresholds for different states
            if self.drawing and time_since_last_hand > self.drawing_lost_threshold:
                # Stop drawing but keep other states longer
                self.drawing = False
                self.smooth_points = []
                self.position_stability_buffer = []

            if time_since_last_hand > self.hand_lost_threshold:
                # Hand has been lost for more than threshold - reset everything
                self.drawing = False
                self.prev_point = None
                self.smooth_points = []
                self.hover_start = None
                self.color_hover_start = None
                self.mode_hover_start = None
                self.last_valid_position = None
                self.position_stability_buffer = []
            # If within threshold, keep current state to handle brief tracking losses

        # Optimized canvas blending - only process if there's actual drawing
        if np.any(self.canvas):  # Only blend if canvas has content
            canvas_resized = cv2.resize(self.canvas, (img_width, img_height))

            # Simple additive blending for better performance
            img = cv2.addWeighted(img, 0.7, canvas_resized, 0.3, 0)
        # If no canvas content, skip blending entirely for better performance

        # Show drawing status - simplified to avoid overlaps
        status_lines = []

        # Main mode status only
        status_text = f"Mode: {self.mode}"
        if self.drawing:
            status_text += " | DRAWING"
        status_lines.append(status_text)

        # Simplified tracking status (no frequent updates)
        current_time = time.time()
        time_since_last_hand = current_time - self.last_hand_time

        if lm_list:
            tracking_text = "TRACKING: ACTIVE"
            # Removed pinch distance to avoid FPS-like number updates
        elif time_since_last_hand <= self.hand_lost_threshold:
            tracking_text = f"TRACKING: PERSISTENT ({self.hand_lost_threshold - time_since_last_hand:.1f}s)"
        else:
            tracking_text = "TRACKING: LOST"

        status_lines.append(tracking_text)

        # Display status lines with proper spacing to avoid overlaps
        y_start = img.shape[0] - 45  # Start higher to accommodate multiple lines
        for i, line in enumerate(status_lines):
            if line.strip():  # Only display non-empty lines
                cv2.putText(
                    img,
                    line,
                    (10, y_start + i * 20),  # 20 pixel spacing between lines
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        return img, should_exit
