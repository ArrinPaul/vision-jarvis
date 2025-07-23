import cv2
import numpy as np
import time
import torch
import torchvision
from utils import load_icon, overlay_image, is_point_in_rect
from PIL import Image
from torchvision import transforms


class CanvasModule:
    def __init__(self):
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.prev_point = None
        self.drawing = False
        self.color = (0, 0, 255)  # Red
        self.mode = "DRAW"
        self.hover_start = None
        self.color_hover_start = None  # Separate timer for color selection
        self.mode_hover_start = None  # Separate timer for mode selection
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

        # Load object recognition model
        self.model = None
        try:
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.eval()
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.classes = [
                "car",
                "flower",
                "sun",
                "tree",
                "house",
                "cat",
                "dog",
                "bird",
            ]
        except:
            print("Object recognition model not available")

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
        # Draw return button
        img = overlay_image(img, self.return_icon, 20, 20)

        # Draw color palette
        colors = [
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]

        color_rects = []
        for i, color in enumerate(colors):
            x = img.shape[1] - 70
            y = 100 + i * 70
            cv2.rectangle(img, (x, y), (x + 50, y + 50), color, -1)
            cv2.rectangle(img, (x, y), (x + 50, y + 50), (200, 200, 200), 2)
            color_rects.append((x, y, x + 50, y + 50))

            # Highlight selected color
            if color == self.color:
                cv2.rectangle(img, (x - 5, y - 5), (x + 55, y + 55), (255, 255, 255), 2)

        # Draw mode buttons
        modes = ["DRAW", "CLEAR", "RECOGNIZE"]
        mode_rects = []
        for i, mode in enumerate(modes):
            x = 50
            y = img.shape[0] - 150 + i * 50
            cv2.rectangle(
                img,
                (x, y),
                (x + 150, y + 40),
                (50, 50, 50) if self.mode != mode else (0, 150, 0),
                -1,
            )
            cv2.putText(
                img,
                mode,
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            mode_rects.append((x, y, x + 150, y + 40))

        return color_rects, mode_rects

    def recognize_object(self):
        if self.model is None:
            return

        # Convert canvas to PIL image
        canvas_rgb = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(canvas_rgb)

        # Preprocess and predict
        input_tensor = self.transform(pil_img)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_batch)

        # Get prediction
        _, predicted_idx = torch.max(output, 1)
        class_idx = predicted_idx.item() % len(self.classes)
        class_name = self.classes[class_idx]

        # Display result on canvas
        cv2.putText(
            self.canvas,
            class_name,
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            3,
        )

    def run(self, img, lm_list):
        should_exit = False
        img_height, img_width, _ = img.shape

        # Draw UI and get interactive areas
        img = overlay_image(img, self.brush_icon, img.shape[1] // 2 - 60, 50)
        color_rects, mode_rects = self.draw_ui(img)

        # Process hand gestures
        if lm_list:
            self.last_hand_time = time.time()  # Update last seen time
            index_tip = lm_list[8]
            thumb_tip = lm_list[4]

            # Stabilize position to reduce jitter
            raw_position = (index_tip[1], index_tip[2])
            stable_position = self.stabilize_position(raw_position)

            if stable_position:
                x, y = stable_position
                self.last_valid_position = stable_position  # Store for persistence
            else:
                # Fallback to raw position if stabilization fails
                x, y = raw_position
                self.last_valid_position = raw_position

            # Calculate pinch distance once for the entire frame
            pinch_distance = (
                (thumb_tip[1] - index_tip[1]) ** 2 + (thumb_tip[2] - index_tip[2]) ** 2
            ) ** 0.5

            # Check if finger is in UI areas (to prevent drawing on UI)
            in_ui_area = False
            current_time = time.time()

            # Check return button - improved sensitivity
            if is_point_in_rect(x, y, (20, 20, 100, 100)):
                in_ui_area = True
                if self.hover_start is None:
                    self.hover_start = current_time
                elif current_time - self.hover_start >= self.return_hover_time:
                    should_exit = True
            else:
                self.hover_start = None

            # Check color selection - improved sensitivity
            color_selected = False
            for i, rect in enumerate(color_rects):
                if is_point_in_rect(x, y, rect):
                    in_ui_area = True
                    color_selected = True
                    if self.color_hover_start is None:
                        self.color_hover_start = current_time
                    elif current_time - self.color_hover_start >= self.ui_hover_time:
                        # Change color
                        colors = [
                            (0, 0, 255),  # Red
                            (0, 255, 0),  # Green
                            (255, 0, 0),  # Blue
                            (0, 255, 255),  # Yellow
                            (255, 0, 255),  # Magenta
                            (255, 255, 0),  # Cyan
                        ]
                        self.color = colors[i]
                        self.color_hover_start = None
                    break

            if not color_selected:
                self.color_hover_start = None

            # Check mode selection - improved sensitivity
            mode_selected = False
            for i, rect in enumerate(mode_rects):
                if is_point_in_rect(x, y, rect):
                    in_ui_area = True
                    mode_selected = True
                    if self.mode_hover_start is None:
                        self.mode_hover_start = current_time
                    elif current_time - self.mode_hover_start >= self.mode_hover_time:
                        mode = ["DRAW", "CLEAR", "RECOGNIZE"][i]
                        if mode == "CLEAR":
                            self.canvas = np.zeros_like(self.canvas)
                        elif mode == "RECOGNIZE":
                            self.recognize_object()
                        else:
                            self.mode = mode
                        self.mode_hover_start = None
                    break

            if not mode_selected:
                self.mode_hover_start = None

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
                    cv2.putText(
                        img,
                        "DRAWING",
                        (x - 30, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cursor_color = (0, 255, 0)  # Green for ready
                    cursor_size = 10
                    # Show pinch distance for debugging
                    cv2.putText(
                        img,
                        f"Dist: {int(pinch_distance)}",
                        (x + 20, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )

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
                        self.prev_point = smooth_point
                    else:
                        # Draw line from previous point to current point
                        if self.prev_point:
                            # Adaptive line thickness based on drawing speed
                            distance_moved = (
                                (smooth_point[0] - self.prev_point[0]) ** 2
                                + (smooth_point[1] - self.prev_point[1]) ** 2
                            ) ** 0.5
                            # Thinner lines for fast movement, thicker for slow/steady drawing
                            line_thickness = max(
                                4, min(12, int(15 - distance_moved / 3))
                            )

                            cv2.line(
                                self.canvas,
                                self.prev_point,
                                smooth_point,
                                self.color,
                                line_thickness,
                            )
                        self.prev_point = smooth_point

                    # Show preview line on the camera feed for better feedback
                    if self.prev_point and self.drawing:
                        # Convert canvas coordinates back to image coordinates for preview
                        prev_img_x = int(
                            self.prev_point[0] * img_width / self.canvas.shape[1]
                        )
                        prev_img_y = int(
                            self.prev_point[1] * img_height / self.canvas.shape[0]
                        )
                        cv2.line(img, (prev_img_x, prev_img_y), (x, y), self.color, 3)
                else:
                    # Stop drawing - fingers not pinched
                    if self.drawing:
                        self.drawing = False
                        self.smooth_points = []  # Clear smoothing points

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
                cv2.putText(
                    img,
                    "TRACKING...",
                    (x + 15, y - 15),
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

        # Show drawing status with more details and tracking info
        status_text = f"Mode: {self.mode}"
        if self.drawing:
            status_text += " | DRAWING"

        # Add tracking status
        current_time = time.time()
        time_since_last_hand = current_time - self.last_hand_time
        if lm_list:
            status_text += " | TRACKING: ACTIVE"
            # Show pinch distance for fine-tuning
            thumb_tip = lm_list[4]
            index_tip = lm_list[8]
            current_distance = (
                (thumb_tip[1] - index_tip[1]) ** 2 + (thumb_tip[2] - index_tip[2]) ** 2
            ) ** 0.5
            status_text += f" | Pinch: {int(current_distance)}"
        elif time_since_last_hand <= self.hand_lost_threshold:
            status_text += f" | TRACKING: PERSISTENT ({self.hand_lost_threshold - time_since_last_hand:.1f}s)"
        else:
            status_text += " | TRACKING: LOST"

        cv2.putText(
            img,
            status_text,
            (10, img.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # Smaller font to fit more info
            (255, 255, 255),
            2,
        )

        return img, should_exit
