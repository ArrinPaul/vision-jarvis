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
        self.recognition_start = 0
        self.return_icon = load_icon("return_icon.png", (80, 80))
        self.brush_icon = load_icon("paint_icon.png", (120, 120))
        
        # Load object recognition model
        self.model = None
        try:
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            self.classes = ['car', 'flower', 'sun', 'tree', 'house', 'cat', 'dog', 'bird']
        except:
            print("Object recognition model not available")
        
    def draw_ui(self, img):
        # Draw return button
        img = overlay_image(img, self.return_icon, 20, 20)
        
        # Draw color palette
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0)   # Cyan
        ]
        
        color_rects = []
        for i, color in enumerate(colors):
            x = img.shape[1] - 70
            y = 100 + i * 70
            cv2.rectangle(img, (x, y), (x+50, y+50), color, -1)
            cv2.rectangle(img, (x, y), (x+50, y+50), (200, 200, 200), 2)
            color_rects.append((x, y, x+50, y+50))
            
            # Highlight selected color
            if color == self.color:
                cv2.rectangle(img, (x-5, y-5), (x+55, y+55), (255, 255, 255), 2)
        
        # Draw mode buttons
        modes = ["DRAW", "CLEAR", "RECOGNIZE"]
        mode_rects = []
        for i, mode in enumerate(modes):
            x = 50
            y = img.shape[0] - 150 + i * 50
            cv2.rectangle(img, (x, y), (x+150, y+40), 
                         (50, 50, 50) if self.mode != mode else (0, 150, 0), -1)
            cv2.putText(img, mode, (x+10, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            mode_rects.append((x, y, x+150, y+40))
        
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
        cv2.putText(self.canvas, class_name, (100, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    def run(self, img, lm_list):
        should_exit = False
        img_height, img_width, _ = img.shape
        
        # Draw UI and get interactive areas
        img = overlay_image(img, self.brush_icon, img.shape[1]//2 - 60, 50)
        color_rects, mode_rects = self.draw_ui(img)
        
        # Process hand gestures
        if lm_list:
            index_tip = lm_list[8]
            x, y = index_tip[1], index_tip[2]
            
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
                    if self.hover_start is None:
                        self.hover_start = time.time()
                    elif time.time() - self.hover_start >= 1.0:
                        self.color = [
                            (0, 0, 255),
                            (0, 255, 0),
                            (255, 0, 0),
                            (0, 255, 255),
                            (255, 0, 255),
                            (255, 255, 0)
                        ][i]
                        self.hover_start = None
                    break
            
            # Check mode selection
            for i, rect in enumerate(mode_rects):
                if is_point_in_rect(x, y, rect):
                    if self.hover_start is None:
                        self.hover_start = time.time()
                    elif time.time() - self.hover_start >= 1.5:
                        mode = ["DRAW", "CLEAR", "RECOGNIZE"][i]
                        if mode == "CLEAR":
                            self.canvas = np.zeros_like(self.canvas)
                        elif mode == "RECOGNIZE":
                            self.recognize_object()
                        else:
                            self.mode = mode
                        self.hover_start = None
                    break
            
            # Drawing logic
            if self.mode == "DRAW":
                thumb_tip = lm_list[4]
                index_tip = lm_list[8]
                distance = ((thumb_tip[1]-index_tip[1])**2 + 
                            (thumb_tip[2]-index_tip[2])**2)**0.5
                
                if distance < 30:  # Pinch to draw
                    if not self.drawing:
                        self.drawing = True
                        self.prev_point = None
                    
                    # Convert coordinates to canvas space
                    canvas_x = int(x * self.canvas.shape[1] / img_width)
                    canvas_y = int(y * self.canvas.shape[0] / img_height)
                    
                    if self.prev_point:
                        cv2.line(self.canvas, self.prev_point, (canvas_x, canvas_y), 
                                self.color, 5)
                    self.prev_point = (canvas_x, canvas_y)
                else:
                    self.drawing = False
                    self.prev_point = None
        
        # Blend canvas with camera feed
        canvas_resized = cv2.resize(self.canvas, (img_width, img_height))
        img = cv2.addWeighted(img, 0.7, canvas_resized, 0.3, 0)
        
        return img, should_exit