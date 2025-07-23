import cv2
import numpy as np
import os
import urllib.request

def download_assets():
    assets = {
        "mic_icon.png": "https://github.com/microsoft/fluentui-emoji/raw/main/assets/Microphone/3D/microphone_3d.png",
        "camera_icon.png": "https://github.com/microsoft/fluentui-emoji/raw/main/assets/Camera/3D/camera_3d.png",
        "paint_icon.png": "https://github.com/microsoft/fluentui-emoji/raw/main/assets/Paintbrush/3D/paintbrush_3d.png",
        "return_icon.png": "https://github.com/microsoft/fluentui-emoji/raw/main/assets/Back%20arrow/3D/back_arrow_3d.png"
    }
    
    if not os.path.exists("assets"):
        os.makedirs("assets")
    
    for filename, url in assets.items():
        if not os.path.exists(f"assets/{filename}"):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, f"assets/{filename}")

def load_icon(name, size=(120, 120)):
    download_assets()
    img = cv2.imread(f"assets/{name}", cv2.IMREAD_UNCHANGED)
    if img is None:
        return create_fallback_icon(name, size)
    
    # Resize and convert to BGRA
    img = cv2.resize(img, size)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

def create_fallback_icon(name, size):
    icon = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    icon[:, :, 3] = 255  # Alpha channel
    
    if "mic" in name:
        cv2.circle(icon, (size[0]//2, size[1]//2), 40, (255, 255, 255, 255), 2)
        cv2.line(icon, (size[0]//2, 20), (size[0]//2, 40), (255, 255, 255, 255), 2)
    elif "camera" in name:
        cv2.rectangle(icon, (20, 20), (size[0]-20, size[1]-20), (255, 255, 255, 255), 2)
        cv2.circle(icon, (size[0]-30, 30), 10, (255, 255, 255, 255), -1)
    elif "paint" in name:
        cv2.line(icon, (30, size[1]-30), (size[0]-30, 30), (255, 255, 255, 255), 5)
    elif "return" in name:
        cv2.arrowedLine(icon, (size[0]-20, size[1]//2), (20, size[1]//2), (255, 255, 255, 255), 5)
    
    return icon

def overlay_image(background, overlay, x, y):
    if overlay.shape[2] == 4:
        # Extract alpha channel and create mask
        alpha = overlay[:, :, 3] / 255.0
        overlay_bgr = overlay[:, :, :3]
        
        # Region of interest
        h, w = overlay_bgr.shape[:2]
        roi = background[y:y+h, x:x+w]
        
        # Blend images
        for c in range(0, 3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay_bgr[:, :, c]
    else:
        background[y:y+overlay.shape[0], x:x+overlay.shape[1]] = overlay
    return background

def is_point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2