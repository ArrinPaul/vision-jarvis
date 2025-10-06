"""
JARVIS Augmented Reality Interface System
========================================

This module provides comprehensive AR capabilities featuring:
- Real-time augmented reality overlays
- Spatial computing and 3D object tracking
- Hand tracking in 3D space with AR visualization
- Virtual object manipulation and interaction
- AR data visualization and HUD elements
- Iron Man-style AR interface
- Computer vision-based AR tracking
- Gesture-controlled AR interactions
"""

import cv2
import numpy as np
import mediapipe as mp
import pygame
from pygame.locals import *
import OpenGL.GL as gl
import OpenGL.GLU as glu
from OpenGL.arrays import vbo
import glfw
import math
import time
import threading
import json
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import asyncio
from datetime import datetime

# 3D math and transformations
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

class ARTrackingMode(Enum):
    """AR tracking modes"""
    MARKER_BASED = "marker_based"           # Using AR markers/QR codes
    MARKERLESS = "markerless"              # SLAM-based tracking
    HAND_TRACKING = "hand_tracking"        # Hand-based reference frame
    PLANE_DETECTION = "plane_detection"     # Planar surface tracking
    FACE_TRACKING = "face_tracking"        # Face-based AR
    HYBRID = "hybrid"                      # Multiple tracking methods

class ARObjectType(Enum):
    """Types of AR objects"""
    HOLOGRAM = "hologram"                  # Holographic display
    UI_PANEL = "ui_panel"                  # Interactive UI element
    DATA_VISUALIZATION = "data_viz"        # Charts and graphs
    VIRTUAL_BUTTON = "virtual_button"      # Clickable button
    PARTICLE_EFFECT = "particle_effect"   # Visual effects
    TEXT_LABEL = "text_label"             # Text information
    MODEL_3D = "model_3d"                 # 3D model
    PORTAL = "portal"                     # AR portal/window
    MENU = "menu"                         # Context menu
    WIDGET = "widget"                     # Interactive widget

@dataclass
class Transform3D:
    """3D transformation matrix"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix"""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation * self.scale
        matrix[:3, 3] = self.position
        return matrix
    
    def from_matrix(self, matrix: np.ndarray):
        """Set from 4x4 transformation matrix"""
        self.position = matrix[:3, 3]
        # Extract scale
        self.scale = np.array([
            np.linalg.norm(matrix[:3, 0]),
            np.linalg.norm(matrix[:3, 1]),
            np.linalg.norm(matrix[:3, 2])
        ])
        # Extract rotation (remove scale)
        self.rotation = matrix[:3, :3] / self.scale

@dataclass
class ARObject:
    """Augmented reality object"""
    id: str
    name: str
    object_type: ARObjectType
    transform: Transform3D = field(default_factory=Transform3D)
    visible: bool = True
    interactive: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Visual properties
    color: Tuple[float, float, float, float] = (0.0, 0.8, 1.0, 0.8)
    size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    material_properties: Dict[str, float] = field(default_factory=dict)
    
    # Interaction
    on_click: Optional[Callable] = None
    on_hover: Optional[Callable] = None
    on_gesture: Optional[Callable] = None
    
    # Animation
    animation_state: Dict[str, Any] = field(default_factory=dict)
    last_update: float = 0.0

class ARMarkerDetector:
    """Detects AR markers for tracking"""
    
    def __init__(self):
        # ArUco marker detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_size = 0.05  # 5cm markers
        
        # Camera calibration parameters (would be calibrated for specific camera)
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4, 1))
        
        # Tracking state
        self.detected_markers = {}
        self.tracking_history = deque(maxlen=30)
    
    def detect_markers(self, frame: np.ndarray) -> Dict[int, Transform3D]:
        """Detect AR markers and estimate their poses"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        detected_poses = {}
        
        if ids is not None:
            # Estimate pose for each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            
            for i, marker_id in enumerate(ids.flatten()):
                # Convert to Transform3D
                transform = Transform3D()
                transform.position = tvecs[i].flatten()
                
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
                transform.rotation = rotation_matrix
                
                detected_poses[marker_id] = transform
        
        self.detected_markers = detected_poses
        self.tracking_history.append(detected_poses.copy())
        
        return detected_poses
    
    def draw_marker_axes(self, frame: np.ndarray, transform: Transform3D, length: float = 0.03):
        """Draw coordinate axes on detected marker"""
        # Project 3D axes to 2D
        axes_3d = np.array([
            [0, 0, 0],           # Origin
            [length, 0, 0],      # X axis (red)
            [0, length, 0],      # Y axis (green)
            [0, 0, -length]      # Z axis (blue)
        ], dtype=np.float32)
        
        # Convert rotation matrix to rotation vector
        rvec, _ = cv2.Rodrigues(transform.rotation)
        tvec = transform.position.reshape(3, 1)
        
        # Project to image plane
        axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        axes_2d = axes_2d.reshape(-1, 2).astype(int)
        
        # Draw axes
        origin = tuple(axes_2d[0])
        frame = cv2.line(frame, origin, tuple(axes_2d[1]), (0, 0, 255), 3)    # X - Red
        frame = cv2.line(frame, origin, tuple(axes_2d[2]), (0, 255, 0), 3)    # Y - Green
        frame = cv2.line(frame, origin, tuple(axes_2d[3]), (255, 0, 0), 3)    # Z - Blue
        
        return frame

class PlanarSLAM:
    """Simple planar SLAM for markerless tracking"""
    
    def __init__(self):
        # Feature detection
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Reference frame
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.reference_points_3d = None
        
        # Current tracking state
        self.current_pose = Transform3D()
        self.is_tracking = False
        self.min_features = 50
        
        # Camera parameters (simplified)
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))
    
    def initialize_tracking(self, frame: np.ndarray) -> bool:
        """Initialize tracking with reference frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if len(keypoints) < self.min_features:
            return False
        
        # Store reference frame
        self.reference_frame = gray.copy()
        self.reference_keypoints = keypoints
        self.reference_descriptors = descriptors
        
        # Assume planar scene at z=0 for initialization
        self.reference_points_3d = np.array([
            [kp.pt[0] * 0.001, kp.pt[1] * 0.001, 0.0] for kp in keypoints
        ], dtype=np.float32)
        
        self.is_tracking = True
        print("✅ SLAM tracking initialized")
        return True
    
    def track_frame(self, frame: np.ndarray) -> Optional[Transform3D]:
        """Track current frame pose"""
        if not self.is_tracking or self.reference_descriptors is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < self.min_features:
            return None
        
        # Match features
        matches = self.matcher.match(self.reference_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < self.min_features:
            return None
        
        # Extract matched points
        ref_points = np.array([self.reference_keypoints[m.queryIdx].pt for m in matches])
        cur_points = np.array([keypoints[m.trainIdx].pt for m in matches])
        ref_points_3d = np.array([self.reference_points_3d[m.queryIdx] for m in matches])
        
        # Solve PnP for camera pose
        success, rvec, tvec = cv2.solvePnP(
            ref_points_3d, cur_points, self.camera_matrix, self.dist_coeffs
        )
        
        if success:
            # Update current pose
            self.current_pose.position = tvec.flatten()
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            self.current_pose.rotation = rotation_matrix
            
            return self.current_pose
        
        return None

class HandTrackingAR:
    """Hand-based AR tracking system"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Hand coordinate system
        self.hand_transforms = {}
        self.gesture_ar_objects = {}
        
    def process_hands(self, frame: np.ndarray) -> Dict[str, Transform3D]:
        """Process hand tracking for AR"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_transforms = {}
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                
                # Create coordinate system from hand
                transform = self.create_hand_coordinate_system(hand_landmarks, frame.shape)
                hand_transforms[hand_label] = transform
        
        self.hand_transforms = hand_transforms
        return hand_transforms
    
    def create_hand_coordinate_system(self, landmarks, frame_shape) -> Transform3D:
        """Create 3D coordinate system from hand landmarks"""
        h, w = frame_shape[:2]
        
        # Key points for coordinate system
        wrist = np.array([landmarks.landmark[0].x * w, landmarks.landmark[0].y * h, landmarks.landmark[0].z * w])
        middle_mcp = np.array([landmarks.landmark[9].x * w, landmarks.landmark[9].y * h, landmarks.landmark[9].z * w])
        index_mcp = np.array([landmarks.landmark[5].x * w, landmarks.landmark[5].y * h, landmarks.landmark[5].z * w])
        
        # Create coordinate system
        transform = Transform3D()
        transform.position = wrist
        
        # Y-axis: wrist to middle finger
        y_axis = middle_mcp - wrist
        y_axis = y_axis / np.linalg.norm(y_axis) if np.linalg.norm(y_axis) > 0 else np.array([0, 1, 0])
        
        # X-axis: perpendicular to Y in palm plane
        palm_vector = index_mcp - wrist
        x_axis = np.cross(y_axis, palm_vector)
        x_axis = x_axis / np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 0 else np.array([1, 0, 0])
        
        # Z-axis: complete the right-handed system
        z_axis = np.cross(x_axis, y_axis)
        
        # Build rotation matrix
        transform.rotation = np.column_stack([x_axis, y_axis, z_axis])
        
        return transform

class ARRenderer:
    """OpenGL-based AR rendering system"""
    
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.initialized = False
        
        # OpenGL state
        self.shader_program = None
        self.projection_matrix = np.eye(4)
        self.view_matrix = np.eye(4)
        
        # Rendering resources
        self.vao = None
        self.vbo = None
        self.texture_cache = {}
        
        # AR objects
        self.ar_objects = {}
        self.render_queue = []
        
        self.init_opengl()
    
    def init_opengl(self):
        """Initialize OpenGL for AR rendering"""
        try:
            # Initialize pygame with OpenGL
            pygame.init()
            pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
            
            # Enable depth testing and blending
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            
            # Set up projection matrix
            self.setup_camera_projection()
            
            # Create shaders
            self.create_ar_shaders()
            
            self.initialized = True
            print("✅ AR OpenGL renderer initialized")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize AR renderer: {e}")
            self.initialized = False
    
    def setup_camera_projection(self):
        """Setup camera projection matrix"""
        fov = 60.0  # Field of view in degrees
        aspect = self.width / self.height
        near = 0.01
        far = 100.0
        
        # Create perspective projection matrix
        self.projection_matrix = self.create_perspective_matrix(fov, aspect, near, far)
    
    def create_perspective_matrix(self, fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create perspective projection matrix"""
        f = 1.0 / math.tan(math.radians(fov) / 2.0)
        
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
    
    def create_ar_shaders(self):
        """Create AR-specific shaders"""
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec2 aTexCoord;
        layout (location = 3) in vec4 aColor;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform float time;
        uniform float alpha;
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;
        out vec4 Color;
        out float Alpha;
        
        void main()
        {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = mat3(transpose(inverse(model))) * aNormal;
            TexCoord = aTexCoord;
            Color = aColor;
            Alpha = alpha;
            
            // Add holographic distortion
            vec3 pos = aPos;
            pos.y += sin(time * 3.0 + pos.x * 2.0) * 0.01;
            
            gl_Position = projection * view * model * vec4(pos, 1.0);
        }
        """
        
        fragment_shader_source = """
        #version 330 core
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;
        in vec4 Color;
        in float Alpha;
        
        uniform float time;
        uniform int objectType;
        uniform sampler2D texture1;
        
        out vec4 FragColor;
        
        void main()
        {
            vec3 baseColor = Color.rgb;
            float alpha = Color.a * Alpha;
            
            // Add holographic effects based on object type
            if (objectType == 0) { // HOLOGRAM
                // Scanline effect
                float scanline = sin(gl_FragCoord.y * 0.5 + time * 10.0) * 0.1 + 0.9;
                baseColor *= scanline;
                
                // Edge glow
                float edge = 1.0 - abs(TexCoord.x - 0.5) * 2.0;
                edge *= 1.0 - abs(TexCoord.y - 0.5) * 2.0;
                baseColor += vec3(0.0, 0.5, 1.0) * edge * 0.3;
                
                // Flicker
                alpha *= sin(time * 20.0) * 0.1 + 0.9;
            }
            else if (objectType == 1) { // UI_PANEL
                // Subtle glow for UI elements
                baseColor += vec3(0.1, 0.3, 0.5) * 0.2;
            }
            
            FragColor = vec4(baseColor, alpha);
        }
        """
        
        # Compile shaders (simplified - would use proper shader compilation)
        self.shader_program = self.compile_shader_program(vertex_shader_source, fragment_shader_source)
    
    def compile_shader_program(self, vertex_source: str, fragment_source: str):
        """Compile OpenGL shader program"""
        # This is a placeholder - actual shader compilation would be more complex
        return 1  # Mock shader program ID
    
    def add_ar_object(self, ar_object: ARObject):
        """Add AR object to scene"""
        self.ar_objects[ar_object.id] = ar_object
        print(f"Added AR object: {ar_object.name}")
    
    def remove_ar_object(self, object_id: str):
        """Remove AR object from scene"""
        if object_id in self.ar_objects:
            del self.ar_objects[object_id]
            print(f"Removed AR object: {object_id}")
    
    def render_frame(self, camera_transform: Transform3D, background_frame: Optional[np.ndarray] = None):
        """Render AR frame"""
        if not self.initialized:
            return
        
        # Clear buffers
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Render background frame if provided
        if background_frame is not None:
            self.render_background(background_frame)
        
        # Set up view matrix from camera transform
        self.view_matrix = np.linalg.inv(camera_transform.to_matrix())
        
        # Use AR shader
        gl.glUseProgram(self.shader_program)
        
        # Set uniforms
        current_time = time.time()
        gl.glUniform1f(gl.glGetUniformLocation(self.shader_program, "time"), current_time)
        
        # Render all AR objects
        self.render_ar_objects()
        
        # Swap buffers
        pygame.display.flip()
    
    def render_background(self, frame: np.ndarray):
        """Render camera background"""
        # Convert frame to OpenGL texture and render as fullscreen quad
        # This is simplified - would involve proper texture management
        pass
    
    def render_ar_objects(self):
        """Render all AR objects"""
        for ar_object in self.ar_objects.values():
            if ar_object.visible:
                self.render_single_object(ar_object)
    
    def render_single_object(self, ar_object: ARObject):
        """Render individual AR object"""
        # Set model matrix
        model_matrix = ar_object.transform.to_matrix()
        
        # Set uniforms
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(self.shader_program, "model"),
            1, gl.GL_FALSE, model_matrix.T.astype(np.float32)
        )
        
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(self.shader_program, "view"),
            1, gl.GL_FALSE, self.view_matrix.T.astype(np.float32)
        )
        
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(self.shader_program, "projection"),
            1, gl.GL_FALSE, self.projection_matrix.T.astype(np.float32)
        )
        
        # Set object type for shader
        object_type_map = {
            ARObjectType.HOLOGRAM: 0,
            ARObjectType.UI_PANEL: 1,
            ARObjectType.DATA_VISUALIZATION: 2
        }
        
        object_type = object_type_map.get(ar_object.object_type, 0)
        gl.glUniform1i(gl.glGetUniformLocation(self.shader_program, "objectType"), object_type)
        
        # Render based on object type
        if ar_object.object_type == ARObjectType.HOLOGRAM:
            self.render_hologram(ar_object)
        elif ar_object.object_type == ARObjectType.UI_PANEL:
            self.render_ui_panel(ar_object)
        elif ar_object.object_type == ARObjectType.DATA_VISUALIZATION:
            self.render_data_viz(ar_object)
        elif ar_object.object_type == ARObjectType.VIRTUAL_BUTTON:
            self.render_virtual_button(ar_object)
    
    def render_hologram(self, ar_object: ARObject):
        """Render holographic object"""
        # Render a glowing cube as example
        self.render_wireframe_cube(ar_object.size, ar_object.color)
    
    def render_ui_panel(self, ar_object: ARObject):
        """Render UI panel"""
        # Render a semi-transparent quad with text
        self.render_quad(ar_object.size[:2], ar_object.color)
    
    def render_data_viz(self, ar_object: ARObject):
        """Render data visualization"""
        # Render based on data type
        if ar_object.data.get('type') == 'bar_chart':
            self.render_3d_bar_chart(ar_object.data.get('values', []))
        elif ar_object.data.get('type') == 'line_chart':
            self.render_3d_line_chart(ar_object.data.get('values', []))
    
    def render_virtual_button(self, ar_object: ARObject):
        """Render virtual button"""
        # Render interactive button
        color = ar_object.color
        if ar_object.data.get('hovered', False):
            color = (color[0] * 1.5, color[1] * 1.5, color[2] * 1.5, color[3])
        
        self.render_rounded_cube(ar_object.size, color)
    
    def render_wireframe_cube(self, size: Tuple[float, float, float], color: Tuple[float, float, float, float]):
        """Render wireframe cube"""
        # Simplified wireframe cube rendering
        gl.glColor4f(*color)
        gl.glLineWidth(2.0)
        
        # This would render actual cube geometry
        # Placeholder for OpenGL cube rendering
        pass
    
    def render_quad(self, size: Tuple[float, float], color: Tuple[float, float, float, float]):
        """Render quad"""
        gl.glColor4f(*color)
        # Render quad geometry
        pass
    
    def render_3d_bar_chart(self, values: List[float]):
        """Render 3D bar chart"""
        if not values:
            return
        
        bar_width = 0.8 / len(values)
        max_height = max(values) if values else 1.0
        
        for i, value in enumerate(values):
            x = (i - len(values)/2) * bar_width * 2
            height = (value / max_height) * 2.0
            
            # Render bar at position (x, 0, 0) with height
            self.render_bar(x, height, bar_width)
    
    def render_bar(self, x: float, height: float, width: float):
        """Render individual bar"""
        # Render 3D bar geometry
        pass

class GestureARController:
    """Controls AR objects through gestures"""
    
    def __init__(self, ar_renderer: ARRenderer):
        self.ar_renderer = ar_renderer
        self.gesture_mappings = {}
        self.active_selections = {}
        self.interaction_state = {}
        
        self.setup_default_gestures()
    
    def setup_default_gestures(self):
        """Setup default gesture to AR action mappings"""
        self.gesture_mappings = {
            'air_tap': self.handle_air_tap,
            'pinch_move': self.handle_pinch_move,
            'palm_push': self.handle_palm_push,
            'finger_point': self.handle_finger_point,
            'grab': self.handle_grab_gesture,
            'spread_fingers': self.handle_spread_fingers
        }
    
    def handle_gesture(self, gesture_name: str, gesture_data: Dict[str, Any], hand_transform: Transform3D):
        """Handle gesture input for AR interaction"""
        if gesture_name in self.gesture_mappings:
            self.gesture_mappings[gesture_name](gesture_data, hand_transform)
    
    def handle_air_tap(self, gesture_data: Dict[str, Any], hand_transform: Transform3D):
        """Handle air tap gesture - select/click AR objects"""
        # Cast ray from hand position
        ray_origin = hand_transform.position
        ray_direction = hand_transform.rotation[:, 2]  # Z-axis (forward)
        
        # Find intersected AR objects
        intersected_object = self.raycast_ar_objects(ray_origin, ray_direction)
        
        if intersected_object and intersected_object.interactive:
            # Trigger click callback
            if intersected_object.on_click:
                intersected_object.on_click(intersected_object, gesture_data)
            
            print(f"Air tap on AR object: {intersected_object.name}")
    
    def handle_pinch_move(self, gesture_data: Dict[str, Any], hand_transform: Transform3D):
        """Handle pinch and move gesture - manipulate AR objects"""
        pinch_strength = gesture_data.get('strength', 0.0)
        
        if pinch_strength > 0.7:  # Strong pinch
            # If not already grabbing, start grab
            if 'grabbed_object' not in self.interaction_state:
                ray_origin = hand_transform.position
                ray_direction = hand_transform.rotation[:, 2]
                
                grabbed_object = self.raycast_ar_objects(ray_origin, ray_direction)
                if grabbed_object and grabbed_object.interactive:
                    self.interaction_state['grabbed_object'] = grabbed_object
                    self.interaction_state['grab_offset'] = grabbed_object.transform.position - hand_transform.position
            
            # Move grabbed object
            if 'grabbed_object' in self.interaction_state:
                grabbed_object = self.interaction_state['grabbed_object']
                grab_offset = self.interaction_state['grab_offset']
                
                grabbed_object.transform.position = hand_transform.position + grab_offset
                print(f"Moving AR object: {grabbed_object.name}")
        
        else:  # Release
            if 'grabbed_object' in self.interaction_state:
                print(f"Released AR object: {self.interaction_state['grabbed_object'].name}")
                del self.interaction_state['grabbed_object']
                del self.interaction_state['grab_offset']
    
    def handle_palm_push(self, gesture_data: Dict[str, Any], hand_transform: Transform3D):
        """Handle palm push gesture - dismiss/close AR objects"""
        # Create a larger interaction sphere for palm gestures
        affected_objects = self.sphere_query_ar_objects(hand_transform.position, radius=0.3)
        
        for ar_object in affected_objects:
            if ar_object.object_type == ARObjectType.UI_PANEL:
                # Close UI panels
                ar_object.visible = False
                print(f"Closed AR panel: {ar_object.name}")
    
    def handle_finger_point(self, gesture_data: Dict[str, Any], hand_transform: Transform3D):
        """Handle finger pointing - hover effects"""
        finger_tip_pos = gesture_data.get('finger_tip_position')
        if finger_tip_pos is None:
            return
        
        # Check which objects are being pointed at
        ray_origin = np.array(finger_tip_pos)
        ray_direction = hand_transform.rotation[:, 2]
        
        pointed_object = self.raycast_ar_objects(ray_origin, ray_direction)
        
        # Clear previous hover states
        for ar_object in self.ar_renderer.ar_objects.values():
            ar_object.data['hovered'] = False
        
        # Set hover state
        if pointed_object:
            pointed_object.data['hovered'] = True
            if pointed_object.on_hover:
                pointed_object.on_hover(pointed_object, gesture_data)
    
    def handle_grab_gesture(self, gesture_data: Dict[str, Any], hand_transform: Transform3D):
        """Handle grab gesture - select multiple objects"""
        grab_strength = gesture_data.get('strength', 0.0)
        
        if grab_strength > 0.8:
            # Select objects in grab sphere
            selected_objects = self.sphere_query_ar_objects(hand_transform.position, radius=0.2)
            
            for ar_object in selected_objects:
                ar_object.data['selected'] = True
                print(f"Selected AR object: {ar_object.name}")
    
    def handle_spread_fingers(self, gesture_data: Dict[str, Any], hand_transform: Transform3D):
        """Handle spread fingers - create new AR objects"""
        # Create a new holographic interface at hand position
        new_object = ARObject(
            id=f"gesture_object_{int(time.time())}",
            name="Gesture-Created Interface",
            object_type=ARObjectType.UI_PANEL,
            color=(0.0, 0.8, 1.0, 0.7),
            size=(0.3, 0.2, 0.05)
        )
        
        new_object.transform.position = hand_transform.position + hand_transform.rotation[:, 2] * 0.3
        new_object.transform.rotation = hand_transform.rotation
        
        self.ar_renderer.add_ar_object(new_object)
        print(f"Created new AR interface: {new_object.name}")
    
    def raycast_ar_objects(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[ARObject]:
        """Cast ray and find first intersected AR object"""
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        closest_object = None
        closest_distance = float('inf')
        
        for ar_object in self.ar_renderer.ar_objects.values():
            if not ar_object.visible:
                continue
            
            # Simple sphere intersection test
            object_center = ar_object.transform.position
            object_radius = max(ar_object.size) / 2
            
            # Vector from ray origin to sphere center
            oc = ray_origin - object_center
            
            # Quadratic equation coefficients
            a = np.dot(ray_direction, ray_direction)
            b = 2.0 * np.dot(oc, ray_direction)
            c = np.dot(oc, oc) - object_radius * object_radius
            
            discriminant = b * b - 4 * a * c
            
            if discriminant >= 0:
                # Ray intersects sphere
                distance = (-b - np.sqrt(discriminant)) / (2 * a)
                
                if distance > 0 and distance < closest_distance:
                    closest_distance = distance
                    closest_object = ar_object
        
        return closest_object
    
    def sphere_query_ar_objects(self, center: np.ndarray, radius: float) -> List[ARObject]:
        """Find all AR objects within sphere"""
        objects_in_sphere = []
        
        for ar_object in self.ar_renderer.ar_objects.values():
            if not ar_object.visible:
                continue
            
            distance = np.linalg.norm(ar_object.transform.position - center)
            if distance <= radius:
                objects_in_sphere.append(ar_object)
        
        return objects_in_sphere

class JarvisARSystem:
    """Main JARVIS Augmented Reality System"""
    
    def __init__(self, tracking_mode: ARTrackingMode = ARTrackingMode.HYBRID):
        self.tracking_mode = tracking_mode
        
        # Core components
        self.marker_detector = ARMarkerDetector()
        self.slam_tracker = PlanarSLAM()
        self.hand_tracker = HandTrackingAR()
        self.ar_renderer = ARRenderer()
        self.gesture_controller = GestureARController(self.ar_renderer)
        
        # System state
        self.is_running = False
        self.current_camera_pose = Transform3D()
        self.tracking_quality = 0.0
        
        # AR scene management
        self.scene_objects = {}
        self.ui_elements = {}
        
        print("🥽 JARVIS AR System initialized")
        self.setup_default_scene()
    
    def setup_default_scene(self):
        """Setup default AR scene"""
        # Main JARVIS interface panel
        main_panel = ARObject(
            id="jarvis_main_panel",
            name="JARVIS Main Interface",
            object_type=ARObjectType.UI_PANEL,
            color=(0.0, 0.8, 1.0, 0.8),
            size=(0.4, 0.3, 0.02)
        )
        main_panel.transform.position = np.array([0.0, 0.0, -0.5])
        main_panel.data = {
            'title': 'J.A.R.V.I.S.',
            'subtitle': 'Just A Rather Very Intelligent System',
            'content': 'Ready for commands...'
        }
        
        # System status display
        status_display = ARObject(
            id="system_status",
            name="System Status",
            object_type=ARObjectType.DATA_VISUALIZATION,
            color=(0.0, 1.0, 0.5, 0.7),
            size=(0.3, 0.2, 0.1)
        )
        status_display.transform.position = np.array([0.5, 0.0, -0.4])
        status_display.data = {
            'type': 'bar_chart',
            'values': [75, 60, 90, 45],  # CPU, Memory, Network, Power
            'labels': ['CPU', 'Memory', 'Network', 'Power']
        }
        
        # Virtual buttons
        button_positions = [
            (np.array([-0.3, -0.2, -0.3]), "Voice Control"),
            (np.array([0.0, -0.2, -0.3]), "Smart Home"),
            (np.array([0.3, -0.2, -0.3]), "Data Analysis")
        ]
        
        for i, (pos, label) in enumerate(button_positions):
            button = ARObject(
                id=f"button_{i}",
                name=f"Button: {label}",
                object_type=ARObjectType.VIRTUAL_BUTTON,
                color=(0.2, 0.6, 1.0, 0.9),
                size=(0.12, 0.06, 0.03)
            )
            button.transform.position = pos
            button.data = {'label': label, 'hovered': False}
            button.on_click = self.button_click_handler
            
            self.ar_renderer.add_ar_object(button)
        
        # Add main objects
        self.ar_renderer.add_ar_object(main_panel)
        self.ar_renderer.add_ar_object(status_display)
        
        # Holographic logo
        logo_hologram = ARObject(
            id="jarvis_logo",
            name="JARVIS Logo",
            object_type=ARObjectType.HOLOGRAM,
            color=(0.0, 0.8, 1.0, 0.6),
            size=(0.2, 0.2, 0.2)
        )
        logo_hologram.transform.position = np.array([0.0, 0.3, -0.4])
        self.ar_renderer.add_ar_object(logo_hologram)
    
    def button_click_handler(self, ar_object: ARObject, gesture_data: Dict[str, Any]):
        """Handle button clicks"""
        button_label = ar_object.data.get('label', 'Unknown')
        print(f"🔘 AR Button clicked: {button_label}")
        
        # Visual feedback
        ar_object.color = (1.0, 0.5, 0.0, 1.0)  # Orange flash
        
        # Reset color after delay (would be done with proper animation system)
        threading.Timer(0.5, lambda: setattr(ar_object, 'color', (0.2, 0.6, 1.0, 0.9))).start()
        
        # Handle specific button actions
        if button_label == "Voice Control":
            self.activate_voice_ar_interface()
        elif button_label == "Smart Home":
            self.show_smart_home_ar_panel()
        elif button_label == "Data Analysis":
            self.show_data_analysis_ar()
    
    def activate_voice_ar_interface(self):
        """Activate voice control AR interface"""
        voice_panel = ARObject(
            id="voice_interface",
            name="Voice Control Interface",
            object_type=ARObjectType.UI_PANEL,
            color=(0.5, 1.0, 0.5, 0.8),
            size=(0.5, 0.4, 0.02)
        )
        voice_panel.transform.position = np.array([0.0, 0.1, -0.6])
        voice_panel.data = {
            'title': 'Voice Commands',
            'content': 'Listening... Say "Hey JARVIS" to start'
        }
        
        self.ar_renderer.add_ar_object(voice_panel)
        print("🎤 Voice AR interface activated")
    
    def show_smart_home_ar_panel(self):
        """Show smart home control AR panel"""
        smart_home_panel = ARObject(
            id="smart_home_panel",
            name="Smart Home Control",
            object_type=ARObjectType.UI_PANEL,
            color=(1.0, 0.7, 0.0, 0.8),
            size=(0.6, 0.5, 0.02)
        )
        smart_home_panel.transform.position = np.array([0.0, 0.2, -0.7])
        smart_home_panel.data = {
            'title': 'Smart Home Control',
            'devices': ['Living Room Lights', 'Thermostat', 'Security System', 'Kitchen Appliances']
        }
        
        self.ar_renderer.add_ar_object(smart_home_panel)
        print("🏠 Smart Home AR panel activated")
    
    def show_data_analysis_ar(self):
        """Show data analysis AR visualization"""
        # Create 3D data visualization
        data_viz = ARObject(
            id="data_analysis_3d",
            name="3D Data Analysis",
            object_type=ARObjectType.DATA_VISUALIZATION,
            color=(1.0, 0.2, 0.8, 0.7),
            size=(0.4, 0.3, 0.3)
        )
        data_viz.transform.position = np.array([0.0, 0.0, -0.8])
        data_viz.data = {
            'type': 'line_chart',
            'values': [10, 25, 30, 45, 60, 55, 70, 85, 90, 75],
            'title': 'System Performance Over Time'
        }
        
        self.ar_renderer.add_ar_object(data_viz)
        print("📊 Data Analysis AR visualization activated")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process camera frame for AR"""
        if not self.is_running:
            return frame
        
        try:
            # Update tracking based on mode
            camera_pose = None
            
            if self.tracking_mode in [ARTrackingMode.MARKER_BASED, ARTrackingMode.HYBRID]:
                # Try marker-based tracking first
                detected_markers = self.marker_detector.detect_markers(frame)
                if detected_markers:
                    # Use first detected marker as reference
                    marker_id = list(detected_markers.keys())[0]
                    marker_pose = detected_markers[marker_id]
                    
                    # Camera pose is inverse of marker pose
                    camera_pose = Transform3D()
                    camera_pose.position = -marker_pose.position
                    camera_pose.rotation = marker_pose.rotation.T
                    
                    self.tracking_quality = 1.0
                    
                    # Draw marker axes for debugging
                    frame = self.marker_detector.draw_marker_axes(frame, marker_pose)
            
            if camera_pose is None and self.tracking_mode in [ARTrackingMode.MARKERLESS, ARTrackingMode.HYBRID]:
                # Try SLAM tracking
                if not self.slam_tracker.is_tracking:
                    self.slam_tracker.initialize_tracking(frame)
                else:
                    camera_pose = self.slam_tracker.track_frame(frame)
                    if camera_pose:
                        self.tracking_quality = 0.8
            
            # Hand tracking for interaction
            hand_transforms = self.hand_tracker.process_hands(frame)
            
            # Process gestures for AR interaction
            for hand_label, hand_transform in hand_transforms.items():
                # Simple gesture detection (would integrate with advanced gesture system)
                # For now, just handle basic interactions
                self.gesture_controller.handle_gesture('finger_point', {}, hand_transform)
            
            # Update current camera pose
            if camera_pose:
                self.current_camera_pose = camera_pose
            
            # Render AR content
            if self.ar_renderer.initialized:
                self.ar_renderer.render_frame(self.current_camera_pose, frame)
            
            # Add AR overlay to frame
            frame = self.add_ar_overlay_to_frame(frame)
            
        except Exception as e:
            print(f"Error in AR processing: {e}")
        
        return frame
    
    def add_ar_overlay_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Add AR overlay information to camera frame"""
        # Add tracking status
        status_text = f"AR Tracking: {'ACTIVE' if self.tracking_quality > 0.5 else 'LOST'}"
        status_color = (0, 255, 0) if self.tracking_quality > 0.5 else (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Add tracking quality bar
        bar_width = int(200 * self.tracking_quality)
        cv2.rectangle(frame, (10, 50), (10 + bar_width, 70), status_color, -1)
        cv2.rectangle(frame, (10, 50), (210, 70), (255, 255, 255), 2)
        
        # Add object count
        object_count = len([obj for obj in self.ar_renderer.ar_objects.values() if obj.visible])
        cv2.putText(frame, f"AR Objects: {object_count}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def start_ar_system(self):
        """Start AR system"""
        self.is_running = True
        print("🚀 JARVIS AR System started")
    
    def stop_ar_system(self):
        """Stop AR system"""
        self.is_running = False
        print("⏹️ JARVIS AR System stopped")
    
    def get_ar_status(self) -> Dict[str, Any]:
        """Get AR system status"""
        return {
            'running': self.is_running,
            'tracking_mode': self.tracking_mode.value,
            'tracking_quality': self.tracking_quality,
            'camera_pose': {
                'position': self.current_camera_pose.position.tolist(),
                'rotation': self.current_camera_pose.rotation.tolist()
            },
            'ar_objects_count': len(self.ar_renderer.ar_objects),
            'visible_objects': len([obj for obj in self.ar_renderer.ar_objects.values() if obj.visible]),
            'renderer_initialized': self.ar_renderer.initialized
        }

# Example usage and testing
def main():
    """Test JARVIS AR system"""
    ar_system = JarvisARSystem(ARTrackingMode.HYBRID)
    
    # Start AR system
    ar_system.start_ar_system()
    
    # Simulate camera input
    cap = cv2.VideoCapture(0)
    
    print("🥽 JARVIS AR System running... Press 'q' to quit")
    print("📱 Show AR markers to the camera for tracking")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame through AR system
        ar_frame = ar_system.process_frame(frame)
        
        # Display result
        cv2.imshow('JARVIS AR Interface', ar_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset tracking
            ar_system.slam_tracker.is_tracking = False
            print("🔄 AR tracking reset")
        elif key == ord('s'):
            # Show status
            status = ar_system.get_ar_status()
            print(f"📊 AR Status: {json.dumps(status, indent=2)}")
    
    cap.release()
    cv2.destroyAllWindows()
    ar_system.stop_ar_system()

if __name__ == "__main__":
    main()