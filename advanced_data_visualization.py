"""
JARVIS Advanced Data Visualization System
========================================

This module provides comprehensive data visualization capabilities featuring:
- Real-time holographic data displays
- 3D charts and interactive graphs
- System monitoring dashboards
- Data stream visualization
- Interactive data manipulation
- Multi-dimensional data analysis
- Iron Man-style HUD displays
- Augmented reality data overlays
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import seaborn as sns
import cv2
import pygame
from pygame import gfxdraw
import OpenGL.GL as gl
import OpenGL.arrays.vbo as vbo
from OpenGL.GL import shaders
import glfw
import json
import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import sqlite3
from datetime import datetime, timedelta
import websocket
import psutil
import GPUtil

class VisualizationType(Enum):
    """Types of visualizations"""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    PIE_CHART = "pie_chart"
    RADAR_CHART = "radar_chart"
    TREEMAP = "treemap"
    NETWORK_GRAPH = "network_graph"
    HOLOGRAM_3D = "hologram_3d"
    PARTICLE_FIELD = "particle_field"
    MATRIX_DISPLAY = "matrix_display"
    HUD_OVERLAY = "hud_overlay"

class DataSource(Enum):
    """Data source types"""
    REAL_TIME = "real_time"
    DATABASE = "database"
    API = "api"
    FILE = "file"
    SENSOR = "sensor"
    SYSTEM_METRICS = "system_metrics"
    NETWORK = "network"
    CUSTOM = "custom"

@dataclass
class DataStream:
    """Represents a data stream"""
    id: str
    name: str
    source: DataSource
    data_type: str  # "numeric", "categorical", "time_series", "image", "text"
    update_frequency: float  # Hz
    buffer_size: int = 1000
    data_buffer: deque = field(default_factory=lambda: deque(maxlen=1000))
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    last_update: Optional[datetime] = None

@dataclass
class VisualizationConfig:
    """Configuration for a visualization"""
    id: str
    name: str
    viz_type: VisualizationType
    data_streams: List[str]  # Data stream IDs
    properties: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[int, int] = (0, 0)
    size: Tuple[int, int] = (400, 300)
    refresh_rate: float = 30.0  # FPS
    interactive: bool = True
    holographic: bool = False
    transparency: float = 1.0

class HolographicRenderer:
    """3D holographic visualization renderer"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.initialized = False
        self.shader_program = None
        self.vao = None
        self.vbo = None
        
        # Holographic effects
        self.particle_systems = {}
        self.holographic_materials = {}
        self.animation_time = 0.0
        
        # Initialize GLFW and OpenGL
        self.init_opengl()
    
    def init_opengl(self):
        """Initialize OpenGL context"""
        try:
            if not glfw.init():
                raise Exception("Failed to initialize GLFW")
            
            # Create window
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            
            self.window = glfw.create_window(self.width, self.height, "JARVIS Holographic Display", None, None)
            if not self.window:
                glfw.terminate()
                raise Exception("Failed to create GLFW window")
            
            glfw.make_context_current(self.window)
            
            # Setup OpenGL
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glClearColor(0.0, 0.0, 0.1, 1.0)
            
            self.create_shaders()
            self.initialized = True
            
        except Exception as e:
            print(f"Failed to initialize OpenGL: {e}")
            self.initialized = False
    
    def create_shaders(self):
        """Create holographic shaders"""
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        layout (location = 2) in vec2 aTexCoord;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform float time;
        
        out vec3 vertexColor;
        out vec2 texCoord;
        out float hologramIntensity;
        
        void main()
        {
            // Add holographic distortion
            vec3 pos = aPos;
            pos.y += sin(time * 2.0 + pos.x * 0.5) * 0.02;
            pos.x += cos(time * 1.5 + pos.z * 0.3) * 0.01;
            
            gl_Position = projection * view * model * vec4(pos, 1.0);
            vertexColor = aColor;
            texCoord = aTexCoord;
            
            // Calculate hologram intensity based on viewing angle
            hologramIntensity = abs(sin(time)) * 0.3 + 0.7;
        }
        """
        
        fragment_shader_source = """
        #version 330 core
        in vec3 vertexColor;
        in vec2 texCoord;
        in float hologramIntensity;
        
        uniform float time;
        uniform sampler2D dataTexture;
        
        out vec4 FragColor;
        
        void main()
        {
            vec3 color = vertexColor;
            
            // Add holographic scanlines
            float scanline = sin(gl_FragCoord.y * 0.5 + time * 5.0) * 0.1 + 0.9;
            
            // Add holographic flicker
            float flicker = sin(time * 20.0) * 0.05 + 0.95;
            
            // Add edge glow effect
            float edge = 1.0 - abs(texCoord.x - 0.5) * 2.0;
            edge *= 1.0 - abs(texCoord.y - 0.5) * 2.0;
            edge = pow(edge, 2.0);
            
            color *= scanline * flicker * hologramIntensity;
            color += vec3(0.0, 0.5, 1.0) * edge * 0.3;
            
            FragColor = vec4(color, hologramIntensity * 0.8);
        }
        """
        
        # Compile shaders
        vertex_shader = shaders.compileShader(vertex_shader_source, gl.GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(fragment_shader_source, gl.GL_FRAGMENT_SHADER)
        
        self.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
    
    def render_holographic_chart(self, data: np.ndarray, chart_type: str = "bar"):
        """Render holographic chart"""
        if not self.initialized:
            return
        
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(self.shader_program)
        
        # Update time uniform
        self.animation_time += 0.016  # ~60 FPS
        time_location = gl.glGetUniformLocation(self.shader_program, "time")
        gl.glUniform1f(time_location, self.animation_time)
        
        # Setup matrices (simplified for demo)
        model = np.eye(4, dtype=np.float32)
        view = np.eye(4, dtype=np.float32)
        projection = self.create_perspective_matrix(45.0, self.width/self.height, 0.1, 100.0)
        
        # Set matrix uniforms
        model_loc = gl.glGetUniformLocation(self.shader_program, "model")
        view_loc = gl.glGetUniformLocation(self.shader_program, "view")
        proj_loc = gl.glGetUniformLocation(self.shader_program, "projection")
        
        gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, model)
        gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, view)
        gl.glUniformMatrix4fv(proj_loc, 1, gl.GL_FALSE, projection)
        
        # Render based on chart type
        if chart_type == "bar":
            self.render_holographic_bars(data)
        elif chart_type == "line":
            self.render_holographic_lines(data)
        elif chart_type == "scatter":
            self.render_holographic_points(data)
        
        glfw.swap_buffers(self.window)
        glfw.poll_events()
    
    def render_holographic_bars(self, data: np.ndarray):
        """Render holographic bar chart"""
        # Create vertices for bars
        vertices = []
        colors = []
        
        bar_width = 0.8 / len(data)
        max_height = np.max(data) if len(data) > 0 else 1.0
        
        for i, value in enumerate(data):
            x = (i - len(data)/2) * bar_width * 2
            height = (value / max_height) * 2.0
            
            # Create bar vertices (simplified cube)
            bar_vertices = [
                [x - bar_width/2, 0, 0],
                [x + bar_width/2, 0, 0],
                [x + bar_width/2, height, 0],
                [x - bar_width/2, height, 0]
            ]
            
            vertices.extend(bar_vertices)
            
            # Add holographic blue color with height-based intensity
            intensity = value / max_height
            bar_color = [0.0, intensity * 0.8, 1.0]
            colors.extend([bar_color] * 4)
        
        # Render vertices (simplified - would use proper VBO in production)
        self.render_vertices(vertices, colors)
    
    def render_holographic_lines(self, data: np.ndarray):
        """Render holographic line chart"""
        vertices = []
        colors = []
        
        x_step = 4.0 / len(data) if len(data) > 1 else 1.0
        max_value = np.max(data) if len(data) > 0 else 1.0
        
        for i, value in enumerate(data):
            x = (i - len(data)/2) * x_step
            y = (value / max_value) * 2.0 - 1.0
            
            vertices.append([x, y, 0])
            colors.append([0.0, 0.8, 1.0])  # Holographic blue
        
        self.render_line_strip(vertices, colors)
    
    def render_vertices(self, vertices: List, colors: List):
        """Render vertices using OpenGL"""
        if not vertices:
            return
        
        # Convert to numpy arrays
        vertex_data = np.array(vertices, dtype=np.float32)
        color_data = np.array(colors, dtype=np.float32)
        
        # Simple immediate mode rendering (would use VBOs in production)
        gl.glBegin(gl.GL_QUADS)
        for i in range(0, len(vertices), 4):
            for j in range(4):
                if i + j < len(vertices):
                    gl.glColor3fv(color_data[i + j])
                    gl.glVertex3fv(vertex_data[i + j])
        gl.glEnd()
    
    def render_line_strip(self, vertices: List, colors: List):
        """Render line strip"""
        if not vertices:
            return
        
        vertex_data = np.array(vertices, dtype=np.float32)
        color_data = np.array(colors, dtype=np.float32)
        
        gl.glLineWidth(3.0)
        gl.glBegin(gl.GL_LINE_STRIP)
        for i, vertex in enumerate(vertices):
            gl.glColor3fv(color_data[i])
            gl.glVertex3fv(vertex_data[i])
        gl.glEnd()
    
    def create_perspective_matrix(self, fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create perspective projection matrix"""
        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

class DataStreamManager:
    """Manages data streams for visualization"""
    
    def __init__(self):
        self.streams: Dict[str, DataStream] = {}
        self.collectors: Dict[str, threading.Thread] = {}
        self.is_running = False
    
    def create_stream(self, stream_id: str, name: str, source: DataSource, 
                     data_type: str, update_frequency: float = 1.0) -> DataStream:
        """Create a new data stream"""
        stream = DataStream(
            id=stream_id,
            name=name,
            source=source,
            data_type=data_type,
            update_frequency=update_frequency
        )
        
        self.streams[stream_id] = stream
        return stream
    
    def start_stream(self, stream_id: str):
        """Start collecting data for a stream"""
        if stream_id not in self.streams:
            return False
        
        stream = self.streams[stream_id]
        
        if stream.source == DataSource.SYSTEM_METRICS:
            collector = threading.Thread(target=self.collect_system_metrics, args=(stream,))
        elif stream.source == DataSource.REAL_TIME:
            collector = threading.Thread(target=self.collect_real_time_data, args=(stream,))
        elif stream.source == DataSource.SENSOR:
            collector = threading.Thread(target=self.collect_sensor_data, args=(stream,))
        else:
            return False
        
        collector.daemon = True
        collector.start()
        self.collectors[stream_id] = collector
        stream.is_active = True
        
        return True
    
    def collect_system_metrics(self, stream: DataStream):
        """Collect system performance metrics"""
        while stream.is_active and self.is_running:
            try:
                if stream.name == "cpu_usage":
                    data = psutil.cpu_percent(interval=1)
                elif stream.name == "memory_usage":
                    data = psutil.virtual_memory().percent
                elif stream.name == "disk_usage":
                    data = psutil.disk_usage('/').percent
                elif stream.name == "network_io":
                    stats = psutil.net_io_counters()
                    data = stats.bytes_sent + stats.bytes_recv
                elif stream.name == "gpu_usage":
                    try:
                        gpus = GPUtil.getGPUs()
                        data = gpus[0].load * 100 if gpus else 0
                    except:
                        data = 0
                else:
                    data = np.random.random() * 100  # Default random data
                
                stream.data_buffer.append({
                    'timestamp': datetime.now(),
                    'value': data
                })
                stream.last_update = datetime.now()
                
                time.sleep(1.0 / stream.update_frequency)
                
            except Exception as e:
                print(f"Error collecting system metrics for {stream.name}: {e}")
                time.sleep(1)
    
    def collect_real_time_data(self, stream: DataStream):
        """Collect real-time data (simulated)"""
        while stream.is_active and self.is_running:
            try:
                # Generate simulated real-time data
                base_value = 50
                noise = np.random.normal(0, 10)
                trend = np.sin(time.time() * 0.1) * 20
                data = base_value + noise + trend
                
                stream.data_buffer.append({
                    'timestamp': datetime.now(),
                    'value': data
                })
                stream.last_update = datetime.now()
                
                time.sleep(1.0 / stream.update_frequency)
                
            except Exception as e:
                print(f"Error collecting real-time data for {stream.name}: {e}")
                time.sleep(1)
    
    def collect_sensor_data(self, stream: DataStream):
        """Collect sensor data (simulated)"""
        while stream.is_active and self.is_running:
            try:
                # Simulate sensor readings
                if stream.name == "temperature":
                    data = 20 + np.random.normal(0, 2)  # Temperature around 20°C
                elif stream.name == "humidity":
                    data = 50 + np.random.normal(0, 5)  # Humidity around 50%
                elif stream.name == "light_level":
                    data = 300 + np.random.normal(0, 50)  # Light level in lux
                else:
                    data = np.random.random() * 100
                
                stream.data_buffer.append({
                    'timestamp': datetime.now(),
                    'value': data
                })
                stream.last_update = datetime.now()
                
                time.sleep(1.0 / stream.update_frequency)
                
            except Exception as e:
                print(f"Error collecting sensor data for {stream.name}: {e}")
                time.sleep(1)
    
    def get_stream_data(self, stream_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get data from a stream"""
        if stream_id not in self.streams:
            return []
        
        stream = self.streams[stream_id]
        data = list(stream.data_buffer)
        
        if limit:
            data = data[-limit:]
        
        return data
    
    def start_all_streams(self):
        """Start all data collection"""
        self.is_running = True
        for stream_id in self.streams:
            self.start_stream(stream_id)
    
    def stop_all_streams(self):
        """Stop all data collection"""
        self.is_running = False
        for stream in self.streams.values():
            stream.is_active = False

class InteractiveChart:
    """Interactive chart with Iron Man styling"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.fig = None
        self.ax = None
        self.canvas = None
        self.animation = None
        self.setup_chart()
    
    def setup_chart(self):
        """Setup matplotlib chart with Iron Man styling"""
        plt.style.use('dark_background')
        
        self.fig, self.ax = plt.subplots(figsize=(self.config.size[0]/100, self.config.size[1]/100))
        
        # Iron Man color scheme
        self.fig.patch.set_facecolor('#0a0a0a')
        self.ax.set_facecolor('#1a1a1a')
        
        # Grid styling
        self.ax.grid(True, alpha=0.3, color='#00ccff')
        self.ax.spines['bottom'].set_color('#00ccff')
        self.ax.spines['top'].set_color('#00ccff')
        self.ax.spines['right'].set_color('#00ccff')
        self.ax.spines['left'].set_color('#00ccff')
        
        # Text color
        self.ax.tick_params(colors='#00ccff')
        self.ax.xaxis.label.set_color('#00ccff')
        self.ax.yaxis.label.set_color('#00ccff')
        self.ax.title.set_color('#ffffff')
    
    def update_data(self, data: List[Dict[str, Any]]):
        """Update chart with new data"""
        if not data:
            return
        
        self.ax.clear()
        self.setup_chart()
        
        # Extract values and timestamps
        values = [d['value'] for d in data]
        timestamps = [d['timestamp'] for d in data]
        
        if self.config.viz_type == VisualizationType.LINE_CHART:
            self.ax.plot(timestamps, values, color='#00ccff', linewidth=2, alpha=0.8)
            self.ax.fill_between(timestamps, values, alpha=0.2, color='#00ccff')
            
        elif self.config.viz_type == VisualizationType.BAR_CHART:
            bars = self.ax.bar(range(len(values)), values, color='#00ccff', alpha=0.7)
            
            # Add glow effect to bars
            for bar in bars:
                height = bar.get_height()
                self.ax.bar(bar.get_x(), height, bar.get_width(), 
                          color='#66ddff', alpha=0.3, bottom=height)
        
        elif self.config.viz_type == VisualizationType.SCATTER_PLOT:
            x_vals = range(len(values))
            colors = ['#ff6b6b' if v > np.mean(values) else '#00ccff' for v in values]
            self.ax.scatter(x_vals, values, c=colors, alpha=0.7, s=50)
        
        # Set title and labels
        self.ax.set_title(self.config.name, fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Time', fontsize=10)
        self.ax.set_ylabel('Value', fontsize=10)
        
        # Auto-scale
        if values:
            self.ax.set_ylim(min(values) * 0.95, max(values) * 1.05)
        
        plt.tight_layout()

class HUDOverlay:
    """Heads-up display overlay for real-time data"""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.surface = None
        self.font = None
        self.data_elements = {}
        self.init_hud()
    
    def init_hud(self):
        """Initialize HUD surface"""
        pygame.init()
        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 18)
    
    def add_data_element(self, element_id: str, position: Tuple[int, int], 
                        size: Tuple[int, int], element_type: str = "text"):
        """Add a data display element to the HUD"""
        self.data_elements[element_id] = {
            'position': position,
            'size': size,
            'type': element_type,
            'value': None,
            'last_update': None
        }
    
    def update_element(self, element_id: str, value: Any):
        """Update HUD element value"""
        if element_id in self.data_elements:
            self.data_elements[element_id]['value'] = value
            self.data_elements[element_id]['last_update'] = time.time()
    
    def render_hud(self) -> pygame.Surface:
        """Render HUD overlay"""
        self.surface.fill((0, 0, 0, 0))  # Clear with transparency
        
        # Render each data element
        for element_id, element in self.data_elements.items():
            self.render_element(element_id, element)
        
        # Render system status
        self.render_system_status()
        
        # Render JARVIS branding
        self.render_jarvis_branding()
        
        return self.surface
    
    def render_element(self, element_id: str, element: Dict[str, Any]):
        """Render individual HUD element"""
        pos = element['position']
        size = element['size']
        value = element['value']
        
        if value is None:
            return
        
        # Draw element background
        rect = pygame.Rect(pos[0], pos[1], size[0], size[1])
        pygame.draw.rect(self.surface, (0, 50, 100, 100), rect)
        pygame.draw.rect(self.surface, (0, 200, 255), rect, 2)
        
        # Render value based on type
        if element['type'] == 'text':
            text_surface = self.font.render(str(value), True, (0, 255, 255))
            self.surface.blit(text_surface, (pos[0] + 10, pos[1] + 10))
        
        elif element['type'] == 'bar':
            # Render as progress bar
            if isinstance(value, (int, float)):
                bar_width = int((size[0] - 20) * (value / 100))
                bar_rect = pygame.Rect(pos[0] + 10, pos[1] + size[1]//2, bar_width, 10)
                pygame.draw.rect(self.surface, (0, 255, 255), bar_rect)
                
                # Add value text
                text = f"{value:.1f}%"
                text_surface = self.small_font.render(text, True, (255, 255, 255))
                self.surface.blit(text_surface, (pos[0] + 10, pos[1] + 5))
        
        elif element['type'] == 'gauge':
            # Render as circular gauge
            center = (pos[0] + size[0]//2, pos[1] + size[1]//2)
            radius = min(size[0], size[1]) // 2 - 10
            
            # Draw gauge background
            pygame.draw.circle(self.surface, (0, 50, 100), center, radius, 2)
            
            # Draw gauge needle
            if isinstance(value, (int, float)):
                angle = (value / 100) * 270 - 135  # -135 to 135 degrees
                angle_rad = np.radians(angle)
                end_x = center[0] + int(radius * 0.8 * np.cos(angle_rad))
                end_y = center[1] + int(radius * 0.8 * np.sin(angle_rad))
                pygame.draw.line(self.surface, (255, 100, 100), center, (end_x, end_y), 3)
    
    def render_system_status(self):
        """Render system status indicators"""
        status_y = 50
        
        # System time
        current_time = datetime.now().strftime("%H:%M:%S")
        time_surface = self.large_font.render(current_time, True, (0, 255, 255))
        self.surface.blit(time_surface, (self.width - 200, status_y))
        
        # Connection status
        status_text = "ONLINE"
        status_color = (0, 255, 0)
        status_surface = self.font.render(status_text, True, status_color)
        self.surface.blit(status_surface, (self.width - 200, status_y + 40))
    
    def render_jarvis_branding(self):
        """Render JARVIS branding"""
        # JARVIS logo text
        logo_surface = self.large_font.render("J.A.R.V.I.S.", True, (0, 255, 255))
        self.surface.blit(logo_surface, (50, 50))
        
        # Subtitle
        subtitle_surface = self.small_font.render("Just A Rather Very Intelligent System", True, (150, 150, 150))
        self.surface.blit(subtitle_surface, (50, 85))
        
        # Draw arc reactor symbol
        center = (150, 150)
        pygame.draw.circle(self.surface, (0, 200, 255), center, 30, 3)
        pygame.draw.circle(self.surface, (100, 200, 255), center, 20, 2)
        pygame.draw.circle(self.surface, (200, 200, 255), center, 10, 1)

class JarvisDataVisualization:
    """Main JARVIS Data Visualization System"""
    
    def __init__(self):
        self.stream_manager = DataStreamManager()
        self.holographic_renderer = HolographicRenderer()
        self.hud_overlay = HUDOverlay()
        self.charts: Dict[str, InteractiveChart] = {}
        self.visualizations: Dict[str, VisualizationConfig] = {}
        
        # Web dashboard
        self.dash_app = None
        self.web_server_thread = None
        
        self.is_running = False
        self.setup_default_streams()
        self.setup_default_visualizations()
        
        print("🎨 JARVIS Data Visualization System initialized")
    
    def setup_default_streams(self):
        """Setup default data streams"""
        # System metrics
        self.stream_manager.create_stream("cpu_usage", "CPU Usage", DataSource.SYSTEM_METRICS, "numeric", 1.0)
        self.stream_manager.create_stream("memory_usage", "Memory Usage", DataSource.SYSTEM_METRICS, "numeric", 1.0)
        self.stream_manager.create_stream("gpu_usage", "GPU Usage", DataSource.SYSTEM_METRICS, "numeric", 1.0)
        self.stream_manager.create_stream("network_io", "Network I/O", DataSource.SYSTEM_METRICS, "numeric", 1.0)
        
        # Sensor data (simulated)
        self.stream_manager.create_stream("temperature", "Temperature", DataSource.SENSOR, "numeric", 0.5)
        self.stream_manager.create_stream("humidity", "Humidity", DataSource.SENSOR, "numeric", 0.5)
        
        # Real-time data
        self.stream_manager.create_stream("power_consumption", "Power Consumption", DataSource.REAL_TIME, "numeric", 2.0)
        self.stream_manager.create_stream("network_latency", "Network Latency", DataSource.REAL_TIME, "numeric", 1.0)
    
    def setup_default_visualizations(self):
        """Setup default visualizations"""
        # System performance dashboard
        self.create_visualization(
            "system_dashboard",
            "System Performance",
            VisualizationType.LINE_CHART,
            ["cpu_usage", "memory_usage", "gpu_usage"],
            position=(100, 100),
            size=(600, 400)
        )
        
        # Environmental sensors
        self.create_visualization(
            "environment_chart",
            "Environmental Data",
            VisualizationType.BAR_CHART,
            ["temperature", "humidity"],
            position=(100, 550),
            size=(400, 300)
        )
        
        # Holographic display
        self.create_visualization(
            "holographic_display",
            "Holographic Data",
            VisualizationType.HOLOGRAM_3D,
            ["power_consumption"],
            position=(750, 100),
            size=(500, 400),
            holographic=True
        )
        
        # HUD elements
        self.hud_overlay.add_data_element("cpu_gauge", (1600, 100), (150, 150), "gauge")
        self.hud_overlay.add_data_element("memory_bar", (1600, 300), (200, 50), "bar")
        self.hud_overlay.add_data_element("gpu_text", (1600, 400), (200, 30), "text")
    
    def create_visualization(self, viz_id: str, name: str, viz_type: VisualizationType,
                           data_streams: List[str], position: Tuple[int, int] = (0, 0),
                           size: Tuple[int, int] = (400, 300), **kwargs) -> bool:
        """Create a new visualization"""
        config = VisualizationConfig(
            id=viz_id,
            name=name,
            viz_type=viz_type,
            data_streams=data_streams,
            position=position,
            size=size,
            **kwargs
        )
        
        self.visualizations[viz_id] = config
        
        # Create chart if it's a standard chart type
        if viz_type in [VisualizationType.LINE_CHART, VisualizationType.BAR_CHART, VisualizationType.SCATTER_PLOT]:
            self.charts[viz_id] = InteractiveChart(config)
        
        print(f"Created visualization: {name} ({viz_type.value})")
        return True
    
    def start_system(self):
        """Start the visualization system"""
        print("🚀 Starting JARVIS Data Visualization System...")
        
        # Start data streams
        self.stream_manager.start_all_streams()
        
        # Start web dashboard
        self.start_web_dashboard()
        
        # Start main update loop
        self.is_running = True
        self.update_loop()
    
    def start_web_dashboard(self):
        """Start web-based dashboard"""
        self.dash_app = dash.Dash(__name__)
        
        # Define dashboard layout
        self.dash_app.layout = html.Div([
            html.H1("JARVIS Data Visualization Dashboard", 
                   style={'color': '#00ccff', 'textAlign': 'center', 'backgroundColor': '#1a1a1a'}),
            
            html.Div([
                dcc.Graph(id='system-metrics'),
                dcc.Graph(id='environmental-data'),
                dcc.Graph(id='network-stats'),
                dcc.Graph(id='power-consumption')
            ], style={'backgroundColor': '#0a0a0a'}),
            
            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every second
                n_intervals=0
            )
        ])
        
        # Define callbacks
        @self.dash_app.callback(
            [Output('system-metrics', 'figure'),
             Output('environmental-data', 'figure'),
             Output('network-stats', 'figure'),
             Output('power-consumption', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n):
            return self.generate_dashboard_figures()
        
        # Start web server in separate thread
        self.web_server_thread = threading.Thread(
            target=lambda: self.dash_app.run_server(debug=False, host='0.0.0.0', port=8050)
        )
        self.web_server_thread.daemon = True
        self.web_server_thread.start()
        
        print("📊 Web dashboard started at http://localhost:8050")
    
    def generate_dashboard_figures(self) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
        """Generate figures for web dashboard"""
        # System metrics
        cpu_data = self.stream_manager.get_stream_data("cpu_usage", 50)
        memory_data = self.stream_manager.get_stream_data("memory_usage", 50)
        gpu_data = self.stream_manager.get_stream_data("gpu_usage", 50)
        
        system_fig = go.Figure()
        if cpu_data:
            timestamps = [d['timestamp'] for d in cpu_data]
            system_fig.add_trace(go.Scatter(
                x=timestamps, y=[d['value'] for d in cpu_data],
                name='CPU', line=dict(color='#00ccff')
            ))
        if memory_data:
            timestamps = [d['timestamp'] for d in memory_data]
            system_fig.add_trace(go.Scatter(
                x=timestamps, y=[d['value'] for d in memory_data],
                name='Memory', line=dict(color='#ff6b6b')
            ))
        if gpu_data:
            timestamps = [d['timestamp'] for d in gpu_data]
            system_fig.add_trace(go.Scatter(
                x=timestamps, y=[d['value'] for d in gpu_data],
                name='GPU', line=dict(color='#4ecdc4')
            ))
        
        system_fig.update_layout(
            title='System Performance',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='#ffffff')
        )
        
        # Environmental data
        temp_data = self.stream_manager.get_stream_data("temperature", 20)
        humidity_data = self.stream_manager.get_stream_data("humidity", 20)
        
        env_fig = go.Figure()
        if temp_data and humidity_data:
            env_fig.add_trace(go.Bar(
                x=['Temperature', 'Humidity'],
                y=[temp_data[-1]['value'] if temp_data else 0, 
                   humidity_data[-1]['value'] if humidity_data else 0],
                marker=dict(color=['#ff9500', '#00ccff'])
            ))
        
        env_fig.update_layout(
            title='Environmental Sensors',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='#ffffff')
        )
        
        # Network stats (placeholder)
        network_fig = go.Figure()
        network_fig.add_trace(go.Scatter(
            x=list(range(10)), y=np.random.random(10) * 100,
            name='Latency', line=dict(color='#45b7d1')
        ))
        network_fig.update_layout(
            title='Network Statistics',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='#ffffff')
        )
        
        # Power consumption
        power_data = self.stream_manager.get_stream_data("power_consumption", 30)
        power_fig = go.Figure()
        if power_data:
            timestamps = [d['timestamp'] for d in power_data]
            power_fig.add_trace(go.Scatter(
                x=timestamps, y=[d['value'] for d in power_data],
                fill='tozeroy', line=dict(color='#ffd700')
            ))
        
        power_fig.update_layout(
            title='Power Consumption',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='#ffffff')
        )
        
        return system_fig, env_fig, network_fig, power_fig
    
    def update_loop(self):
        """Main update loop for visualizations"""
        while self.is_running:
            try:
                # Update standard charts
                for viz_id, chart in self.charts.items():
                    config = self.visualizations[viz_id]
                    
                    # Get data from first stream (simplified)
                    if config.data_streams:
                        stream_id = config.data_streams[0]
                        data = self.stream_manager.get_stream_data(stream_id, 50)
                        if data:
                            chart.update_data(data)
                
                # Update HUD overlay
                cpu_data = self.stream_manager.get_stream_data("cpu_usage", 1)
                memory_data = self.stream_manager.get_stream_data("memory_usage", 1)
                gpu_data = self.stream_manager.get_stream_data("gpu_usage", 1)
                
                if cpu_data:
                    self.hud_overlay.update_element("cpu_gauge", cpu_data[-1]['value'])
                if memory_data:
                    self.hud_overlay.update_element("memory_bar", memory_data[-1]['value'])
                if gpu_data:
                    self.hud_overlay.update_element("gpu_text", f"GPU: {gpu_data[-1]['value']:.1f}%")
                
                # Update holographic displays
                for viz_id, config in self.visualizations.items():
                    if config.holographic and config.data_streams:
                        stream_id = config.data_streams[0]
                        data = self.stream_manager.get_stream_data(stream_id, 20)
                        if data:
                            values = np.array([d['value'] for d in data])
                            self.holographic_renderer.render_holographic_chart(values, "bar")
                
                time.sleep(1.0 / 30)  # 30 FPS update rate
                
            except Exception as e:
                print(f"Error in visualization update loop: {e}")
                time.sleep(1)
    
    def get_hud_overlay(self) -> pygame.Surface:
        """Get current HUD overlay surface"""
        return self.hud_overlay.render_hud()
    
    def shutdown(self):
        """Shutdown visualization system"""
        print("🔌 Shutting down Data Visualization System...")
        self.is_running = False
        self.stream_manager.stop_all_streams()
        
        if hasattr(self.holographic_renderer, 'window'):
            glfw.terminate()
        
        print("✅ Data Visualization System shutdown complete")

# Example usage
def main():
    """Test the data visualization system"""
    viz_system = JarvisDataVisualization()
    
    try:
        # Start the system
        viz_system.start_system()
        
        print("🎨 JARVIS Data Visualization System running...")
        print("📊 Web dashboard: http://localhost:8050")
        print("Press Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Stopping system...")
        viz_system.shutdown()

if __name__ == "__main__":
    main()