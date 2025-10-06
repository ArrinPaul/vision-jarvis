"""
Iron Man JARVIS - Advanced Holographic Interface System
=======================================================

This module creates a futuristic holographic-style interface inspired by Tony Stark's JARVIS.
Features include:
- Holographic visual effects with glowing elements
- Particle systems and animated UI components  
- 3D interface elements with depth and perspective
- Advanced animations and transitions
- Real-time system monitoring displays
- AI status indicators
"""

import cv2
import numpy as np
import pygame
import math
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import json
from pygame.locals import *

# Try to import OpenGL for advanced 3D effects
try:
    import OpenGL.GL as gl
    import OpenGL.GLU as glu
    from pygame.opengl import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("OpenGL not available - using fallback 2D renderer")

class UITheme(Enum):
    """JARVIS UI Themes"""
    CLASSIC_BLUE = "classic_blue"
    ARC_REACTOR = "arc_reactor" 
    STEALTH = "stealth"
    MARK_85 = "mark_85"
    FRIDAY = "friday"

@dataclass
class HolographicConfig:
    """Configuration for holographic effects"""
    glow_intensity: float = 0.8
    particle_count: int = 150
    scan_line_speed: float = 2.0
    pulse_frequency: float = 1.5
    transparency: float = 0.7
    theme: UITheme = UITheme.ARC_REACTOR
    enable_3d: bool = True
    enable_particles: bool = True
    enable_animations: bool = True

class ParticleSystem:
    """Advanced particle system for holographic effects"""
    
    def __init__(self, config: HolographicConfig):
        self.config = config
        self.particles = []
        self.init_particles()
    
    def init_particles(self):
        """Initialize particle system"""
        for i in range(self.config.particle_count):
            particle = {
                'x': np.random.random() * 1920,
                'y': np.random.random() * 1080,
                'vx': (np.random.random() - 0.5) * 2,
                'vy': (np.random.random() - 0.5) * 2,
                'life': np.random.random(),
                'size': np.random.random() * 3 + 1,
                'color': self.get_theme_color(),
                'type': np.random.choice(['dot', 'line', 'glow'])
            }
            self.particles.append(particle)
    
    def get_theme_color(self) -> Tuple[int, int, int]:
        """Get color based on current theme"""
        theme_colors = {
            UITheme.CLASSIC_BLUE: (100, 200, 255),
            UITheme.ARC_REACTOR: (150, 220, 255),
            UITheme.STEALTH: (100, 100, 120),
            UITheme.MARK_85: (255, 215, 100),
            UITheme.FRIDAY: (255, 100, 150)
        }
        return theme_colors.get(self.config.theme, (100, 200, 255))
    
    def update(self, dt: float):
        """Update particle positions and states"""
        for particle in self.particles:
            # Update position
            particle['x'] += particle['vx'] * dt * 60
            particle['y'] += particle['vy'] * dt * 60
            
            # Update life
            particle['life'] -= dt * 0.5
            
            # Wrap around screen
            if particle['x'] < 0: particle['x'] = 1920
            if particle['x'] > 1920: particle['x'] = 0
            if particle['y'] < 0: particle['y'] = 1080
            if particle['y'] > 1080: particle['y'] = 0
            
            # Respawn if dead
            if particle['life'] <= 0:
                particle['life'] = 1.0
                particle['x'] = np.random.random() * 1920
                particle['y'] = np.random.random() * 1080
    
    def render(self, surface):
        """Render particles to surface"""
        for particle in self.particles:
            alpha = int(particle['life'] * 255 * self.config.transparency)
            color = (*particle['color'], alpha)
            size = int(particle['size'] * particle['life'])
            
            if particle['type'] == 'dot':
                pygame.draw.circle(surface, particle['color'], 
                                 (int(particle['x']), int(particle['y'])), size)
            elif particle['type'] == 'glow':
                self.draw_glow_particle(surface, particle)

    def draw_glow_particle(self, surface, particle):
        """Draw glowing particle effect"""
        pos = (int(particle['x']), int(particle['y']))
        color = particle['color']
        alpha = int(particle['life'] * 255 * self.config.transparency)
        
        # Create multiple circles for glow effect
        for i in range(5):
            radius = int(particle['size'] * (i + 1) * 2)
            glow_alpha = max(0, alpha - i * 50)
            glow_color = (*color, glow_alpha)
            
            # Create temporary surface for alpha blending
            temp_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, (*color, glow_alpha), 
                             (radius, radius), radius)
            surface.blit(temp_surface, (pos[0] - radius, pos[1] - radius), 
                        special_flags=pygame.BLEND_ADD)

class HolographicPanel:
    """Individual holographic UI panel"""
    
    def __init__(self, x: int, y: int, width: int, height: int, title: str, config: HolographicConfig):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.title = title
        self.config = config
        self.visible = True
        self.alpha = 255
        self.content = []
        self.scan_offset = 0
        self.pulse_phase = 0
        
    def add_content(self, content_type: str, data: Dict):
        """Add content to the panel"""
        self.content.append({
            'type': content_type,
            'data': data,
            'timestamp': time.time()
        })
        
        # Keep only recent content
        if len(self.content) > 10:
            self.content = self.content[-10:]
    
    def update(self, dt: float):
        """Update panel animations"""
        self.scan_offset += self.config.scan_line_speed * dt * 100
        if self.scan_offset > self.height:
            self.scan_offset = -20
            
        self.pulse_phase += self.config.pulse_frequency * dt * 2 * math.pi
        if self.pulse_phase > 2 * math.pi:
            self.pulse_phase -= 2 * math.pi
    
    def render(self, surface):
        """Render the holographic panel"""
        if not self.visible:
            return
            
        # Create panel surface with alpha
        panel_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw panel background with glow
        self.draw_holographic_background(panel_surface)
        
        # Draw panel border
        self.draw_holographic_border(panel_surface)
        
        # Draw title
        self.draw_title(panel_surface)
        
        # Draw content
        self.draw_content(panel_surface)
        
        # Draw scan lines
        self.draw_scan_lines(panel_surface)
        
        # Blit to main surface
        surface.blit(panel_surface, (self.x, self.y))
    
    def draw_holographic_background(self, surface):
        """Draw holographic background effect"""
        base_color = self.get_theme_color()
        
        # Create gradient background
        for y in range(self.height):
            alpha = int(30 + 20 * math.sin(y * 0.01 + time.time()))
            color = (*base_color, alpha)
            pygame.draw.line(surface, base_color, (0, y), (self.width, y))
    
    def draw_holographic_border(self, surface):
        """Draw animated holographic border"""
        color = self.get_theme_color()
        pulse_alpha = int(128 + 127 * math.sin(self.pulse_phase))
        pulse_color = (*color, pulse_alpha)
        
        # Draw corner elements
        corner_size = 20
        thickness = 2
        
        # Top-left corner
        pygame.draw.line(surface, color, (0, 0), (corner_size, 0), thickness)
        pygame.draw.line(surface, color, (0, 0), (0, corner_size), thickness)
        
        # Top-right corner
        pygame.draw.line(surface, color, (self.width - corner_size, 0), (self.width, 0), thickness)
        pygame.draw.line(surface, color, (self.width, 0), (self.width, corner_size), thickness)
        
        # Bottom-left corner
        pygame.draw.line(surface, color, (0, self.height - corner_size), (0, self.height), thickness)
        pygame.draw.line(surface, color, (0, self.height), (corner_size, self.height), thickness)
        
        # Bottom-right corner
        pygame.draw.line(surface, color, (self.width - corner_size, self.height), (self.width, self.height), thickness)
        pygame.draw.line(surface, color, (self.width, self.height - corner_size), (self.width, self.height), thickness)
        
        # Pulsing side lines
        mid_y = self.height // 2
        pygame.draw.line(surface, pulse_color, (0, mid_y - 1), (0, mid_y + 1), thickness * 2)
        pygame.draw.line(surface, pulse_color, (self.width, mid_y - 1), (self.width, mid_y + 1), thickness * 2)
    
    def draw_title(self, surface):
        """Draw panel title with glow effect"""
        if not hasattr(self, 'font'):
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)
        
        color = self.get_theme_color()
        title_surface = self.font.render(self.title, True, color)
        
        # Draw title with glow
        glow_surface = self.font.render(self.title, True, (255, 255, 255))
        
        # Position title
        title_x = 10
        title_y = 5
        
        # Draw glow effect
        for offset in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
            surface.blit(glow_surface, (title_x + offset[0], title_y + offset[1]))
        
        surface.blit(title_surface, (title_x, title_y))
    
    def draw_content(self, surface):
        """Draw panel content"""
        if not hasattr(self, 'content_font'):
            self.content_font = pygame.font.Font(None, 18)
        
        y_offset = 35
        for item in self.content[-5:]:  # Show last 5 items
            if item['type'] == 'text':
                text = item['data'].get('text', '')
                color = item['data'].get('color', self.get_theme_color())
                text_surface = self.content_font.render(text, True, color)
                surface.blit(text_surface, (10, y_offset))
                y_offset += 25
            elif item['type'] == 'progress':
                self.draw_progress_bar(surface, 10, y_offset, item['data'])
                y_offset += 30
            elif item['type'] == 'graph':
                self.draw_mini_graph(surface, 10, y_offset, item['data'])
                y_offset += 60
    
    def draw_progress_bar(self, surface, x: int, y: int, data: Dict):
        """Draw animated progress bar"""
        width = self.width - 20
        height = 8
        progress = data.get('value', 0.0)
        color = data.get('color', self.get_theme_color())
        
        # Background
        pygame.draw.rect(surface, (50, 50, 50), (x, y, width, height))
        
        # Progress fill with glow
        fill_width = int(width * progress)
        pygame.draw.rect(surface, color, (x, y, fill_width, height))
        
        # Animated sweep effect
        sweep_pos = int((time.time() * 200) % width)
        if sweep_pos < fill_width:
            pygame.draw.line(surface, (255, 255, 255), 
                           (x + sweep_pos, y), (x + sweep_pos, y + height), 2)
    
    def draw_mini_graph(self, surface, x: int, y: int, data: Dict):
        """Draw mini graph with holographic style"""
        width = self.width - 20
        height = 40
        values = data.get('values', [])
        color = data.get('color', self.get_theme_color())
        
        if len(values) < 2:
            return
        
        # Draw background grid
        for i in range(0, width, 20):
            pygame.draw.line(surface, (30, 30, 30), (x + i, y), (x + i, y + height))
        for i in range(0, height, 10):
            pygame.draw.line(surface, (30, 30, 30), (x, y + i), (x + width, y + i))
        
        # Draw graph line
        points = []
        max_val = max(values) if values else 1
        for i, value in enumerate(values):
            px = x + int(i * width / len(values))
            py = y + height - int((value / max_val) * height)
            points.append((px, py))
        
        if len(points) > 1:
            pygame.draw.lines(surface, color, False, points, 2)
            
            # Add glow effect
            for point in points:
                pygame.draw.circle(surface, color, point, 3)
    
    def draw_scan_lines(self, surface):
        """Draw animated scan lines"""
        if not self.config.enable_animations:
            return
            
        scan_color = (*self.get_theme_color(), 30)
        
        # Vertical scan line
        scan_x = int(self.scan_offset) % self.width
        pygame.draw.line(surface, self.get_theme_color(), 
                        (scan_x, 0), (scan_x, self.height), 1)
        
        # Horizontal scan lines
        for y in range(0, self.height, 4):
            alpha = int(50 * math.sin(y * 0.1 + time.time() * 3))
            if alpha > 0:
                scan_line_color = (*self.get_theme_color(), alpha)
                pygame.draw.line(surface, self.get_theme_color(), 
                               (0, y), (self.width, y))
    
    def get_theme_color(self) -> Tuple[int, int, int]:
        """Get color based on current theme"""
        theme_colors = {
            UITheme.CLASSIC_BLUE: (100, 200, 255),
            UITheme.ARC_REACTOR: (150, 220, 255),
            UITheme.STEALTH: (100, 100, 120),
            UITheme.MARK_85: (255, 215, 100),
            UITheme.FRIDAY: (255, 100, 150)
        }
        return theme_colors.get(self.config.theme, (100, 200, 255))

class JarvisHolographicUI:
    """Main JARVIS Holographic UI System"""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.config = HolographicConfig()
        self.running = True
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | HWSURFACE)
        pygame.display.set_caption("JARVIS - Iron Man AI Interface")
        
        # Initialize components
        self.particle_system = ParticleSystem(self.config)
        self.panels = self.create_default_panels()
        self.clock = pygame.time.Clock()
        
        # UI State
        self.is_listening = False
        self.ai_thinking = False
        self.system_status = "ONLINE"
        self.last_command = ""
        
        print("🤖 JARVIS Holographic Interface initialized")
        print(f"   Resolution: {width}x{height}")
        print(f"   Theme: {self.config.theme.value}")
        print(f"   OpenGL Available: {OPENGL_AVAILABLE}")
    
    def create_default_panels(self) -> List[HolographicPanel]:
        """Create default UI panels"""
        panels = []
        
        # System Status Panel
        status_panel = HolographicPanel(50, 50, 300, 200, "SYSTEM STATUS", self.config)
        status_panel.add_content('text', {'text': f'Status: {self.system_status}', 'color': (0, 255, 0)})
        status_panel.add_content('progress', {'value': 0.95, 'color': (0, 255, 100)})
        panels.append(status_panel)
        
        # AI Assistant Panel
        ai_panel = HolographicPanel(370, 50, 400, 200, "AI ASSISTANT", self.config)
        ai_panel.add_content('text', {'text': 'Ready for commands', 'color': (100, 200, 255)})
        panels.append(ai_panel)
        
        # Environment Analysis Panel
        env_panel = HolographicPanel(790, 50, 350, 200, "ENVIRONMENT", self.config)
        env_panel.add_content('text', {'text': 'Scanning environment...', 'color': (255, 200, 100)})
        panels.append(env_panel)
        
        # System Monitoring Panel
        monitor_panel = HolographicPanel(50, 270, 500, 300, "SYSTEM MONITORING", self.config)
        monitor_panel.add_content('graph', {
            'values': [0.2, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3, 0.6, 0.9, 0.4],
            'color': (255, 100, 100)
        })
        panels.append(monitor_panel)
        
        # Voice Recognition Panel
        voice_panel = HolographicPanel(570, 270, 350, 150, "VOICE RECOGNITION", self.config)
        voice_panel.add_content('text', {'text': 'Listening for wake word...', 'color': (150, 150, 255)})
        panels.append(voice_panel)
        
        return panels
    
    def update_system_status(self, status: str):
        """Update system status"""
        self.system_status = status
        # Update status panel
        if self.panels:
            self.panels[0].content = []
            self.panels[0].add_content('text', {'text': f'Status: {status}', 'color': (0, 255, 0)})
    
    def update_ai_response(self, response: str):
        """Update AI response panel"""
        if len(self.panels) > 1:
            self.panels[1].add_content('text', {'text': response, 'color': (100, 200, 255)})
    
    def update_environment_data(self, data: Dict):
        """Update environment analysis panel"""
        if len(self.panels) > 2:
            for key, value in data.items():
                self.panels[2].add_content('text', {'text': f'{key}: {value}', 'color': (255, 200, 100)})
    
    def update_monitoring_data(self, data: List[float]):
        """Update system monitoring graph"""
        if len(self.panels) > 3:
            self.panels[3].add_content('graph', {'values': data, 'color': (255, 100, 100)})
    
    def set_listening_state(self, listening: bool):
        """Set voice recognition listening state"""
        self.is_listening = listening
        if len(self.panels) > 4:
            status_text = "LISTENING..." if listening else "Waiting for wake word..."
            color = (0, 255, 0) if listening else (150, 150, 255)
            self.panels[4].content = []
            self.panels[4].add_content('text', {'text': status_text, 'color': color})
    
    def run(self):
        """Main UI loop"""
        dt = 0
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
                    elif event.key == K_1:
                        self.config.theme = UITheme.CLASSIC_BLUE
                    elif event.key == K_2:
                        self.config.theme = UITheme.ARC_REACTOR
                    elif event.key == K_3:
                        self.config.theme = UITheme.STEALTH
                    elif event.key == K_4:
                        self.config.theme = UITheme.MARK_85
                    elif event.key == K_5:
                        self.config.theme = UITheme.FRIDAY
            
            # Clear screen
            self.screen.fill((0, 0, 0))
            
            # Update components
            if self.config.enable_particles:
                self.particle_system.update(dt)
            
            for panel in self.panels:
                panel.update(dt)
            
            # Render components
            if self.config.enable_particles:
                self.particle_system.render(self.screen)
            
            for panel in self.panels:
                panel.render(self.screen)
            
            # Draw central arc reactor effect
            self.draw_arc_reactor()
            
            # Update display
            pygame.display.flip()
            dt = self.clock.tick(60) / 1000.0  # 60 FPS
        
        pygame.quit()
    
    def draw_arc_reactor(self):
        """Draw central arc reactor animation"""
        center_x = self.width // 2
        center_y = self.height - 100
        
        # Get theme color
        theme_colors = {
            UITheme.CLASSIC_BLUE: (100, 200, 255),
            UITheme.ARC_REACTOR: (150, 220, 255),
            UITheme.STEALTH: (100, 100, 120),
            UITheme.MARK_85: (255, 215, 100),
            UITheme.FRIDAY: (255, 100, 150)
        }
        color = theme_colors.get(self.config.theme, (150, 220, 255))
        
        # Animated arc reactor
        time_factor = time.time() * 2
        base_radius = 30
        
        # Outer ring
        for i in range(8):
            angle = (i * math.pi / 4) + time_factor
            radius = base_radius + 10 * math.sin(time_factor + i)
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            
            pygame.draw.circle(self.screen, color, (x, y), 3)
        
        # Inner glow
        for radius in range(5, 25, 3):
            alpha = int(255 * (1 - radius / 25) * 0.5)
            glow_color = (*color, alpha)
            
            # Create temporary surface for alpha blending
            temp_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, (*color, alpha), (radius, radius), radius)
            self.screen.blit(temp_surface, (center_x - radius, center_y - radius), 
                           special_flags=pygame.BLEND_ADD)

def main():
    """Test the holographic UI system"""
    ui = JarvisHolographicUI()
    
    # Simulate some data updates
    import threading
    import random
    
    def simulate_data():
        while ui.running:
            time.sleep(2)
            
            # Update monitoring data
            data = [random.random() for _ in range(10)]
            ui.update_monitoring_data(data)
            
            # Update environment data
            env_data = {
                'Temperature': f'{random.randint(20, 25)}°C',
                'Humidity': f'{random.randint(40, 60)}%',
                'Objects': random.randint(0, 5)
            }
            ui.update_environment_data(env_data)
            
            # Simulate voice recognition
            if random.random() > 0.7:
                ui.set_listening_state(True)
                time.sleep(1)
                ui.set_listening_state(False)
                ui.update_ai_response("Command processed successfully")
    
    # Start simulation thread
    sim_thread = threading.Thread(target=simulate_data, daemon=True)
    sim_thread.start()
    
    # Run UI
    ui.run()

if __name__ == "__main__":
    main()