"""
Enhanced UI Theme and Constants for Vision Jarvis
Provides modern, responsive UI styling with dark theme and animations
"""

import cv2
import numpy as np
import time
from typing import Tuple, Dict, Any


class UITheme:
    """Modern UI theme with dark styling and smooth animations"""

    # Color palette (BGR format for OpenCV)
    COLORS = {
        # Primary colors
        "primary": (255, 100, 50),  # Electric blue
        "primary_dark": (200, 80, 40),  # Darker blue
        "primary_light": (255, 150, 100),  # Lighter blue
        # Accent colors
        "accent": (0, 200, 255),  # Orange
        "accent_hover": (0, 150, 200),  # Darker orange
        # Status colors
        "success": (0, 255, 100),  # Green
        "warning": (0, 255, 255),  # Yellow
        "error": (0, 100, 255),  # Red
        "info": (255, 200, 0),  # Cyan
        # UI elements
        "background": (30, 30, 30),  # Dark gray
        "surface": (50, 50, 50),  # Medium gray
        "surface_light": (70, 70, 70),  # Light gray
        "border": (100, 100, 100),  # Border gray
        # Text colors
        "text_primary": (255, 255, 255),  # White
        "text_secondary": (200, 200, 200),  # Light gray
        "text_disabled": (120, 120, 120),  # Medium gray
        # Module specific colors
        "voice": (255, 100, 50),  # Blue for voice
        "camera": (0, 200, 100),  # Green for camera
        "canvas": (100, 100, 255),  # Red for canvas
        # Special effects
        "glow": (255, 255, 255),  # White glow
        "shadow": (0, 0, 0),  # Black shadow
    }

    # Typography
    FONTS = {
        "title": cv2.FONT_HERSHEY_DUPLEX,
        "subtitle": cv2.FONT_HERSHEY_SIMPLEX,
        "body": cv2.FONT_HERSHEY_SIMPLEX,
        "caption": cv2.FONT_HERSHEY_PLAIN,
        "mono": cv2.FONT_HERSHEY_PLAIN,
    }

    FONT_SCALES = {
        "title": 1.2,
        "subtitle": 0.9,
        "body": 0.7,
        "caption": 0.6,
        "mono": 0.5,
    }

    FONT_THICKNESS = {
        "title": 2,
        "subtitle": 2,
        "body": 1,
        "caption": 1,
        "mono": 1,
    }

    # Spacing and sizing
    SPACING = {
        "xs": 4,
        "sm": 8,
        "md": 16,
        "lg": 24,
        "xl": 32,
        "xxl": 48,
    }

    BORDER_RADIUS = {
        "sm": 4,
        "md": 8,
        "lg": 12,
        "xl": 16,
    }

    # Animation settings
    ANIMATION = {
        "duration_fast": 0.2,
        "duration_normal": 0.3,
        "duration_slow": 0.5,
        "ease_in_out": lambda t: t * t * (3.0 - 2.0 * t),  # Smooth easing
        "bounce": lambda t: 1 - abs(np.sin(t * np.pi)),  # Bounce effect
    }


class UIAnimator:
    """Handles smooth animations and transitions"""

    def __init__(self):
        self.animations = {}

    def start_animation(
        self,
        name: str,
        duration: float,
        start_value: float = 0.0,
        end_value: float = 1.0,
    ):
        """Start a new animation"""
        self.animations[name] = {
            "start_time": time.time(),
            "duration": duration,
            "start_value": start_value,
            "end_value": end_value,
        }

    def get_animation_value(self, name: str, easing_func=None) -> float:
        """Get current animation value (0.0 to 1.0)"""
        if name not in self.animations:
            return 0.0

        anim = self.animations[name]
        elapsed = time.time() - anim["start_time"]
        progress = min(1.0, elapsed / anim["duration"])

        if easing_func:
            progress = easing_func(progress)

        value = (
            anim["start_value"] + (anim["end_value"] - anim["start_value"]) * progress
        )

        # Clean up completed animations
        if progress >= 1.0:
            del self.animations[name]

        return value

    def is_animating(self, name: str) -> bool:
        """Check if animation is still running"""
        return name in self.animations


class UIRenderer:
    """Advanced UI rendering utilities"""

    @staticmethod
    def draw_rounded_rect(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        radius: int = 8,
        thickness: int = -1,
    ):
        """Draw a rounded rectangle"""
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        radius = int(radius)

        # Ensure coordinates are in correct order
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Ensure color values are integers
        color = tuple(int(c) for c in color)

        # Ensure radius doesn't exceed rectangle dimensions
        max_radius = min((x2 - x1) // 2, (y2 - y1) // 2)
        radius = min(radius, max_radius)

        if radius <= 0:
            # Fall back to regular rectangle if radius is too small
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            return

        # Draw main rectangle parts
        if thickness == -1:  # Filled
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        else:  # Outlined
            # Top and bottom edges
            cv2.rectangle(
                img, (x1 + radius, y1), (x2 - radius, y1 + thickness), color, -1
            )
            cv2.rectangle(
                img, (x1 + radius, y2 - thickness), (x2 - radius, y2), color, -1
            )
            # Left and right edges
            cv2.rectangle(
                img, (x1, y1 + radius), (x1 + thickness, y2 - radius), color, -1
            )
            cv2.rectangle(
                img, (x2 - thickness, y1 + radius), (x2, y2 - radius), color, -1
            )

        # Draw corner circles
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

    @staticmethod
    def draw_gradient_rect(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
    ):
        """Draw a gradient rectangle"""
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])

        # Ensure colors are integers
        color1 = tuple(int(c) for c in color1)
        color2 = tuple(int(c) for c in color2)

        for y in range(y1, y2):
            ratio = (y - y1) / (y2 - y1) if y2 != y1 else 0
            color = tuple(int(c1 + (c2 - c1) * ratio) for c1, c2 in zip(color1, color2))
            cv2.line(img, (x1, y), (x2, y), color, 1)

    @staticmethod
    def draw_glow_effect(
        img: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int],
        intensity: float = 0.5,
    ):
        """Draw a glowing effect around a point"""
        center_x, center_y = int(center[0]), int(center[1])
        radius = int(radius)
        color = tuple(int(c) for c in color)

        overlay = img.copy()

        # Create multiple circles with decreasing opacity
        for i in range(5):
            alpha = intensity * (1 - i * 0.2)
            current_radius = radius + i * 10
            cv2.circle(overlay, (center_x, center_y), current_radius, color, -1)
            cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0, img)

    @staticmethod
    def draw_text_with_shadow(
        img: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font: int,
        scale: float,
        color: Tuple[int, int, int],
        thickness: int = 1,
        shadow_offset: Tuple[int, int] = (2, 2),
    ):
        """Draw text with shadow effect"""
        x, y = int(position[0]), int(position[1])
        shadow_x, shadow_y = int(shadow_offset[0]), int(shadow_offset[1])
        color = tuple(int(c) for c in color)

        # Draw shadow
        cv2.putText(
            img,
            text,
            (x + shadow_x, y + shadow_y),
            font,
            scale,
            theme.COLORS["shadow"],
            thickness,
        )

        # Draw main text
        cv2.putText(img, text, (x, y), font, scale, color, thickness)

    @staticmethod
    def draw_progress_bar(
        img: np.ndarray,
        position: Tuple[int, int],
        size: Tuple[int, int],
        progress: float,
        color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int] = None,
        rounded: bool = True,
    ):
        """Draw an animated progress bar"""
        x, y = int(position[0]), int(position[1])
        width, height = int(size[0]), int(size[1])
        color = tuple(int(c) for c in color)

        if bg_color is None:
            bg_color = theme.COLORS["surface"]
        else:
            bg_color = tuple(int(c) for c in bg_color)

        # Background
        if rounded:
            UIRenderer.draw_rounded_rect(
                img, (x, y), (x + width, y + height), bg_color, height // 2
            )
        else:
            cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)

        # Progress fill
        fill_width = int(width * progress)
        if fill_width > 0:
            if rounded:
                UIRenderer.draw_rounded_rect(
                    img, (x, y), (x + fill_width, y + height), color, height // 2
                )
            else:
                cv2.rectangle(img, (x, y), (x + fill_width, y + height), color, -1)


class ModernButton:
    """A modern, interactive button with hover and click animations"""

    def __init__(
        self,
        text: str,
        position: Tuple[int, int],
        size: Tuple[int, int],
        color: Tuple[int, int, int] = None,
        text_color: Tuple[int, int, int] = None,
    ):
        self.text = text
        self.position = (int(position[0]), int(position[1]))
        self.size = (int(size[0]), int(size[1]))
        self.color = tuple(int(c) for c in color) if color else theme.COLORS["primary"]
        self.text_color = (
            tuple(int(c) for c in text_color)
            if text_color
            else theme.COLORS["text_primary"]
        )
        self.is_hovered = False
        self.is_pressed = False
        self.hover_start_time = None
        self.animator = UIAnimator()

    def update_hover(self, mouse_pos: Tuple[int, int]):
        """Update hover state based on mouse position"""
        x, y = self.position
        w, h = self.size
        mx, my = int(mouse_pos[0]), int(mouse_pos[1])

        was_hovered = self.is_hovered
        self.is_hovered = x <= mx <= x + w and y <= my <= y + h

        if self.is_hovered and not was_hovered:
            self.hover_start_time = time.time()
            self.animator.start_animation("hover", theme.ANIMATION["duration_fast"])
        elif not self.is_hovered and was_hovered:
            self.animator.start_animation("hover_out", theme.ANIMATION["duration_fast"])

    def draw(self, img: np.ndarray):
        """Draw the button with current state"""
        x, y = int(self.position[0]), int(self.position[1])
        w, h = int(self.size[0]), int(self.size[1])

        # Calculate hover animation
        hover_progress = 0.0
        if self.animator.is_animating("hover"):
            hover_progress = self.animator.get_animation_value(
                "hover", theme.ANIMATION["ease_in_out"]
            )
        elif self.animator.is_animating("hover_out"):
            hover_progress = 1.0 - self.animator.get_animation_value(
                "hover_out", theme.ANIMATION["ease_in_out"]
            )
        elif self.is_hovered:
            hover_progress = 1.0

        # Interpolate colors based on hover state
        current_color = tuple(
            int(c + (theme.COLORS["accent"][i] - c) * hover_progress * 0.3)
            for i, c in enumerate(self.color)
        )

        # Draw button background
        UIRenderer.draw_rounded_rect(
            img, (x, y), (x + w, y + h), current_color, theme.BORDER_RADIUS["md"]
        )

        # Draw glow effect when hovered
        if hover_progress > 0:
            center = (x + w // 2, y + h // 2)
            UIRenderer.draw_glow_effect(
                img, center, max(w, h) // 2, theme.COLORS["glow"], hover_progress * 0.2
            )

        # Draw text
        text_size = cv2.getTextSize(
            self.text,
            theme.FONTS["body"],
            theme.FONT_SCALES["body"],
            theme.FONT_THICKNESS["body"],
        )[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2

        UIRenderer.draw_text_with_shadow(
            img,
            self.text,
            (text_x, text_y),
            theme.FONTS["body"],
            theme.FONT_SCALES["body"],
            self.text_color,
            theme.FONT_THICKNESS["body"],
        )


# Global theme instance
theme = UITheme()
