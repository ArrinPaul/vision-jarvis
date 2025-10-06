from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image


class ImageExporter:
    """Export utility for saving images in additional formats without touching UI/gestures.

    Usage:
        webp_path = ImageExporter.export_webp(image_bgr, src_jpg_path, quality=80, method=4)
    """

    @staticmethod
    def _derive_webp_path(src_jpg_path: str) -> str:
        base, _ = os.path.splitext(src_jpg_path)
        return base + ".webp"

    @staticmethod
    def export_webp(
        image_bgr: np.ndarray,
        src_jpg_path: str,
        quality: int = 80,
        method: int = 4,
        lossless: bool = False,
        optimize: bool = True,
    ) -> str:
        """Save a WebP version of the provided OpenCV BGR image next to the JPG.

        Parameters:
            image_bgr: np.ndarray - OpenCV image in BGR color order
            src_jpg_path: str - Path of the already-saved JPG (used to derive .webp path)
            quality: 0..100 - Compression quality (higher = better quality, larger file)
            method: 0..6 - Encoding speed/quality tradeoff (6 is slowest/best)
            lossless: bool - Use lossless WebP (quality parameter is ignored if True)
            optimize: bool - Let Pillow optimize encoding when possible

        Returns:
            str: Path to the saved .webp file
        """
        if image_bgr is None or not isinstance(image_bgr, np.ndarray):
            raise ValueError("image_bgr must be a numpy ndarray (OpenCV image)")
        if image_bgr.ndim != 3 or image_bgr.shape[2] not in (3, 4):
            raise ValueError("image_bgr must be HxWx3 or HxWx4 array")

        # Convert BGR(A) -> RGB(A) for Pillow
        if image_bgr.shape[2] == 3:
            image_rgb = image_bgr[:, :, ::-1]
            pil_img = Image.fromarray(image_rgb)
        else:  # 4 channels
            # Preserve alpha if present; Pillow expects RGBA
            image_rgba = image_bgr[:, :, [2, 1, 0, 3]]
            pil_img = Image.fromarray(image_rgba)

        webp_path = ImageExporter._derive_webp_path(src_jpg_path)

        save_kwargs = {
            "format": "WEBP",
            "lossless": lossless,
            "method": max(0, min(6, int(method))),
        }
        if not lossless:
            save_kwargs["quality"] = max(0, min(100, int(quality)))
        if optimize:
            save_kwargs["optimize"] = True

        # Ensure destination directory exists
        os.makedirs(os.path.dirname(webp_path) or ".", exist_ok=True)

        pil_img.save(webp_path, **save_kwargs)
        return webp_path

