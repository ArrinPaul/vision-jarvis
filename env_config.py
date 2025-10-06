#!/usr/bin/env python3
"""
Environment Configuration Manager for Jarvis AI Assistant

This module provides a centralized way to manage environment variables
and configuration settings with proper defaults and validation.
"""

import os
import logging
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import json

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class EnvironmentConfig:
    """Centralized environment configuration manager"""

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize environment configuration

        Args:
            env_file: Path to .env file (defaults to .env in current directory)
        """
        self.logger = logging.getLogger(__name__)

        # Load environment variables
        if DOTENV_AVAILABLE:
            env_path = env_file or ".env"
            if os.path.exists(env_path):
                load_dotenv(env_path)
                self.logger.info(f"Loaded environment from {env_path}")
            else:
                self.logger.warning(f"Environment file {env_path} not found")
        else:
            self.logger.warning("python-dotenv not available, using system environment only")

        # Initialize configuration cache
        self._config_cache = {}
        self._load_configuration()

    def _load_configuration(self):
        """Load and validate all configuration settings"""
        self._config_cache = {
            # API Keys and Credentials
            'api_keys': {
                'google_api_key': self.get_str('GOOGLE_API_KEY') or self.get_str('GEMINI_API_KEY'),
                'openai_api_key': self.get_str('OPENAI_API_KEY'),
                'porcupine_access_key': self.get_str('PORCUPINE_ACCESS_KEY'),
            },

            # Voice Assistant Configuration
            'voice_assistant': {
                'persona_name': self.get_str('JARVIS_PERSONA_NAME', 'Jarvis'),
                'wake_word': self.get_str('JARVIS_WAKE_WORD', 'jarvis'),
                'require_wake_word': self.get_bool('JARVIS_REQUIRE_WAKE_WORD', True),
                'asr_backend': self.get_str('ASR_BACKEND', 'faster_whisper'),
                'whisper_model_size': self.get_str('WHISPER_MODEL_SIZE', 'base'),
                'whisper_device': self.get_str('WHISPER_DEVICE', 'cpu'),
                'whisper_compute_type': self.get_str('WHISPER_COMPUTE_TYPE', 'int8'),
                'tts_backend': self.get_str('TTS_BACKEND', 'edge_tts'),
                'tts_voice': self.get_str('TTS_VOICE', 'en-US-GuyNeural'),
                'tts_rate': self.get_str('TTS_RATE', '+0%'),
                'tts_pitch': self.get_str('TTS_PITCH', '+0Hz'),
                'wake_word_backend': self.get_str('WAKE_WORD_BACKEND', 'openwakeword'),
            },

            # Vision and AI Models
            'vision': {
                'model_path': self.get_str('VISION_MODEL_PATH', 'models/yolov8n.onnx'),
                'confidence_threshold': self.get_float('VISION_CONFIDENCE_THRESHOLD', 0.5),
                'enable_object_detection': self.get_bool('ENABLE_OBJECT_DETECTION', True),
                'enable_sketch_recognition': self.get_bool('ENABLE_SKETCH_RECOGNITION', False),
                'sketch_model_path': self.get_str('SKETCH_MODEL_PATH', 'models/sketch_classifier.onnx'),
            },

            # System Configuration
            'system': {
                'camera_index': self.get_int('CAMERA_INDEX', 0),
                'camera_width': self.get_int('CAMERA_WIDTH', 1280),
                'camera_height': self.get_int('CAMERA_HEIGHT', 720),
                'camera_fps': self.get_int('CAMERA_FPS', 30),
                'max_hands': self.get_int('MAX_HANDS', 1),
                'hand_detection_confidence': self.get_float('HAND_DETECTION_CONFIDENCE', 0.8),
                'hand_tracking_confidence': self.get_float('HAND_TRACKING_CONFIDENCE', 0.7),
                'pinch_threshold': self.get_float('PINCH_THRESHOLD', 0.04),
                'point_threshold': self.get_float('POINT_THRESHOLD', 0.03),
                'hover_time': self.get_float('HOVER_TIME', 2.0),
            },

            # Performance Settings
            'performance': {
                'max_memory_entries': self.get_int('MAX_MEMORY_ENTRIES', 100),
                'tts_cache_size': self.get_int('TTS_CACHE_SIZE', 50),
                'max_audio_queue': self.get_int('MAX_AUDIO_QUEUE', 10),
                'enable_gpu_acceleration': self.get_bool('ENABLE_GPU_ACCELERATION', False),
                'model_quantization': self.get_bool('MODEL_QUANTIZATION', True),
                'batch_processing': self.get_bool('BATCH_PROCESSING', False),
            },

            # Logging and Debugging
            'logging': {
                'log_level': self.get_str('LOG_LEVEL', 'INFO'),
                'enable_gesture_logging': self.get_bool('ENABLE_GESTURE_LOGGING', False),
                'enable_performance_metrics': self.get_bool('ENABLE_PERFORMANCE_METRICS', True),
                'enable_debug_mode': self.get_bool('ENABLE_DEBUG_MODE', False),
                'log_dir': self.get_str('LOG_DIR', 'logs'),
                'gesture_log_file': self.get_str('GESTURE_LOG_FILE', 'gesture_events.jsonl'),
                'performance_log_file': self.get_str('PERFORMANCE_LOG_FILE', 'performance_metrics.json'),
            },

            # Feature Flags
            'features': {
                'enable_voice_assistant': self.get_bool('ENABLE_VOICE_ASSISTANT', True),
                'enable_camera_module': self.get_bool('ENABLE_CAMERA_MODULE', True),
                'enable_canvas_module': self.get_bool('ENABLE_CANVAS_MODULE', True),
                'enable_context_monitoring': self.get_bool('ENABLE_CONTEXT_MONITORING', True),
                'enable_memory_persistence': self.get_bool('ENABLE_MEMORY_PERSISTENCE', True),
                'enable_system_control': self.get_bool('ENABLE_SYSTEM_CONTROL', True),
                'enable_web_integration': self.get_bool('ENABLE_WEB_INTEGRATION', True),
                'enable_continuous_listening': self.get_bool('ENABLE_CONTINUOUS_LISTENING', False),
                'enable_gesture_prediction': self.get_bool('ENABLE_GESTURE_PREDICTION', False),
                'enable_emotion_detection': self.get_bool('ENABLE_EMOTION_DETECTION', False),
            },

            # Security Settings
            'security': {
                'api_rate_limit': self.get_int('API_RATE_LIMIT', 60),
                'api_timeout': self.get_int('API_TIMEOUT', 30),
                'enable_telemetry': self.get_bool('ENABLE_TELEMETRY', False),
                'save_audio_recordings': self.get_bool('SAVE_AUDIO_RECORDINGS', False),
                'save_video_recordings': self.get_bool('SAVE_VIDEO_RECORDINGS', False),
            },

            # Platform Specific Settings
            'platform': {
                'enable_sapi_tts': self.get_bool('ENABLE_SAPI_TTS', True),
                'enable_windows_notifications': self.get_bool('ENABLE_WINDOWS_NOTIFICATIONS', True),
                'enable_pulseaudio': self.get_bool('ENABLE_PULSEAUDIO', False),
                'enable_alsa': self.get_bool('ENABLE_ALSA', True),
                'enable_coreaudio': self.get_bool('ENABLE_COREAUDIO', False),
                'enable_macos_notifications': self.get_bool('ENABLE_MACOS_NOTIFICATIONS', False),
            },

            # Development Settings
            'development': {
                'dev_mode': self.get_bool('DEV_MODE', False),
                'mock_hardware': self.get_bool('MOCK_HARDWARE', False),
                'enable_hot_reload': self.get_bool('ENABLE_HOT_RELOAD', False),
                'run_health_checks': self.get_bool('RUN_HEALTH_CHECKS', True),
                'enable_unit_tests': self.get_bool('ENABLE_UNIT_TESTS', False),
                'enable_integration_tests': self.get_bool('ENABLE_INTEGRATION_TESTS', False),
                'enable_profiling': self.get_bool('ENABLE_PROFILING', False),
                'profile_output_dir': self.get_str('PROFILE_OUTPUT_DIR', 'profiles'),
            }
        }

    def get_str(self, key: str, default: str = None) -> Optional[str]:
        """Get string environment variable"""
        value = os.getenv(key, default)
        return value if value != "" else default

    def get_int(self, key: str, default: int = None) -> Optional[int]:
        """Get integer environment variable"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            self.logger.warning(f"Invalid integer value for {key}: {value}, using default: {default}")
            return default

    def get_float(self, key: str, default: float = None) -> Optional[float]:
        """Get float environment variable"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            self.logger.warning(f"Invalid float value for {key}: {value}, using default: {default}")
            return default

    def get_bool(self, key: str, default: bool = None) -> Optional[bool]:
        """Get boolean environment variable"""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')

    def get_list(self, key: str, default: List[str] = None, separator: str = ',') -> Optional[List[str]]:
        """Get list environment variable"""
        value = os.getenv(key)
        if value is None:
            return default or []
        return [item.strip() for item in value.split(separator) if item.strip()]

    def get_config(self, section: str = None) -> Union[Dict[str, Any], Any]:
        """
        Get configuration section or entire configuration

        Args:
            section: Configuration section name (e.g., 'voice_assistant', 'vision')
                    If None, returns entire configuration

        Returns:
            Configuration dictionary or specific section
        """
        if section is None:
            return self._config_cache
        return self._config_cache.get(section, {})

    def validate_configuration(self) -> Dict[str, List[str]]:
        """
        Validate configuration and return any issues

        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        issues = {'errors': [], 'warnings': []}

        # Check required API keys
        api_keys = self.get_config('api_keys')
        if not api_keys.get('google_api_key'):
            issues['warnings'].append("Google API key not set - LLM features will be limited")

        # Check model paths
        vision_config = self.get_config('vision')
        if vision_config.get('enable_object_detection'):
            model_path = vision_config.get('model_path')
            if model_path and not os.path.exists(model_path):
                issues['warnings'].append(f"Vision model not found: {model_path}")

        # Check log directory
        log_config = self.get_config('logging')
        log_dir = log_config.get('log_dir')
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        return issues

    def export_config_json(self, output_path: str = "current_config.json"):
        """Export current configuration to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self._config_cache, f, indent=2, default=str)
            self.logger.info(f"Configuration exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration"""
        return self.get_config(key)

    def __contains__(self, key: str) -> bool:
        """Check if configuration section exists"""
        return key in self._config_cache


# Global configuration instance
_global_config = None


def get_config(env_file: Optional[str] = None) -> EnvironmentConfig:
    """
    Get global configuration instance

    Args:
        env_file: Path to .env file (only used on first call)

    Returns:
        EnvironmentConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = EnvironmentConfig(env_file)
    return _global_config


def reload_config(env_file: Optional[str] = None) -> EnvironmentConfig:
    """
    Reload configuration (useful for development)

    Args:
        env_file: Path to .env file

    Returns:
        New EnvironmentConfig instance
    """
    global _global_config
    _global_config = EnvironmentConfig(env_file)
    return _global_config


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()

    print("=== Jarvis Configuration ===")
    print(f"Voice Assistant: {config.get_config('voice_assistant')}")
    print(f"Vision: {config.get_config('vision')}")
    print(f"Features: {config.get_config('features')}")

    # Validate configuration
    issues = config.validate_configuration()
    if issues['errors']:
        print(f"\nErrors: {issues['errors']}")
    if issues['warnings']:
        print(f"\nWarnings: {issues['warnings']}")

    # Export configuration
    config.export_config_json()