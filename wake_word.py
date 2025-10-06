import threading
import time
import logging
from typing import Optional, Callable
import numpy as np

try:
    import openwakeword
    from openwakeword import Model
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False

try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False


class WakeWordDetector:
    """Wake word detection with multiple backends"""
    
    def __init__(self, backend="openwakeword", wake_word="jarvis", callback=None):
        self.backend = backend
        self.wake_word = wake_word
        self.callback = callback
        self.running = False
        self.thread = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize backend
        self.detector = None
        if backend == "openwakeword" and OPENWAKEWORD_AVAILABLE:
            self._init_openwakeword()
        elif backend == "porcupine" and PORCUPINE_AVAILABLE:
            self._init_porcupine()
        else:
            self.logger.warning(f"Wake word backend '{backend}' not available")
    
    def _init_openwakeword(self):
        """Initialize OpenWakeWord"""
        try:
            # Use built-in models or custom ones
            model_paths = ["alexa", "hey_jarvis"]  # Built-in models
            self.detector = Model(wakeword_models=model_paths, inference_framework="onnx")
            self.logger.info("OpenWakeWord initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenWakeWord: {e}")
            self.detector = None
    
    def _init_porcupine(self):
        """Initialize Porcupine (requires license key)"""
        try:
            # Note: Porcupine requires an access key for commercial use
            # For demo purposes, this is a placeholder
            keywords = ["jarvis"]  # Built-in keywords
            self.detector = pvporcupine.create(keywords=keywords)
            self.logger.info("Porcupine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Porcupine: {e}")
            self.detector = None
    
    def start(self):
        """Start wake word detection in background thread"""
        if not self.detector or self.running:
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        self.logger.info("Wake word detection started")
        return True
    
    def stop(self):
        """Stop wake word detection"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.logger.info("Wake word detection stopped")
    
    def _detection_loop(self):
        """Main detection loop (placeholder - needs audio input)"""
        # This is a simplified version. In practice, you'd need:
        # 1. Audio capture from microphone
        # 2. Audio preprocessing (16kHz, 16-bit PCM)
        # 3. Feed audio frames to detector
        # 4. Handle detection results
        
        while self.running:
            try:
                # Placeholder: simulate audio processing
                # In real implementation, capture audio frames here
                time.sleep(0.1)
                
                # Simulate detection (for demo)
                # detected = self.detector.predict(audio_frame)
                # if detected and self.callback:
                #     self.callback(self.wake_word)
                
            except Exception as e:
                self.logger.error(f"Wake word detection error: {e}")
                time.sleep(1.0)
    
    def process_audio(self, audio_data: np.ndarray) -> bool:
        """Process audio data and return True if wake word detected"""
        if not self.detector:
            return False
        
        try:
            if self.backend == "openwakeword":
                # OpenWakeWord expects specific format
                prediction = self.detector.predict(audio_data)
                # Check if any wake word was detected
                for mdl in self.detector.prediction_buffer.keys():
                    if prediction[mdl] > 0.5:  # Threshold
                        return True
            elif self.backend == "porcupine":
                # Porcupine returns keyword index or -1
                keyword_index = self.detector.process(audio_data)
                if keyword_index >= 0:
                    return True
        except Exception as e:
            self.logger.error(f"Audio processing error: {e}")
        
        return False


class MockWakeWordDetector:
    """Mock detector for testing when backends unavailable"""
    
    def __init__(self, wake_word="jarvis", callback=None):
        self.wake_word = wake_word
        self.callback = callback
        self.running = False
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        self.running = True
        self.logger.info("Mock wake word detector started")
        return True
    
    def stop(self):
        self.running = False
        self.logger.info("Mock wake word detector stopped")
    
    def process_audio(self, audio_data: np.ndarray) -> bool:
        # Always return False for mock
        return False


def create_wake_word_detector(backend="openwakeword", wake_word="jarvis", callback=None):
    """Factory function to create appropriate wake word detector"""
    if backend == "openwakeword" and OPENWAKEWORD_AVAILABLE:
        return WakeWordDetector(backend, wake_word, callback)
    elif backend == "porcupine" and PORCUPINE_AVAILABLE:
        return WakeWordDetector(backend, wake_word, callback)
    else:
        logging.getLogger(__name__).warning(f"Using mock wake word detector (backend '{backend}' unavailable)")
        return MockWakeWordDetector(wake_word, callback)
