import logging
import time
from typing import Optional, Tuple
import numpy as np

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


class ASRBackend:
    """Base class for ASR backends"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def transcribe(self, audio_data) -> Tuple[Optional[str], str]:
        """Transcribe audio and return (text, status)"""
        raise NotImplementedError


class SpeechRecognitionBackend(ASRBackend):
    """Google Speech Recognition backend (original)"""
    
    def __init__(self, config=None):
        super().__init__()
        if not SR_AVAILABLE:
            raise ImportError("speech_recognition not available")
        
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.config = config or {}
        self._calibrate_microphone()
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.logger.info("Microphone calibrated for ambient noise")
        except Exception as e:
            self.logger.warning(f"Microphone calibration failed: {e}")
    
    def transcribe(self, audio_data=None) -> Tuple[Optional[str], str]:
        """Transcribe from microphone (ignores audio_data parameter)"""
        retry_attempts = self.config.get("retry_attempts", 3)
        speech_timeout = self.config.get("speech_timeout", 5.0)
        phrase_time_limit = self.config.get("phrase_time_limit", 10.0)
        retry_delay = self.config.get("retry_delay", 1.0)
        
        for attempt in range(retry_attempts):
            try:
                with self.mic as source:
                    self.logger.debug(f"Listening attempt {attempt + 1}")
                    audio = self.recognizer.listen(
                        source,
                        timeout=speech_timeout,
                        phrase_time_limit=phrase_time_limit,
                    )
                    
                    result = self.recognizer.recognize_google(audio)
                    self.logger.info(f"Speech recognized: {result}")
                    return result, "success"
            
            except sr.WaitTimeoutError:
                error_msg = "No speech detected within timeout"
                self.logger.warning(f"{error_msg} (attempt {attempt + 1})")
                if attempt == retry_attempts - 1:
                    return None, error_msg
            
            except sr.UnknownValueError:
                error_msg = "Could not understand speech"
                self.logger.warning(f"{error_msg} (attempt {attempt + 1})")
                if attempt == retry_attempts - 1:
                    return None, error_msg
            
            except sr.RequestError as e:
                error_msg = f"Speech service error: {e}"
                self.logger.error(f"{error_msg} (attempt {attempt + 1})")
                if attempt == retry_attempts - 1:
                    return None, error_msg
            
            # Wait before retry
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
        
        return None, "Max retry attempts reached"


class FasterWhisperBackend(ASRBackend):
    """Faster-Whisper local ASR backend"""
    
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        super().__init__()
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError("faster-whisper not available")
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            self.logger.info("Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            self.model = None
    
    def transcribe(self, audio_data) -> Tuple[Optional[str], str]:
        """Transcribe audio data (numpy array or file path)"""
        if not self.model:
            return None, "Model not loaded"
        
        try:
            # If audio_data is a file path
            if isinstance(audio_data, str):
                segments, info = self.model.transcribe(audio_data)
            else:
                # If audio_data is numpy array, save to temp file first
                import tempfile
                try:
                    import soundfile as sf
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, audio_data, 16000)
                        segments, info = self.model.transcribe(tmp.name)
                except ImportError:
                    # Fallback: use scipy.io.wavfile if soundfile not available
                    try:
                        from scipy.io import wavfile
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            wavfile.write(tmp.name, 16000, audio_data.astype(np.int16))
                            segments, info = self.model.transcribe(tmp.name)
                    except ImportError:
                        raise ImportError("Neither soundfile nor scipy is available for audio file writing")
            
            # Combine all segments
            text = " ".join([segment.text for segment in segments]).strip()
            
            if text:
                self.logger.info(f"Whisper transcribed: {text}")
                return text, "success"
            else:
                return None, "No speech detected"
        
        except Exception as e:
            self.logger.error(f"Whisper transcription error: {e}")
            return None, f"Transcription error: {e}"


class ASRManager:
    """Manages different ASR backends"""
    
    def __init__(self, backend="speech_recognition", config=None):
        self.backend_name = backend
        self.config = config or {}
        self.backend = None
        self.logger = logging.getLogger(__name__)
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected backend"""
        try:
            if self.backend_name == "speech_recognition":
                self.backend = SpeechRecognitionBackend(self.config)
            elif self.backend_name == "faster_whisper":
                model_size = self.config.get("whisper_model_size", "base")
                device = self.config.get("whisper_device", "cpu")
                compute_type = self.config.get("whisper_compute_type", "int8")
                self.backend = FasterWhisperBackend(model_size, device, compute_type)
            else:
                raise ValueError(f"Unknown ASR backend: {self.backend_name}")
            
            self.logger.info(f"ASR backend '{self.backend_name}' initialized")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize ASR backend '{self.backend_name}': {e}")
            # Fallback to speech_recognition if available
            if self.backend_name != "speech_recognition" and SR_AVAILABLE:
                self.logger.info("Falling back to speech_recognition backend")
                self.backend = SpeechRecognitionBackend(self.config)
            else:
                self.backend = None
    
    def transcribe(self, audio_data=None) -> Tuple[Optional[str], str]:
        """Transcribe audio using the configured backend"""
        if not self.backend:
            return None, "No ASR backend available"
        
        return self.backend.transcribe(audio_data)
    
    def is_available(self) -> bool:
        """Check if backend is available"""
        return self.backend is not None


def create_asr_manager(backend="speech_recognition", config=None):
    """Factory function to create ASR manager"""
    return ASRManager(backend, config)
