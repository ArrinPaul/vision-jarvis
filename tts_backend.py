import asyncio
import logging
import tempfile
import time
from typing import Optional
import threading

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


class TTSBackend:
    """Base class for TTS backends"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def speak(self, text: str) -> bool:
        """Speak text and return success status"""
        raise NotImplementedError
    
    def generate_audio(self, text: str) -> Optional[str]:
        """Generate audio file and return path"""
        raise NotImplementedError


class EdgeTTSBackend(TTSBackend):
    """Microsoft Edge TTS backend (fast, natural voices)"""
    
    def __init__(self, voice="en-US-AriaNeural", rate="+0%", pitch="+0Hz"):
        super().__init__()
        if not EDGE_TTS_AVAILABLE:
            raise ImportError("edge-tts not available")
        
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        
        # For Jarvis-like experience, prefer male voices
        self.jarvis_voices = [
            "en-US-GuyNeural",      # Male, confident
            "en-US-DavisNeural",    # Male, warm
            "en-US-TonyNeural",     # Male, professional
            "en-GB-RyanNeural",     # Male, British accent
        ]
    
    def set_jarvis_voice(self):
        """Set a Jarvis-appropriate voice"""
        self.voice = self.jarvis_voices[0]  # Default to Guy
        self.logger.info(f"Set Jarvis voice: {self.voice}")
    
    async def _generate_audio_async(self, text: str, output_path: str):
        """Generate audio asynchronously"""
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
        await communicate.save(output_path)
    
    def generate_audio(self, text: str) -> Optional[str]:
        """Generate audio file"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                output_path = tmp.name
            
            # Run async function in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._generate_audio_async(text, output_path))
            finally:
                loop.close()
            
            self.logger.debug(f"Edge TTS generated: {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"Edge TTS generation error: {e}")
            return None
    
    def speak(self, text: str) -> bool:
        """Generate and play audio"""
        audio_path = self.generate_audio(text)
        if audio_path:
            # Play using system default player
            import os
            try:
                os.startfile(audio_path)
                # Estimate duration and wait
                word_count = len(text.split())
                duration = max(2.0, word_count / 3.0)  # ~3 words per second
                time.sleep(duration)
                return True
            except Exception as e:
                self.logger.error(f"Audio playback error: {e}")
        return False


class SAPIBackend(TTSBackend):
    """Windows SAPI TTS backend"""
    
    def __init__(self):
        super().__init__()
        if not PYTTSX3_AVAILABLE:
            raise ImportError("pyttsx3 not available")
        
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize SAPI engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Configure for Jarvis-like voice
            voices = self.engine.getProperty("voices")
            if voices:
                # Prefer male voices for Jarvis
                male_voice = None
                for voice in voices:
                    name = voice.name.lower()
                    if any(k in name for k in ["david", "guy", "zach", "brian", "male"]):
                        male_voice = voice
                        break
                
                if male_voice:
                    self.engine.setProperty("voice", male_voice.id)
                    self.logger.info(f"Set SAPI voice: {male_voice.name}")
            
            # Set speech properties
            self.engine.setProperty("rate", 185)  # Slightly brisk
            self.engine.setProperty("volume", 0.95)
            
            self.logger.info("SAPI TTS engine initialized")
        
        except Exception as e:
            self.logger.error(f"SAPI initialization error: {e}")
            self.engine = None
    
    def speak(self, text: str) -> bool:
        """Speak text directly"""
        if not self.engine:
            return False
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            self.logger.error(f"SAPI speak error: {e}")
            return False
    
    def generate_audio(self, text: str) -> Optional[str]:
        """SAPI doesn't support file generation easily"""
        return None


class GTTSBackend(TTSBackend):
    """Google TTS backend (fallback)"""
    
    def __init__(self, lang="en", slow=False):
        super().__init__()
        if not GTTS_AVAILABLE:
            raise ImportError("gtts not available")
        
        self.lang = lang
        self.slow = slow
    
    def generate_audio(self, text: str) -> Optional[str]:
        """Generate audio file"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                output_path = tmp.name
            
            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            tts.save(output_path)
            
            self.logger.debug(f"gTTS generated: {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"gTTS generation error: {e}")
            return None
    
    def speak(self, text: str) -> bool:
        """Generate and play audio"""
        audio_path = self.generate_audio(text)
        if audio_path:
            import os
            try:
                os.startfile(audio_path)
                word_count = len(text.split())
                duration = max(2.0, word_count / 3.0)
                time.sleep(duration)
                return True
            except Exception as e:
                self.logger.error(f"Audio playback error: {e}")
        return False


class TTSManager:
    """Manages different TTS backends"""
    
    def __init__(self, backend="edge_tts", config=None):
        self.backend_name = backend
        self.config = config or {}
        self.backend = None
        self.logger = logging.getLogger(__name__)
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected backend"""
        try:
            if self.backend_name == "edge_tts":
                voice = self.config.get("edge_voice", "en-US-GuyNeural")
                self.backend = EdgeTTSBackend(voice=voice)
                # Set Jarvis voice if persona is Jarvis
                if self.config.get("persona_name", "").lower() == "jarvis":
                    self.backend.set_jarvis_voice()
            
            elif self.backend_name == "sapi":
                self.backend = SAPIBackend()
            
            elif self.backend_name == "gtts":
                lang = self.config.get("tts_language", "en")
                self.backend = GTTSBackend(lang=lang)
            
            else:
                raise ValueError(f"Unknown TTS backend: {self.backend_name}")
            
            self.logger.info(f"TTS backend '{self.backend_name}' initialized")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS backend '{self.backend_name}': {e}")
            # Fallback chain: edge_tts -> sapi -> gtts
            fallbacks = ["edge_tts", "sapi", "gtts"]
            for fallback in fallbacks:
                if fallback != self.backend_name:
                    try:
                        if fallback == "edge_tts" and EDGE_TTS_AVAILABLE:
                            self.backend = EdgeTTSBackend()
                        elif fallback == "sapi" and PYTTSX3_AVAILABLE:
                            self.backend = SAPIBackend()
                        elif fallback == "gtts" and GTTS_AVAILABLE:
                            self.backend = GTTSBackend()
                        else:
                            continue
                        
                        self.logger.info(f"Fallback to TTS backend: {fallback}")
                        break
                    except Exception:
                        continue
    
    def speak(self, text: str) -> bool:
        """Speak text using the configured backend"""
        if not self.backend:
            self.logger.error("No TTS backend available")
            return False
        
        return self.backend.speak(text)
    
    def generate_audio(self, text: str) -> Optional[str]:
        """Generate audio file"""
        if not self.backend:
            return None
        
        return self.backend.generate_audio(text)
    
    def is_available(self) -> bool:
        """Check if backend is available"""
        return self.backend is not None


def create_tts_manager(backend="edge_tts", config=None):
    """Factory function to create TTS manager"""
    return TTSManager(backend, config)
