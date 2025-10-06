import cv2
import threading
import speech_recognition as sr
import time
import numpy as np
from utils import load_icon, overlay_image, is_point_in_rect
from llm_service import LLMService
from gtts import gTTS
import queue
import tempfile
from system_controller import SystemController
import pygame
import os
import webbrowser
import logging
import json
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from contextlib import contextmanager
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import subprocess
import sys
import asyncio

from action_registry import ActionRegistry
from speech_asr import create_asr_manager
from tts_backend import create_tts_manager
from wake_word import create_wake_word_detector
from memory_store import create_memory_store
from context_oracle import create_context_oracle
from metrics import get_metrics_collector, start_timer, end_timer

# Optional automation imports
try:
    from task_automation import AutomationEngine, NaturalLanguageRoutineCreator
    AUTOMATION_AVAILABLE = True
except Exception:
    AUTOMATION_AVAILABLE = False

# Try to import Windows SAPI for fallback TTS
try:
    import pyttsx3

    SAPI_AVAILABLE = True
except ImportError:
    SAPI_AVAILABLE = False

# Suppress verbose comtypes debug logging globally
logging.getLogger("comtypes").setLevel(logging.WARNING)
logging.getLogger("comtypes._comobject").setLevel(logging.WARNING)
logging.getLogger("comtypes._vtbl").setLevel(logging.WARNING)
logging.getLogger("comtypes.client._managing").setLevel(logging.WARNING)
logging.getLogger("comtypes.client._generate").setLevel(logging.WARNING)
logging.getLogger("comtypes._post_coinit.unknwn").setLevel(logging.WARNING)


@dataclass
class VoiceAssistantConfig:
    """Configuration class for Voice Assistant"""

    speech_timeout: float = 5.0
    phrase_time_limit: float = 10.0
    hover_timeout: float = 1.5
    activation_radius: int = 100
    tts_language: str = "en"
    max_audio_queue: int = 10
    tts_cache_size: int = 50
    log_level: str = "INFO"
    retry_attempts: int = 3
    retry_delay: float = 1.0
    continuous_listening: bool = True  # Enable continuous listening mode
    listen_pause_duration: float = 2.0  # Pause between listening sessions
    prefer_sapi_tts: bool = True  # Prefer Windows SAPI over gTTS for better reliability
    enable_audio_feedback: bool = (
        True  # Enable audio responses (can be disabled for silent mode)
    )
    # Jarvis-like configuration
    persona_name: str = "Jarvis"
    wake_word: str = "jarvis"
    require_wake_word: bool = True

    @classmethod
    def from_file(cls, config_path: str) -> "VoiceAssistantConfig":
        """Load configuration from JSON file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            # Filter out unknown keys to avoid TypeError
            valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
            filtered_data = {k: v for k, v in config_data.items() if k in valid_keys}

            return cls(**filtered_data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            return cls()  # Return default config


class Constants:
    """Application constants"""

    TTS_SPEED_WPS = 3  # words per second
    MIN_SPEECH_DURATION = 2  # minimum seconds for speech
    UI_COLORS = {
        "LISTENING": (255, 0, 0),  # Red in BGR
        "THINKING": (255, 255, 0),  # Yellow in BGR
        "SPEAKING": (0, 255, 0),  # Green in BGR
        "READY": (0, 255, 0),  # Green in BGR
        "ERROR": (0, 0, 255),  # Blue in BGR
    }
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2


class AudioManager:
    """Manages audio playback and TTS caching"""

    def __init__(self, config: VoiceAssistantConfig):
        self.config = config
        self.audio_queue = queue.Queue(maxsize=config.max_audio_queue)
        self.stop_audio = False
        self.audio_thread = None
        self.tts_cache: Dict[str, str] = {}
        self.temp_files: List[str] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        self.logger = logging.getLogger(__name__)

        # Suppress comtypes debug logging before initializing SAPI
        logging.getLogger("comtypes._comobject").setLevel(logging.WARNING)
        logging.getLogger("comtypes._vtbl").setLevel(logging.WARNING)
        logging.getLogger("comtypes.client._managing").setLevel(logging.WARNING)
        logging.getLogger("comtypes.client._generate").setLevel(logging.WARNING)
        logging.getLogger("comtypes._post_coinit.unknwn").setLevel(logging.WARNING)
        logging.getLogger("comtypes").setLevel(logging.WARNING)

        # Initialize Windows SAPI TTS engine as fallback
        self.sapi_engine = None
        if SAPI_AVAILABLE:
            try:
                self.sapi_engine = pyttsx3.init()
                # Configure SAPI settings
                voices = self.sapi_engine.getProperty("voices")
                if voices:
                    # Try to find a female voice first, fallback to any voice
                    female_voice = None
                    for voice in voices:
                        if (
                            "female" in voice.name.lower()
                            or "zira" in voice.name.lower()
                        ):
                            female_voice = voice
                            break
                    if female_voice:
                        self.sapi_engine.setProperty("voice", female_voice.id)

                # Prefer a "Jarvis"-like male voice when persona is Jarvis; otherwise fallback
                try:
                    preferred_male = None
                    for voice in voices:
                        name = voice.name.lower()
                        if any(k in name for k in ["david", "guy", "zach", "brian", "male"]):
                            preferred_male = voice
                            break
                    if getattr(self.config, "persona_name", "").lower() == "jarvis" and preferred_male:
                        self.sapi_engine.setProperty("voice", preferred_male.id)
                except Exception:
                    pass

                # Set speech rate and volume
                self.sapi_engine.setProperty("rate", 185)  # Slightly brisk Jarvis cadence
                self.sapi_engine.setProperty("volume", 0.95)  # Volume level (0.0 to 1.0)
                self.logger.info("Windows SAPI TTS engine initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Windows SAPI TTS: {e}")
                self.sapi_engine = None

        # Initialize pygame mixer
        try:
            pygame.mixer.init()
            self.logger.info("Pygame mixer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize pygame mixer: {e}")

        self._start_audio_processor()

    def _start_audio_processor(self):
        """Start the audio processing thread"""

        def audio_processor():
            while not self.stop_audio:
                try:
                    audio_file = self.audio_queue.get(timeout=0.5)
                    if audio_file:
                        self._play_audio_file(audio_file)
                        self._cleanup_temp_file(audio_file)
                        self.audio_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Audio playback error: {e}")

        self.audio_thread = threading.Thread(target=audio_processor, daemon=True)
        self.audio_thread.start()

    def _play_audio_file(self, audio_file: str):
        """Play audio file using pygame mixer with fallbacks"""
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            self.logger.debug(f"Successfully played audio file: {audio_file}")

        except Exception as e:
            self.logger.error(f"Failed to play audio file {audio_file}: {e}")
            # Fallback 1: try to play using Windows media player
            try:
                import subprocess

                subprocess.run(
                    ["start", "/min", "wmplayer.exe", f'"{audio_file}"'],
                    shell=True,
                    check=False,
                    timeout=10,
                )
                time.sleep(2)  # Give it time to play
                self.logger.info("Audio played using Windows Media Player fallback")
            except Exception as fallback_error:
                self.logger.error(
                    f"Windows Media Player fallback also failed: {fallback_error}"
                )
                # Fallback 2: try using system default audio player
                try:
                    import os

                    os.startfile(audio_file)
                    time.sleep(3)  # Give it time to play
                    self.logger.info("Audio played using system default player")
                except Exception as final_fallback_error:
                    self.logger.error(
                        f"All audio playback methods failed: {final_fallback_error}"
                    )

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for TTS text"""
        return hashlib.md5(f"{text}_{self.config.tts_language}".encode()).hexdigest()

    def _cleanup_temp_file(self, file_path: str):
        """Safely cleanup temporary file"""
        try:
            if file_path in self.temp_files:
                self.temp_files.remove(file_path)
            os.unlink(file_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

    def _speak_with_sapi(self, text: str) -> bool:
        """Use Windows SAPI to speak text directly (fallback method)"""
        if not self.sapi_engine:
            return False

        try:
            self.logger.info(f"Using Windows SAPI to speak: {text}")
            self.sapi_engine.say(text)
            self.sapi_engine.runAndWait()
            return True
        except Exception as e:
            self.logger.error(f"Windows SAPI TTS failed: {e}")
            return False

    def generate_speech(self, text: str) -> Optional[str]:
        """Generate TTS audio file with caching and better error handling"""
        if not text:
            return None

        cache_key = self._get_cache_key(text)

        # Check cache first
        if cache_key in self.tts_cache and os.path.exists(self.tts_cache[cache_key]):
            return self.tts_cache[cache_key]

        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                temp_filename = fp.name

            # Generate speech with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    tts = gTTS(text=text, lang=self.config.tts_language, slow=False)
                    tts.save(temp_filename)
                    break
                except Exception as e:
                    self.logger.warning(
                        f"TTS generation attempt {attempt + 1} failed: {e}"
                    )
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(1)  # Wait before retry

            # Verify file was created and has content
            if not os.path.exists(temp_filename) or os.path.getsize(temp_filename) == 0:
                raise Exception("Generated audio file is empty or doesn't exist")

            # Cache the result
            if len(self.tts_cache) >= self.config.tts_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.tts_cache))
                old_file = self.tts_cache.pop(oldest_key)
                self._cleanup_temp_file(old_file)

            self.tts_cache[cache_key] = temp_filename
            self.temp_files.append(temp_filename)
            self.logger.debug(f"TTS audio generated successfully: {temp_filename}")
            return temp_filename

        except Exception as e:
            self.logger.error(f"TTS generation error: {e}")
            # Clean up failed temp file
            try:
                if "temp_filename" in locals() and os.path.exists(temp_filename):
                    os.unlink(temp_filename)
            except:
                pass
            return None

    def play_audio(self, audio_file: str):
        """Add audio file to playback queue"""
        try:
            self.audio_queue.put(audio_file, timeout=1.0)
        except queue.Full:
            self.logger.warning("Audio queue is full, skipping audio")

    def cleanup(self):
        """Cleanup resources"""
        self.stop_audio = True
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)

        # Stop pygame mixer
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception as e:
            self.logger.warning(f"Error stopping pygame mixer: {e}")

        # Stop SAPI engine
        try:
            if self.sapi_engine:
                self.sapi_engine.stop()
                self.sapi_engine = None
        except Exception as e:
            self.logger.warning(f"Error stopping SAPI engine: {e}")

        # Cleanup temp files
        for file_path in self.temp_files[:]:
            self._cleanup_temp_file(file_path)

        # Cleanup cache files
        for file_path in self.tts_cache.values():
            self._cleanup_temp_file(file_path)

        self.thread_pool.shutdown(wait=True)


class SpeechRecognizer:
    """Enhanced speech recognition with retry logic"""

    def __init__(self, config: VoiceAssistantConfig):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.logger = logging.getLogger(__name__)
        self._calibrate_microphone()

    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.logger.info("Microphone calibrated for ambient noise")
        except Exception as e:
            self.logger.warning(f"Microphone calibration failed: {e}")

    def recognize_speech(self) -> Tuple[Optional[str], str]:
        """Recognize speech with retry logic and detailed error handling"""
        for attempt in range(self.config.retry_attempts):
            try:
                with self.mic as source:
                    self.logger.debug(f"Listening attempt {attempt + 1}")
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.config.speech_timeout,
                        phrase_time_limit=self.config.phrase_time_limit,
                    )

                    result = self.recognizer.recognize_google(audio)
                    self.logger.info(f"Speech recognized: {result}")
                    return result, "success"

            except sr.WaitTimeoutError:
                error_msg = "No speech detected within timeout"
                self.logger.warning(f"{error_msg} (attempt {attempt + 1})")
                if attempt == self.config.retry_attempts - 1:
                    return None, error_msg

            except sr.UnknownValueError:
                error_msg = "Could not understand speech"
                self.logger.warning(f"{error_msg} (attempt {attempt + 1})")
                if attempt == self.config.retry_attempts - 1:
                    return None, error_msg

            except sr.RequestError as e:
                error_msg = f"Speech service error: {e}"
                self.logger.error(f"{error_msg} (attempt {attempt + 1})")
                if attempt == self.config.retry_attempts - 1:
                    return None, error_msg

            # Wait before retry
            if attempt < self.config.retry_attempts - 1:
                time.sleep(self.config.retry_delay)

        return None, "Max retry attempts reached"


class CommandProcessor:
    """Process and validate commands"""

    def __init__(self, system_controller: SystemController):
        self.system = system_controller
        self.logger = logging.getLogger(__name__)

        # Command validation whitelist
        self.allowed_system_commands = {
            "search",
            "google",
            "open",
            "close",
            "volume",
            "mute",
            "play",
            "pause",
            "next",
            "previous",
            "press",
            "type",
            "move",
            "click",
            "scroll",
            "screenshot",
            "lock",
        }

    def _sanitize_input(self, command: str) -> str:
        """Sanitize user input"""
        # Remove potentially harmful characters
        sanitized = "".join(char for char in command if char.isprintable())
        return sanitized.strip()

    def _validate_command(self, command: str) -> bool:
        """Validate if command is allowed"""
        command_words = command.lower().split()
        if not command_words:
            return False

        # Check if first word is in allowed commands
        return any(
            allowed in command_words[0] for allowed in self.allowed_system_commands
        )

    def handle_system_command(self, command: str) -> Optional[str]:
        """Handle system commands with validation"""
        sanitized_command = self._sanitize_input(command)

        if not self._validate_command(sanitized_command):
            return None

        try:
            # Web search commands
            if "search for" in sanitized_command:
                query = sanitized_command.replace("search for", "").strip()
                encoded_query = urllib.parse.quote_plus(query)
                webbrowser.open(f"https://www.google.com/search?q={encoded_query}")
                return f"Searching Google for {query}"
            elif "google" in sanitized_command:
                query = sanitized_command.replace("google", "").strip()
                encoded_query = urllib.parse.quote_plus(query)
                webbrowser.open(f"https://www.google.com/search?q={encoded_query}")
                return f"Searching Google for {query}"

            # Application control
            elif "open " in sanitized_command:
                app = sanitized_command.replace("open ", "").strip()
                return self.system.open_application(app)
            elif "close " in sanitized_command:
                app = sanitized_command.replace("close ", "").strip()
                return self.system.close_application(app)

            # Volume control
            elif "volume up" in sanitized_command:
                return self.system.volume_up()
            elif "volume down" in sanitized_command:
                return self.system.volume_down()
            elif "mute" in sanitized_command:
                return self.system.mute()
            elif "set volume to " in sanitized_command:
                level_str = (
                    sanitized_command.replace("set volume to ", "")
                    .replace("%", "")
                    .strip()
                )
                if level_str.isdigit() and 0 <= int(level_str) <= 100:
                    return self.system.set_volume(int(level_str))
                return "Please specify a number between 0 and 100"

            # Media control
            elif "play" in sanitized_command or "pause" in sanitized_command:
                return self.system.hotkey("playpause")
            elif "next" in sanitized_command:
                return self.system.hotkey("nexttrack")
            elif "previous" in sanitized_command:
                return self.system.hotkey("prevtrack")

            # Input control
            elif "press " in sanitized_command:
                key = sanitized_command.replace("press ", "").strip()
                return self.system.press_key(key)
            elif "type " in sanitized_command:
                text = sanitized_command.replace("type ", "").strip()
                return self.system.type_text(text)
            elif "move mouse " in sanitized_command:
                direction = sanitized_command.replace("move mouse ", "").strip()
                return self.system.mouse_move(direction)
            elif "click" in sanitized_command:
                button = "right" if "right" in sanitized_command else "left"
                return self.system.click(button)
            elif "scroll " in sanitized_command:
                direction = "down" if "down" in sanitized_command else "up"
                return self.system.scroll(direction)

            # System commands
            elif "screenshot" in sanitized_command:
                return self.system.take_screenshot()
            elif "lock" in sanitized_command:
                return self.system.lock_pc()

        except Exception as e:
            self.logger.error(f"System command error: {e}")
            return f"Error executing command: {str(e)}"

        return None


class VoiceAssistant:
    """Enhanced Voice Assistant with improved architecture"""

    def __init__(self, config_path: Optional[str] = None):
        # Initialize threading attributes first
        self.listen_thread = None

        # Load configuration with error handling
        try:
            self.config = (
                VoiceAssistantConfig.from_file(config_path)
                if config_path
                else VoiceAssistantConfig()
            )
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default configuration")
            self.config = VoiceAssistantConfig()

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        try:
            # Initialize components
            self.audio_manager = AudioManager(self.config)
            self.speech_recognizer = SpeechRecognizer(self.config)
            self.system = SystemController()
            self.command_processor = CommandProcessor(self.system)

            # Initialize LLM service with error handling
            try:
                self.llm_service = LLMService()
                if not self.llm_service.enabled:
                    self.logger.warning(
                        "LLM service is disabled - AI features will not be available. Simple fallback responses will be used."
                    )
                else:
                    self.logger.info("LLM service initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM service: {e}")
                self.llm_service = None

            # Initialize new Jarvis backends
            self._init_jarvis_backends()

            # State management
            self.listening = False
            self.is_speaking = False
            self.thinking = False
            self.result = ""
            self.last_result_time = 0
            self.hover_start = None
            self.first_run = True
            self.continuous_listening_active = False
            self.continuous_listen_thread = None
            self.stop_continuous_listening = False

            # UI components
            self.mic_icon = load_icon("mic_icon.png")
            self.return_icon = load_icon("return_icon.png", (80, 80))

            self.logger.info("Voice Assistant initialized successfully")

            # Initialize automation subsystem (non-fatal if it fails)
            self._init_automation()

        except Exception as e:
            self.logger.error(f"Voice Assistant initialization failed: {e}")
            # Ensure cleanup is safe even if initialization failed
            self.audio_manager = None
            raise

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("voice_assistant.log"),
                logging.StreamHandler(),
            ],
        )

        # Suppress verbose debug output from comtypes library (used by pyttsx3)
        logging.getLogger("comtypes._comobject").setLevel(logging.WARNING)
        logging.getLogger("comtypes._vtbl").setLevel(logging.WARNING)
        logging.getLogger("comtypes.client._managing").setLevel(logging.WARNING)
        logging.getLogger("comtypes.client._generate").setLevel(logging.WARNING)
        logging.getLogger("comtypes._post_coinit.unknwn").setLevel(logging.WARNING)
    def _init_jarvis_backends(self):
        """Initialize Jarvis-specific backends based on config"""
        try:
            # Initialize metrics collector
            self.metrics = get_metrics_collector()

            # Initialize memory store
            self.memory = create_memory_store(
                max_entries=100,
                persistence_file="jarvis_memory.json"
            )

            # Initialize context oracle
            self.context_oracle = create_context_oracle(update_interval=2.0)
            if self.config.__dict__.get("features", {}).get("context_monitoring", True):
                self.context_oracle.start_monitoring()

            # Initialize ASR backend
            asr_backend = self.config.__dict__.get("asr_backend", "speech_recognition")
            if asr_backend != "speech_recognition":
                try:
                    self.asr_manager = create_asr_manager(asr_backend, self.config.__dict__)
                    if self.asr_manager.is_available():
                        self.logger.info(f"ASR backend '{asr_backend}' initialized")
                    else:
                        self.logger.warning(f"ASR backend '{asr_backend}' failed, using fallback")
                        self.asr_manager = None
                except Exception as e:
                    self.logger.error(f"ASR backend initialization failed: {e}")
                    self.asr_manager = None
            else:
                self.asr_manager = None

            # Initialize TTS backend
            tts_backend = self.config.__dict__.get("tts_backend", "sapi")
            if tts_backend != "sapi":
                try:
                    self.tts_manager = create_tts_manager(tts_backend, self.config.__dict__)
                    if self.tts_manager.is_available():
                        self.logger.info(f"TTS backend '{tts_backend}' initialized")
                    else:
                        self.logger.warning(f"TTS backend '{tts_backend}' failed, using fallback")
                        self.tts_manager = None
                except Exception as e:
                    self.logger.error(f"TTS backend initialization failed: {e}")
                    self.tts_manager = None
            else:
                self.tts_manager = None

            # Initialize wake word detector (optional)
            if self.config.__dict__.get("features", {}).get("wake_word_detection", False):
                try:
                    wake_backend = self.config.__dict__.get("wake_word_backend", "openwakeword")
                    wake_word = self.config.__dict__.get("wake_word", "jarvis")
                    self.wake_detector = create_wake_word_detector(wake_backend, wake_word)
                    if hasattr(self.wake_detector, 'start'):
                        self.wake_detector.start()
                        self.logger.info(f"Wake word detection started: {wake_word}")
                except Exception as e:
                    self.logger.error(f"Wake word detection initialization failed: {e}")
                    self.wake_detector = None
            else:
                self.wake_detector = None

            self.logger.info("Jarvis backends initialized successfully")

        except Exception as e:
            self.logger.error(f"Jarvis backend initialization failed: {e}")
            # Set fallback values
            self.metrics = get_metrics_collector()
            self.memory = None
            self.context_oracle = None
            self.asr_manager = None
            self.tts_manager = None
            self.wake_detector = None

    def _init_automation(self):
        """Initialize automation engine and NLP routine creator if available"""
        self.automation_engine = None
        self.nlp_routine_creator = None
        if not AUTOMATION_AVAILABLE:
            self.logger.info("Automation modules not available - skipping automation init")
            return
        try:
            self.automation_engine = AutomationEngine(data_dir="data/automation")
            self.nlp_routine_creator = NaturalLanguageRoutineCreator(self.automation_engine.routine_builder)
            self.logger.info("Automation engine + NLP routine creator initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize automation subsystem: {e}")
            self.automation_engine = None
            self.nlp_routine_creator = None

    def _run_async(self, coro):
        """Run an async coroutine in a background thread with its own event loop"""
        def runner():
            try:
                asyncio.run(coro)
            except Exception as e:
                self.logger.error(f"Async execution error: {e}")
        threading.Thread(target=runner, daemon=True).start()

    # ---------------- Automation Handling ---------------- #
    def _maybe_handle_automation(self, command: str) -> bool:
        """Detect and handle automation-related natural language commands.

        Returns True if the command was handled (and response spoken), else False.
        """
        if not self.automation_engine or not self.nlp_routine_creator:
            return False

        cmd = command.lower().strip()

        # Create routine intents
        if ("create" in cmd or "make" in cmd or "build" in cmd) and ("routine" in cmd or "automation" in cmd):
            return self._automation_create_routine(cmd)

        # List routines
        if "list routines" in cmd or "list automations" in cmd or cmd.startswith("what routines"):
            routines = self.automation_engine.list_routines()
            if not routines:
                self.speak("You don't have any routines yet.")
            else:
                names = ", ".join(r["name"] for r in routines[:10])
                extra = "" if len(routines) <= 10 else f" and {len(routines)-10} more"
                self.speak(f"You have {len(routines)} routines: {names}{extra}.")
            return True

        # Run routine (e.g., "run focus mode routine", "start morning routine")
        if any(cmd.startswith(p) for p in ["run ", "start ", "execute "]):
            keywords = cmd.split()
            # Remove leading verb
            if keywords[0] in ["run", "start", "execute"]:
                remainder = " ".join(keywords[1:])
            else:
                remainder = cmd
            remainder = remainder.replace("routine", "").replace("automation", "").strip()
            if remainder:
                return self._automation_run_by_name(remainder)
        return False

    def _automation_create_routine(self, command: str) -> bool:
        """Create a routine from natural language and persist it."""
        try:
            parse_result = self.nlp_routine_creator.parse_natural_language(command)
            routine = parse_result["routine"]
            # Persist and register
            self.automation_engine.add_routine(routine)
            self.speak(f"Created routine {routine.name} with {len(routine.tasks)} tasks.")
            return True
        except Exception as e:
            self.logger.error(f"Routine creation failed: {e}")
            self.speak("I couldn't create that routine.")
            return True

    def _automation_run_by_name(self, name_fragment: str) -> bool:
        """Find routine by partial name match and execute it asynchronously."""
        routines = list(self.automation_engine.routines.values())
        name_fragment_lower = name_fragment.lower()
        # Simple scoring: substring containment
        matches = [r for r in routines if name_fragment_lower in r.name.lower()]
        if not matches:
            self.speak(f"I couldn't find a routine matching {name_fragment}.")
            return True
        routine = sorted(matches, key=lambda r: len(r.name))[0]  # shortest name first
        self.speak(f"Running routine {routine.name} now.")
        async def runner():
            await self.automation_engine.execute_routine(routine.id)
        self._run_async(runner())
        return True

    @contextmanager
    def _state_manager(self, state_name: str):
        """Context manager for state transitions"""
        self.logger.debug(f"Entering state: {state_name}")
        setattr(self, state_name, True)
        try:
            yield
        finally:
            setattr(self, state_name, False)
            self.logger.debug(f"Exiting state: {state_name}")

    def speak(self, text: str):
        """Enhanced speak method with better error handling and fallbacks"""
        if not text or self.is_speaking:
            return

        # Check if audio feedback is enabled
        if not self.config.enable_audio_feedback:
            self.result = text
            return

        self.logger.info(f"Speaking: {text}")

        def tts_task():
            with self._state_manager("is_speaking"):
                try:
                    timer_id = start_timer("tts_start")
                    success = False

                    # Try new TTS backend first if available
                    if hasattr(self, 'tts_manager') and self.tts_manager:
                        try:
                            success = self.tts_manager.speak(text)
                            if success:
                                end_timer(timer_id, {"backend": "new_tts"})
                                self.logger.debug("New TTS backend succeeded")
                        except Exception as e:
                            self.logger.warning(f"New TTS backend failed: {e}")

                    # Fallback to original TTS logic if new backend failed
                    if not success:
                        # Choose TTS method based on configuration
                        if self.config.prefer_sapi_tts and self.audio_manager.sapi_engine:
                            # Try Windows SAPI first
                            success = self.audio_manager._speak_with_sapi(text)
                            if not success:
                                # Fallback to gTTS
                                self.logger.warning("SAPI failed, trying gTTS fallback")
                                audio_file = self.audio_manager.generate_speech(text)
                                if audio_file:
                                    self.audio_manager.play_audio(audio_file)
                                    success = True
                        else:
                            # Try gTTS first (default behavior)
                            audio_file = self.audio_manager.generate_speech(text)
                            if audio_file:
                                self.audio_manager.play_audio(audio_file)
                                success = True
                            else:
                                # Fallback to Windows SAPI
                                self.logger.warning(
                                    "gTTS failed, trying Windows SAPI fallback"
                                )
                                success = self.audio_manager._speak_with_sapi(text)

                        if success:
                            end_timer(timer_id, {"backend": "legacy_tts"})

                    if success:
                        # Estimate speaking duration for gTTS
                        if not self.config.prefer_sapi_tts:
                            word_count = len(text.split())
                            duration = max(
                                Constants.MIN_SPEECH_DURATION,
                                word_count / Constants.TTS_SPEED_WPS,
                            )
                            time.sleep(duration)
                        else:
                            # For SAPI, add a small delay to ensure completion
                            time.sleep(0.5)
                    else:
                        # All TTS methods failed, at least show the text on screen
                        self.logger.error(
                            "All TTS methods failed, displaying text only"
                        )
                        self.result = f"[AUDIO FAILED] {text}"

                except Exception as e:
                    self.logger.error(f"TTS error: {e}")
                    # Final fallback: show text on screen
                    self.result = f"[AUDIO FAILED] {text}"
                finally:
                    self.thinking = False

        # Run TTS task in current thread if we're in listening context, otherwise use separate thread
        if self.listening:
            tts_task()
        else:
            threading.Thread(target=tts_task, daemon=True).start()

    def start_listening(self):
        """Enhanced listening with better state management"""
        if self.listening or self.is_speaking:
            return

        self.logger.info("Starting to listen")

        def listen_task():
            with self._state_manager("listening"):
                self.result = ""

                # Recognize speech with new ASR backend if available
                timer_id = start_timer("asr_latency")

                if hasattr(self, 'asr_manager') and self.asr_manager:
                    try:
                        recognized_text, error_message = self.asr_manager.transcribe()
                        end_timer(timer_id, {"backend": "new_asr"})
                    except Exception as e:
                        self.logger.warning(f"New ASR backend failed: {e}")
                        recognized_text, error_message = self.speech_recognizer.recognize_speech()
                        end_timer(timer_id, {"backend": "legacy_asr"})
                else:
                    recognized_text, error_message = self.speech_recognizer.recognize_speech()
                    end_timer(timer_id, {"backend": "legacy_asr"})

                if recognized_text:
                    text_low = recognized_text.lower().strip()
                    # Optional wake word gating for Jarvis persona
                    if self.config.require_wake_word:
                        wake = getattr(self.config, "wake_word", "jarvis").lower()
                        if wake not in text_low:
                            # Ignore phrases without the wake word to avoid interrupting flow
                            self.result = recognized_text
                            self.last_result_time = time.time()
                            return
                        # Strip wake word for cleaner command
                        text_low = text_low.replace(wake, "", 1).strip(",. ")
                        recognized_text = text_low if text_low else recognized_text

                    self.result = recognized_text
                    self.last_result_time = time.time()
                    self.speak("Got it")

                    # Process command immediately after "Got it" is spoken
                    time.sleep(1.0)
                    self._process_command()
                else:
                    self.result = error_message
                    if "timeout" in error_message.lower():
                        self.speak("I didn't hear anything, please try again")
                    elif "understand" in error_message.lower():
                        self.speak("I didn't understand that, could you repeat?")
                    else:
                        self.speak("I'm having trouble with the speech service")

        self.listen_thread = threading.Thread(target=listen_task, daemon=True)
        self.listen_thread.start()

    def start_continuous_listening(self):
        """Start continuous listening mode"""
        if self.continuous_listening_active or not self.config.continuous_listening:
            return

        self.continuous_listening_active = True
        self.stop_continuous_listening = False
        self.logger.info("Starting continuous listening mode")

        def continuous_listen_task():
            while (
                not self.stop_continuous_listening and self.continuous_listening_active
            ):
                try:
                    # Only listen if not currently speaking or processing
                    if (
                        not self.listening
                        and not self.is_speaking
                        and not self.thinking
                    ):
                        self.start_listening()

                        # Wait for current listening to complete
                        if self.listen_thread and self.listen_thread.is_alive():
                            self.listen_thread.join(
                                timeout=self.config.speech_timeout + 2
                            )

                    # Pause between listening sessions
                    time.sleep(self.config.listen_pause_duration)

                except Exception as e:
                    self.logger.error(f"Error in continuous listening: {e}")
                    time.sleep(self.config.listen_pause_duration)

        self.continuous_listen_thread = threading.Thread(
            target=continuous_listen_task, daemon=True
        )
        self.continuous_listen_thread.start()

    def stop_continuous_listening_mode(self):
        """Stop continuous listening mode"""
        if not self.continuous_listening_active:
            return

        self.logger.info("Stopping continuous listening mode")
        self.stop_continuous_listening = True
        self.continuous_listening_active = False

        if self.continuous_listen_thread and self.continuous_listen_thread.is_alive():
            self.continuous_listen_thread.join(timeout=2.0)

    def _process_command(self):
        """Enhanced command processing"""
        if not self.result:
            self.logger.warning("No command to process - result is empty")
            return

        command = self.result.lower()
        self.logger.info(f"Processing command: {command}")

        try:
            # Try system commands first
            system_response = self.command_processor.handle_system_command(command)
            if system_response:
                self.logger.info(f"System command executed: {system_response}")
                self.result = system_response
                self.speak(system_response)
                return

            # Automation commands (create/list/run routines)
            if self._maybe_handle_automation(command):
                return

            # Handle time/date queries
            if "time" in command:
                time_str = time.strftime("%I:%M %p")
                self.result = f"Current time is {time_str}"
                self.logger.info(f"Time query answered: {self.result}")
                self.speak(self.result)
                return
            elif "date" in command:
                date_str = time.strftime("%B %d, %Y")
                self.result = f"Today is {date_str}"
                self.logger.info(f"Date query answered: {self.result}")
                self.speak(self.result)
                return

            # Handle LLM queries (structured)
            self.logger.info("Delegating to LLM query handler")
            self._handle_llm_query(command)

        except Exception as e:
            self.logger.error(f"Command processing error: {e}")
            error_msg = "I encountered an error processing your request"
            self.result = error_msg
            self.speak(error_msg)

    def _handle_llm_query(self, command: str):
        """Handle LLM queries in separate thread (structured command mode)"""
        # Check for follow-up queries first
        if hasattr(self, 'memory') and self.memory:
            follow_up_response = self.memory.handle_follow_up(command)
            if follow_up_response:
                self.result = follow_up_response
                self.speak(follow_up_response)
                return

        if not self.llm_service or not self.llm_service.enabled:
            fallback_response = self._get_fallback_response(command)
            self.result = fallback_response
            self.speak(fallback_response)
            return

        # Lazy-create action registry
        if not hasattr(self, "action_registry"):
            self.action_registry = ActionRegistry()

        self.thinking = True
        self.result = "Processing your request..."
        self.speak("Working on it")

        def llm_task():
            try:
                timer_id = start_timer("llm_response")
                data = self.llm_service.generate_structured(command)
                end_timer(timer_id, {"command_type": data.get("type", "unknown")})

                if data.get("type") == "command" and data.get("action"):
                    reply = data.get("reply") or ""
                    action = data["action"]
                    args = data.get("args", {})

                    # Dispatch to system action
                    dispatch_timer = start_timer("command_dispatch")
                    result = self.action_registry.dispatch(action, args)
                    end_timer(dispatch_timer, {"action": action})

                    # Remember the action
                    if hasattr(self, 'memory') and self.memory:
                        success = "error" not in result.lower()
                        self.memory.remember_action(action, args, result, success)

                    # Speak LLM reply (short) if provided; else result
                    out = reply or result
                    self.result = out
                    self.speak(out)
                else:
                    # Chat response
                    out = data.get("reply") or self.llm_service.generate_response(command)
                    self.result = out
                    self.speak(out)
            except Exception as e:
                self.logger.error(f"LLM structured error: {e}")
                fallback_response = self._get_fallback_response(command)
                self.result = fallback_response
                self.speak(fallback_response)
            finally:
                self.thinking = False

        threading.Thread(target=llm_task, daemon=True).start()

    def _get_fallback_response(self, command: str):
        """Provide simple fallback responses when LLM is not available"""
        command = command.lower()

        # Simple pattern matching for common queries
        if any(word in command for word in ["hello", "hi", "hey"]):
            return "Hello! How can I help you today?"
        elif any(word in command for word in ["how", "what", "why", "when", "where"]):
            return "How what? Please clarify your question."
        elif any(word in command for word in ["thank", "thanks"]):
            return "You're welcome!"
        elif any(word in command for word in ["weather"]):
            return (
                "I cannot check the weather right now. AI features are not available."
            )
        elif any(word in command for word in ["joke", "funny"]):
            return "I'd love to tell you a joke, but AI features are currently unavailable."
        else:
            return "I heard you, but I need AI features to be properly configured to understand complex requests."

    def _draw_status(
        self, img: np.ndarray, window_size: Tuple[int, int] = None
    ) -> np.ndarray:
        """Draw status information on image with dynamic sizing"""
        if window_size:
            width, height = window_size
        else:
            height, width = img.shape[:2]

        # Determine status and color
        if self.listening:
            status, color = "LISTENING...", Constants.UI_COLORS["LISTENING"]
        elif self.thinking:
            status, color = "THINKING...", Constants.UI_COLORS["THINKING"]
        elif self.is_speaking:
            status, color = "SPEAKING", Constants.UI_COLORS["SPEAKING"]
        else:
            status, color = "READY", Constants.UI_COLORS["READY"]

        # Dynamic font scaling
        font_scale = max(0.5, min(1.2, width / 1280))
        thickness = max(1, int(2 * font_scale))

        # Status background and text
        status_text = f"Status: {status}"
        text_size = cv2.getTextSize(status_text, Constants.FONT, font_scale, thickness)[
            0
        ]

        # Position status in top-right with margin
        margin = int(20 * font_scale)
        status_x = width - text_size[0] - margin
        status_y = int(40 * font_scale)

        # Draw background rectangle
        cv2.rectangle(
            img,
            (status_x - 10, status_y - 25),
            (status_x + text_size[0] + 10, status_y + 5),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            img,
            (status_x - 10, status_y - 25),
            (status_x + text_size[0] + 10, status_y + 5),
            color,
            2,
        )

        # Draw status text
        cv2.putText(
            img,
            status_text,
            (status_x, status_y),
            Constants.FONT,
            font_scale,
            color,
            thickness,
        )

        # Draw continuous listening indicator
        if self.config.continuous_listening and self.continuous_listening_active:
            cl_text = "Continuous Listening: ON"
            cl_size = cv2.getTextSize(cl_text, Constants.FONT, font_scale * 0.7, 1)[0]
            cl_x = width - cl_size[0] - margin
            cl_y = status_y + int(35 * font_scale)

            cv2.rectangle(
                img,
                (cl_x - 10, cl_y - 20),
                (cl_x + cl_size[0] + 10, cl_y + 5),
                (0, 50, 0),
                -1,
            )
            cv2.rectangle(
                img,
                (cl_x - 10, cl_y - 20),
                (cl_x + cl_size[0] + 10, cl_y + 5),
                Constants.UI_COLORS["READY"],
                1,
            )
            cv2.putText(
                img,
                cl_text,
                (cl_x, cl_y),
                Constants.FONT,
                font_scale * 0.7,
                Constants.UI_COLORS["READY"],
                1,
            )

        return img

    def _draw_listening_animation(
        self,
        img: np.ndarray,
        center_x: int,
        center_y: int,
        window_size: Tuple[int, int] = None,
    ) -> np.ndarray:
        """Draw animated circle during listening with dynamic sizing"""
        if not self.listening:
            return img

        if window_size:
            width, height = window_size
        else:
            height, width = img.shape[:2]

        # Scale animation based on window size
        scale_factor = min(width, height) / 720  # Based on default height

        # Animated pulsing circle with multiple rings
        base_radius = int(50 * scale_factor)
        pulse_amplitude = int(20 * scale_factor)
        pulse_frequency = 3

        current_time = time.time()

        # Multiple concentric circles for better effect
        for i in range(3):
            phase_offset = i * 0.5
            radius = base_radius + int(
                pulse_amplitude
                * abs(np.sin(current_time * pulse_frequency + phase_offset))
            )
            alpha = 1.0 - (i * 0.3)  # Fade outer circles

            color = tuple(int(c * alpha) for c in Constants.UI_COLORS["LISTENING"])
            thickness = max(1, int(3 * scale_factor) - i)

            cv2.circle(
                img,
                (center_x, center_y),
                radius + i * int(10 * scale_factor),
                color,
                thickness,
            )

        # Add inner filled circle
        inner_radius = int(15 * scale_factor)
        cv2.circle(
            img,
            (center_x, center_y),
            inner_radius,
            Constants.UI_COLORS["LISTENING"],
            -1,
        )
        cv2.circle(img, (center_x, center_y), inner_radius, (255, 255, 255), 2)

        return img

    def _draw_result_text(
        self, img: np.ndarray, center_y: int, window_size: Tuple[int, int] = None
    ) -> np.ndarray:
        """Draw wrapped result text with dynamic sizing"""
        if not self.result or self.listening:
            return img

        if window_size:
            width, height = window_size
        else:
            height, width = img.shape[:2]

        # Dynamic scaling
        font_scale = max(0.5, min(1.0, width / 1280))
        thickness = max(1, int(2 * font_scale))
        margin = int(40 * font_scale)

        # Enhanced text wrapping logic
        words = self.result.split(" ")
        lines = []
        current_line = ""
        max_width = width - (margin * 2)

        for word in words:
            test_line = current_line + word + " "
            text_size = cv2.getTextSize(
                test_line, Constants.FONT, font_scale, thickness
            )[0]

            if text_size[0] < max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "

        if current_line:
            lines.append(current_line.strip())

        # Calculate total text height
        line_height = int(35 * font_scale)
        total_height = len(lines) * line_height

        # Position text in lower portion with background
        start_y = max(
            center_y + int(120 * font_scale),
            height - total_height - int(60 * font_scale),
        )

        if lines:
            # Draw background panel for text
            panel_y = start_y - int(20 * font_scale)
            panel_height = total_height + int(40 * font_scale)

            # Gradient background
            overlay = img.copy()
            cv2.rectangle(
                overlay,
                (margin - 10, panel_y),
                (width - margin + 10, panel_y + panel_height),
                (30, 30, 30),
                -1,
            )
            cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

            # Border
            cv2.rectangle(
                img,
                (margin - 10, panel_y),
                (width - margin + 10, panel_y + panel_height),
                Constants.UI_COLORS["READY"],
                2,
            )

        # Draw text lines
        y_pos = start_y
        for i, line in enumerate(lines):
            if y_pos < height - int(20 * font_scale):
                # Draw shadow
                cv2.putText(
                    img,
                    line,
                    (margin + 2, y_pos + 2),
                    Constants.FONT,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                )
                # Draw main text
                cv2.putText(
                    img,
                    line,
                    (margin, y_pos),
                    Constants.FONT,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )
                y_pos += line_height

        return img

    def _process_gestures(
        self,
        img: np.ndarray,
        lm_list: List,
        center_x: int,
        center_y: int,
        window_size: Tuple[int, int] = None,
    ) -> bool:
        """Process hand gestures and return exit status with dynamic sizing"""
        should_exit = False

        if not lm_list:
            self.hover_start = None
            return should_exit

        if window_size:
            width, height = window_size
        else:
            height, width = img.shape[:2]

        # Scale activation radius based on window size
        scale_factor = min(width, height) / 720
        activation_radius = int(self.config.activation_radius * scale_factor)

        # Get finger tip position (index finger)
        index_tip = lm_list[8]
        # Convert normalized coordinates to pixel coordinates
        x, y = int(index_tip[1] * width), int(index_tip[2] * height)

        # Enhanced cursor with multiple layers
        cursor_size = max(8, int(15 * scale_factor))
        cv2.circle(img, (x, y), cursor_size + 5, (0, 0, 0), 2)  # Shadow
        cv2.circle(img, (x, y), cursor_size, Constants.UI_COLORS["READY"], 3)
        cv2.circle(img, (x, y), cursor_size - 5, (255, 255, 255), -1)
        cv2.circle(img, (x, y), cursor_size - 8, Constants.UI_COLORS["READY"], -1)

        # Dynamic return button size and position
        button_size = int(80 * scale_factor)
        margin = int(20 * scale_factor)
        return_rect = (margin, margin, margin + button_size, margin + button_size)

        # Check return button hover
        if is_point_in_rect(x, y, return_rect):
            # Draw hover effect
            cv2.rectangle(
                img,
                (return_rect[0] - 5, return_rect[1] - 5),
                (return_rect[2] + 5, return_rect[3] + 5),
                (0, 255, 255),
                3,
            )

            if self.hover_start is None:
                self.hover_start = time.time()
            elif time.time() - self.hover_start >= self.config.hover_timeout:
                should_exit = True
        else:
            # Check microphone activation (only if continuous listening is disabled)
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            if (
                distance < activation_radius
                and not self.listening
                and not self.is_speaking
                and not self.thinking
                and not self.config.continuous_listening
            ):
                # Draw activation zone
                cv2.circle(
                    img, (center_x, center_y), activation_radius, (0, 255, 255), 2
                )
                cv2.putText(
                    img,
                    "Release to activate",
                    (center_x - 80, center_y + activation_radius + 30),
                    Constants.FONT,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                self.start_listening()

            # Reset hover timer when not hovering over return button
            if not is_point_in_rect(x, y, return_rect):
                self.hover_start = None

        return should_exit

    def run(
        self, img: np.ndarray, lm_list: List, window_size: Tuple[int, int] = None
    ) -> Tuple[np.ndarray, bool]:
        """Main method to run the voice assistant interface with dynamic sizing support"""
        should_exit = False

        try:
            # Get window dimensions
            if window_size:
                width, height = window_size
            else:
                height, width = img.shape[:2]

            # Resize image to match window size if needed
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height))

            # Welcome message on first run
            if self.first_run:
                self.speak(
                    "Welcome to your enhanced personal assistant. How can I help you today?"
                )
                self.first_run = False

                # Start continuous listening if enabled
                if self.config.continuous_listening:
                    self.start_continuous_listening()

            # Calculate center coordinates based on current size
            center_x, center_y = width // 2, height // 2

            # Scale icon sizes based on window size
            scale_factor = min(width, height) / 720
            icon_size = (int(120 * scale_factor), int(120 * scale_factor))
            return_icon_size = (int(80 * scale_factor), int(80 * scale_factor))

            # Resize icons if needed
            mic_icon = (
                cv2.resize(self.mic_icon, icon_size)
                if self.mic_icon is not None
                else None
            )
            return_icon = (
                cv2.resize(self.return_icon, return_icon_size)
                if self.return_icon is not None
                else None
            )

            # Draw enhanced background
            self._draw_background(img, width, height)

            # Draw UI elements with dynamic positioning
            if mic_icon is not None:
                mic_x = center_x - icon_size[0] // 2
                mic_y = center_y - icon_size[1] // 2

                # Draw microphone background circle
                bg_radius = int(icon_size[0] * 0.7)
                cv2.circle(img, (center_x, center_y), bg_radius, (60, 60, 60), -1)
                cv2.circle(img, (center_x, center_y), bg_radius, (120, 120, 120), 3)

                img = overlay_image(img, mic_icon, mic_x, mic_y)

            if return_icon is not None:
                margin = int(20 * scale_factor)
                img = overlay_image(img, return_icon, margin, margin)

            # Draw status information
            img = self._draw_status(img, window_size)

            # Draw listening animation
            img = self._draw_listening_animation(img, center_x, center_y, window_size)

            # Draw result text
            img = self._draw_result_text(img, center_y, window_size)

            # Process hand gestures
            should_exit = self._process_gestures(
                img, lm_list, center_x, center_y, window_size
            )

            # Draw title
            self._draw_title(img, width, height, scale_factor)

        except Exception as e:
            self.logger.error(f"Error in main run loop: {e}")
            cv2.putText(
                img,
                "System Error - Check Logs",
                (50, 50),
                Constants.FONT,
                Constants.FONT_SCALE,
                Constants.UI_COLORS["ERROR"],
                Constants.FONT_THICKNESS,
            )

        return img, should_exit

    def _draw_background(self, img: np.ndarray, width: int, height: int):
        """Draw enhanced background with gradient effect"""
        # Create subtle gradient background
        overlay = img.copy()

        # Dark overlay
        cv2.rectangle(overlay, (0, 0), (width, height), (20, 20, 40), -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Add subtle grid pattern
        grid_spacing = max(50, min(100, width // 20))
        for x in range(0, width, grid_spacing):
            cv2.line(img, (x, 0), (x, height), (40, 40, 60), 1)
        for y in range(0, height, grid_spacing):
            cv2.line(img, (0, y), (width, y), (40, 40, 60), 1)

    def _draw_title(
        self, img: np.ndarray, width: int, height: int, scale_factor: float
    ):
        """Draw title with dynamic scaling"""
        title = "AI Voice Assistant"
        font_scale = max(0.8, min(2.0, scale_factor))
        thickness = max(2, int(3 * scale_factor))

        # Get text size for centering
        text_size = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
        )[0]
        text_x = (width - text_size[0]) // 2
        text_y = int(60 * scale_factor)

        # Draw title background
        cv2.rectangle(
            img,
            (text_x - 20, text_y - 35),
            (text_x + text_size[0] + 20, text_y + 10),
            (30, 30, 30),
            -1,
        )
        cv2.rectangle(
            img,
            (text_x - 20, text_y - 35),
            (text_x + text_size[0] + 20, text_y + 10),
            (100, 100, 100),
            2,
        )

        # Draw shadow
        cv2.putText(
            img,
            title,
            (text_x + 2, text_y + 2),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 1,
        )
        # Draw main text
        cv2.putText(
            img,
            title,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    def cleanup(self):
        """Cleanup resources and threads"""
        self.logger.info("Cleaning up Voice Assistant resources")

        # Cleanup Jarvis backends
        if hasattr(self, 'context_oracle') and self.context_oracle:
            self.context_oracle.stop_monitoring()

        if hasattr(self, 'wake_detector') and self.wake_detector:
            if hasattr(self.wake_detector, 'stop'):
                self.wake_detector.stop()

        if hasattr(self, 'memory') and self.memory:
            self.memory._save_memory()

        # Stop continuous listening
        if (
            hasattr(self, "continuous_listening_active")
            and self.continuous_listening_active
        ):
            self.stop_continuous_listening_mode()

        # Stop audio processing
        if hasattr(self, "audio_manager") and self.audio_manager:
            self.audio_manager.cleanup()

        # Wait for listening thread to complete
        if (
            hasattr(self, "listen_thread")
            and self.listen_thread
            and self.listen_thread.is_alive()
        ):
            self.listen_thread.join(timeout=2.0)

        self.logger.info("Voice Assistant cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
