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

    @classmethod
    def from_file(cls, config_path: str) -> "VoiceAssistantConfig":
        """Load configuration from JSON file"""
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            return cls(**config_data)
        except (FileNotFoundError, json.JSONDecodeError):
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

                # Set speech rate and volume
                self.sapi_engine.setProperty("rate", 180)  # Speed of speech
                self.sapi_engine.setProperty("volume", 0.9)  # Volume level (0.0 to 1.0)
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

        # Load configuration
        self.config = (
            VoiceAssistantConfig.from_file(config_path)
            if config_path
            else VoiceAssistantConfig()
        )

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
        logging.getLogger("comtypes").setLevel(logging.WARNING)

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
                    success = False

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

                # Recognize speech
                recognized_text, error_message = (
                    self.speech_recognizer.recognize_speech()
                )

                if recognized_text:
                    self.result = recognized_text
                    self.last_result_time = time.time()
                    self.speak("Got it")

                    # Process command immediately after "Got it" is spoken
                    # Wait for "Got it" to finish, then process
                    time.sleep(1.0)  # Give time for "Got it" to be said
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

            # Handle LLM queries
            self.logger.info("Delegating to LLM query handler")
            self._handle_llm_query(command)

        except Exception as e:
            self.logger.error(f"Command processing error: {e}")
            error_msg = "I encountered an error processing your request"
            self.result = error_msg
            self.speak(error_msg)

    def _handle_llm_query(self, command: str):
        """Handle LLM queries in separate thread"""
        if not self.llm_service or not self.llm_service.enabled:
            # Provide a simple fallback response instead of just an error
            fallback_response = self._get_fallback_response(command)
            self.result = fallback_response
            self.speak(fallback_response)
            return

        self.thinking = True
        self.result = "Processing your question..."
        self.speak("Let me think about that")

        def llm_task():
            try:
                response = self.llm_service.generate_response(command)
                self.result = response
                self.speak(response)
            except Exception as e:
                self.logger.error(f"LLM error: {e}")
                # Provide fallback response instead of generic error
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

    def _draw_status(self, img: np.ndarray) -> np.ndarray:
        """Draw status information on image"""
        # Determine status and color
        if self.listening:
            status, color = "LISTENING...", Constants.UI_COLORS["LISTENING"]
        elif self.thinking:
            status, color = "THINKING...", Constants.UI_COLORS["THINKING"]
        elif self.is_speaking:
            status, color = "SPEAKING", Constants.UI_COLORS["SPEAKING"]
        else:
            status, color = "READY", Constants.UI_COLORS["READY"]

        # Draw status text
        cv2.putText(
            img,
            f"Status: {status}",
            (img.shape[1] - 250, 40),
            Constants.FONT,
            Constants.FONT_SCALE,
            color,
            Constants.FONT_THICKNESS,
        )

        # Draw continuous listening indicator
        if self.config.continuous_listening and self.continuous_listening_active:
            cv2.putText(
                img,
                "Continuous Listening: ON",
                (img.shape[1] - 300, 70),
                Constants.FONT,
                0.5,
                Constants.UI_COLORS["READY"],
                1,
            )

        return img

    def _draw_listening_animation(
        self, img: np.ndarray, center_x: int, center_y: int
    ) -> np.ndarray:
        """Draw animated circle during listening"""
        if self.listening:
            # Animated pulsing circle
            base_radius = 50
            pulse_amplitude = 20
            pulse_frequency = 5
            radius = base_radius + int(
                pulse_amplitude * abs(np.sin(time.time() * pulse_frequency))
            )
            cv2.circle(
                img, (center_x, center_y), radius, Constants.UI_COLORS["LISTENING"], 3
            )

        return img

    def _draw_result_text(self, img: np.ndarray, center_y: int) -> np.ndarray:
        """Draw wrapped result text"""
        if not self.result or self.listening:
            return img

        # Text wrapping logic
        words = self.result.split(" ")
        lines = []
        current_line = ""
        max_width = img.shape[1] - 80  # Leave margin

        for word in words:
            test_line = current_line + word + " "
            text_size = cv2.getTextSize(
                test_line,
                Constants.FONT,
                Constants.FONT_SCALE,
                Constants.FONT_THICKNESS,
            )[0]

            if text_size[0] < max_width:
                current_line = test_line
            else:
                if current_line:  # Avoid empty lines
                    lines.append(current_line.strip())
                current_line = word + " "

        if current_line:
            lines.append(current_line.strip())

        # Draw text lines
        y_pos = center_y + 150
        line_height = 40

        for line in lines:
            if y_pos < img.shape[0] - 20:  # Stay within image bounds
                cv2.putText(
                    img,
                    line,
                    (40, y_pos),
                    Constants.FONT,
                    Constants.FONT_SCALE,
                    Constants.UI_COLORS["READY"],
                    Constants.FONT_THICKNESS,
                )
                y_pos += line_height

        return img

    def _process_gestures(
        self, img: np.ndarray, lm_list: List, center_x: int, center_y: int
    ) -> bool:
        """Process hand gestures and return exit status"""
        should_exit = False

        if not lm_list:
            self.hover_start = None
            return should_exit

        # Get finger tip position (index finger)
        index_tip = lm_list[8]
        x, y = index_tip[1], index_tip[2]

        # Draw cursor
        cv2.circle(img, (x, y), 10, Constants.UI_COLORS["READY"], cv2.FILLED)

        # Check return button hover
        return_rect = (20, 20, 100, 100)
        if is_point_in_rect(x, y, return_rect):
            if self.hover_start is None:
                self.hover_start = time.time()
            elif time.time() - self.hover_start >= self.config.hover_timeout:
                should_exit = True
        else:
            # Check microphone activation (only if continuous listening is disabled)
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            if (
                distance < self.config.activation_radius
                and not self.listening
                and not self.is_speaking
                and not self.thinking
                and not self.config.continuous_listening
            ):
                self.start_listening()

            # Reset hover timer when not hovering over return button
            if not is_point_in_rect(x, y, return_rect):
                self.hover_start = None

        return should_exit

    def run(self, img: np.ndarray, lm_list: List) -> Tuple[np.ndarray, bool]:
        """Main method to run the voice assistant interface"""
        should_exit = False

        try:
            # Welcome message on first run
            if self.first_run:
                self.speak(
                    "Welcome to your enhanced personal assistant. How can I help you today?"
                )
                self.first_run = False

                # Start continuous listening if enabled
                if self.config.continuous_listening:
                    self.start_continuous_listening()

            # Calculate center coordinates
            center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

            # Draw UI elements
            img = overlay_image(img, self.mic_icon, center_x - 60, center_y - 60)
            img = overlay_image(img, self.return_icon, 20, 20)

            # Draw status information
            img = self._draw_status(img)

            # Draw listening animation
            img = self._draw_listening_animation(img, center_x, center_y)

            # Draw result text
            img = self._draw_result_text(img, center_y)

            # Process hand gestures
            should_exit = self._process_gestures(img, lm_list, center_x, center_y)

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

    def cleanup(self):
        """Cleanup resources and threads"""
        self.logger.info("Cleaning up Voice Assistant resources")

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
