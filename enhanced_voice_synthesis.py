"""
JARVIS Enhanced Voice Synthesis System
=====================================

This module provides advanced text-to-speech capabilities featuring:
- Multiple voice personalities and characters
- Emotional inflection and tone modulation  
- Real-time voice effects and filters
- Custom voice training and cloning
- Neural voice synthesis
- Audio processing and enhancement
- Voice response generation
- Contextual speech adaptation
- Multi-language support
- Iron Man JARVIS-style voice processing
"""

import pyttsx3
import azure.cognitiveservices.speech as speechsdk
import numpy as np
import soundfile as sf
import librosa
import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import threading
import queue
import time
import json
import os
import pickle
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import asyncio
import sounddevice as sd
from datetime import datetime
import re
import random

# Audio processing libraries
import scipy.signal
from scipy.io.wavfile import write as wav_write
import webrtcvad
import pyaudio
import wave

class VoicePersonality(Enum):
    """Voice personality types"""
    JARVIS_CLASSIC = "jarvis_classic"           # Original JARVIS - British, formal, intelligent
    JARVIS_FRIENDLY = "jarvis_friendly"         # Warmer, more conversational
    JARVIS_TECHNICAL = "jarvis_technical"       # Technical, precise, analytical
    FRIDAY = "friday"                           # FRIDAY - Irish accent, more casual
    EDITH = "edith"                             # EDITH - Confident, security-focused  
    STARK_AI = "stark_ai"                       # Tony Stark's personal AI
    PROFESSIONAL = "professional"              # Business/corporate voice
    CASUAL = "casual"                           # Relaxed, informal
    AUTHORITATIVE = "authoritative"            # Command-style, military
    CARING = "caring"                          # Warm, empathetic, supportive

class EmotionalTone(Enum):
    """Emotional tone modulations"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited" 
    CALM = "calm"
    CONCERNED = "concerned"
    URGENT = "urgent"
    CONFIDENT = "confident"
    APOLOGETIC = "apologetic"
    ENCOURAGING = "encouraging"
    SERIOUS = "serious"
    PLAYFUL = "playful"
    MYSTERIOUS = "mysterious"

class SpeechContext(Enum):
    """Speech context types"""
    GREETING = "greeting"
    COMMAND_RESPONSE = "command_response"
    ERROR_MESSAGE = "error_message"
    INFORMATION = "information"
    QUESTION = "question"
    CONFIRMATION = "confirmation"
    WARNING = "warning"
    SUCCESS = "success"
    THINKING = "thinking"
    GOODBYE = "goodbye"

@dataclass
class VoiceProfile:
    """Voice profile configuration"""
    personality: VoicePersonality
    base_pitch: float = 1.0          # Pitch multiplier
    speech_rate: float = 180         # Words per minute
    volume: float = 0.8              # Volume level (0-1)
    accent_strength: float = 0.5     # Accent intensity (0-1)
    formality_level: float = 0.7     # Formal vs casual (0-1)
    emotional_range: float = 0.6     # Emotional expression range
    voice_effects: Dict[str, float] = field(default_factory=dict)
    custom_phrases: Dict[str, str] = field(default_factory=dict)
    preferred_language: str = "en-US"

@dataclass
class SpeechRequest:
    """Speech synthesis request"""
    text: str
    personality: VoicePersonality = VoicePersonality.JARVIS_CLASSIC
    emotion: EmotionalTone = EmotionalTone.NEUTRAL
    context: SpeechContext = SpeechContext.INFORMATION
    priority: int = 1                # 1-5, higher = more urgent
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class AudioEffectProcessor:
    """Processes audio effects and filters"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.effects_chain = []
    
    def add_reverb(self, audio: np.ndarray, room_size: float = 0.3, damping: float = 0.5) -> np.ndarray:
        """Add reverb effect"""
        # Simple reverb using convolution with impulse response
        impulse_length = int(0.5 * self.sample_rate)  # 0.5 second reverb
        impulse = np.random.exponential(0.1, impulse_length)
        impulse *= np.linspace(1, 0, impulse_length)  # Decay
        impulse *= room_size
        
        # Apply convolution
        reverb_audio = scipy.signal.convolve(audio, impulse, mode='same')
        
        # Mix with original
        return audio * (1 - damping) + reverb_audio * damping
    
    def add_echo(self, audio: np.ndarray, delay_ms: float = 300, decay: float = 0.3) -> np.ndarray:
        """Add echo effect"""
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        
        if delay_samples >= len(audio):
            return audio
        
        echo_audio = np.zeros_like(audio)
        echo_audio[delay_samples:] = audio[:-delay_samples] * decay
        
        return audio + echo_audio
    
    def add_chorus(self, audio: np.ndarray, depth: float = 0.02, rate: float = 2.0) -> np.ndarray:
        """Add chorus effect"""
        t = np.linspace(0, len(audio) / self.sample_rate, len(audio))
        delay_variation = depth * np.sin(2 * np.pi * rate * t)
        
        # Create delayed versions with varying delay
        chorus_audio = np.zeros_like(audio)
        for i, delay in enumerate(delay_variation):
            delay_samples = int(delay * self.sample_rate)
            if 0 < delay_samples < len(audio) - i:
                chorus_audio[i] = audio[i - delay_samples]
        
        return (audio + chorus_audio * 0.3) / 1.3
    
    def apply_eq(self, audio: np.ndarray, bass_gain: float = 0, mid_gain: float = 0, 
                treble_gain: float = 0) -> np.ndarray:
        """Apply 3-band EQ"""
        # Simple EQ using butterworth filters
        nyquist = self.sample_rate / 2
        
        # Bass (80-300 Hz)
        if bass_gain != 0:
            sos_bass = scipy.signal.butter(2, [80/nyquist, 300/nyquist], 
                                         btype='band', output='sos')
            bass = scipy.signal.sosfilt(sos_bass, audio)
            audio = audio + bass * (bass_gain / 10)
        
        # Mid (300-3000 Hz)  
        if mid_gain != 0:
            sos_mid = scipy.signal.butter(2, [300/nyquist, 3000/nyquist], 
                                        btype='band', output='sos')
            mid = scipy.signal.sosfilt(sos_mid, audio)
            audio = audio + mid * (mid_gain / 10)
        
        # Treble (3000+ Hz)
        if treble_gain != 0:
            sos_treble = scipy.signal.butter(2, 3000/nyquist, 
                                           btype='high', output='sos')
            treble = scipy.signal.sosfilt(sos_treble, audio)
            audio = audio + treble * (treble_gain / 10)
        
        return np.clip(audio, -1, 1)
    
    def apply_robotic_effect(self, audio: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply robotic/AI voice effect"""
        # Ring modulation for robotic effect
        t = np.linspace(0, len(audio) / self.sample_rate, len(audio))
        carrier_freq = 30 + intensity * 70  # 30-100 Hz carrier
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        
        modulated = audio * carrier * intensity
        return audio * (1 - intensity) + modulated
    
    def apply_holographic_effect(self, audio: np.ndarray) -> np.ndarray:
        """Apply holographic/JARVIS-style effect"""
        # Combine multiple effects for holographic sound
        audio = self.add_reverb(audio, room_size=0.2, damping=0.3)
        audio = self.apply_eq(audio, bass_gain=-2, mid_gain=1, treble_gain=3)
        audio = self.apply_robotic_effect(audio, intensity=0.15)
        
        # Add subtle harmonic distortion
        harmonics = np.sin(2 * np.pi * 150 * np.linspace(0, len(audio) / self.sample_rate, len(audio)))
        audio = audio + harmonics * 0.05
        
        return np.clip(audio, -1, 1)

class NeuralVoiceSynthesizer:
    """Neural network-based voice synthesis"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False
        self.current_model = None
        self.processor = None
        self.vocoder = None
        
        # Try to load models
        self.load_models()
    
    def load_models(self):
        """Load neural TTS models"""
        try:
            # Load SpeechT5 models
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.current_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Move to device
            self.current_model.to(self.device)
            self.vocoder.to(self.device)
            
            self.models_loaded = True
            print("✅ Neural voice models loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Could not load neural models: {e}")
            print("   Falling back to traditional TTS")
            self.models_loaded = False
    
    def synthesize_neural(self, text: str, speaker_embeddings: Optional[torch.Tensor] = None) -> Optional[np.ndarray]:
        """Synthesize speech using neural models"""
        if not self.models_loaded:
            return None
        
        try:
            # Process text
            inputs = self.processor(text=text, return_tensors="pt")
            
            # Use default speaker if none provided
            if speaker_embeddings is None:
                # Load default speaker embeddings
                embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
            
            # Generate speech
            with torch.no_grad():
                speech = self.current_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder)
            
            # Convert to numpy
            audio = speech.cpu().numpy()
            return audio
            
        except Exception as e:
            print(f"Error in neural synthesis: {e}")
            return None
    
    def create_custom_voice(self, audio_samples: List[np.ndarray], sample_rate: int = 16000) -> torch.Tensor:
        """Create custom speaker embeddings from audio samples"""
        # This would implement speaker embedding extraction
        # For now, return a random embedding as placeholder
        return torch.randn(512)

class VoicePersonalityEngine:
    """Manages voice personalities and their characteristics"""
    
    def __init__(self):
        self.profiles = self.create_default_profiles()
        self.custom_profiles = {}
        self.phrase_templates = self.load_phrase_templates()
    
    def create_default_profiles(self) -> Dict[VoicePersonality, VoiceProfile]:
        """Create default voice profiles"""
        profiles = {}
        
        # JARVIS Classic - British, formal, intelligent
        profiles[VoicePersonality.JARVIS_CLASSIC] = VoiceProfile(
            personality=VoicePersonality.JARVIS_CLASSIC,
            base_pitch=0.85,
            speech_rate=165,
            volume=0.8,
            formality_level=0.9,
            emotional_range=0.4,
            voice_effects={"reverb": 0.2, "eq_treble": 2, "holographic": 0.3},
            custom_phrases={
                "greeting": "Good {time_of_day}, Mr. Stark. How may I assist you today?",
                "acknowledgment": "Of course, sir.",
                "thinking": "Let me process that for you...",
                "error": "I apologize, but I encountered an issue.",
                "success": "Task completed successfully, sir."
            }
        )
        
        # JARVIS Friendly - Warmer, more conversational
        profiles[VoicePersonality.JARVIS_FRIENDLY] = VoiceProfile(
            personality=VoicePersonality.JARVIS_FRIENDLY,
            base_pitch=0.9,
            speech_rate=175,
            volume=0.85,
            formality_level=0.6,
            emotional_range=0.7,
            voice_effects={"reverb": 0.15, "eq_mid": 1, "holographic": 0.2},
            custom_phrases={
                "greeting": "Hello there! Ready to get things done?",
                "acknowledgment": "Absolutely!",
                "thinking": "Give me just a moment to figure this out...",
                "error": "Oops, something went wrong there.",
                "success": "Great! We got it done!"
            }
        )
        
        # FRIDAY - Irish accent, casual
        profiles[VoicePersonality.FRIDAY] = VoiceProfile(
            personality=VoicePersonality.FRIDAY,
            base_pitch=1.1,
            speech_rate=185,
            volume=0.9,
            formality_level=0.4,
            emotional_range=0.8,
            voice_effects={"chorus": 0.1, "eq_bass": 1},
            custom_phrases={
                "greeting": "Hey there, boss! What's the plan?",
                "acknowledgment": "You got it!",
                "thinking": "Hmm, let me have a look at this...",
                "error": "Well, that didn't go as expected.",
                "success": "Boom! Nailed it!"
            }
        )
        
        # Technical JARVIS - Precise, analytical
        profiles[VoicePersonality.JARVIS_TECHNICAL] = VoiceProfile(
            personality=VoicePersonality.JARVIS_TECHNICAL,
            base_pitch=0.8,
            speech_rate=155,
            volume=0.75,
            formality_level=0.95,
            emotional_range=0.3,
            voice_effects={"robotic": 0.1, "eq_treble": 3, "holographic": 0.4},
            custom_phrases={
                "greeting": "System initialized. Awaiting input parameters.",
                "acknowledgment": "Command acknowledged. Executing.",
                "thinking": "Processing data... analyzing patterns...",
                "error": "Error detected. Initiating diagnostic protocols.",
                "success": "Operation completed within normal parameters."
            }
        )
        
        return profiles
    
    def load_phrase_templates(self) -> Dict[str, List[str]]:
        """Load contextual phrase templates"""
        return {
            "time_greeting": [
                "Good morning", "Good afternoon", "Good evening", "Hello"
            ],
            "acknowledgment": [
                "Understood", "Of course", "Certainly", "Right away", "Consider it done"
            ],
            "thinking": [
                "Let me think about that", "Processing...", "Give me a moment",
                "Analyzing the situation", "Working on it"
            ],
            "error_polite": [
                "I apologize for the inconvenience", "Something seems to have gone wrong",
                "I'm having difficulty with that", "There appears to be an issue"
            ],
            "success_confident": [
                "Task completed successfully", "Done!", "All set", "Perfect!",
                "Mission accomplished"
            ]
        }
    
    def get_profile(self, personality: VoicePersonality) -> VoiceProfile:
        """Get voice profile for personality"""
        return self.profiles.get(personality, self.profiles[VoicePersonality.JARVIS_CLASSIC])
    
    def customize_phrase(self, text: str, profile: VoiceProfile, context: SpeechContext, 
                        emotion: EmotionalTone) -> str:
        """Customize phrase based on personality and context"""
        # Apply personality-specific customizations
        if context in profile.custom_phrases:
            template = profile.custom_phrases[context]
            
            # Replace placeholders
            current_time = datetime.now()
            hour = current_time.hour
            if hour < 12:
                time_of_day = "morning"
            elif hour < 18:
                time_of_day = "afternoon"
            else:
                time_of_day = "evening"
            
            text = template.replace("{time_of_day}", time_of_day)
        
        # Apply emotional modifications
        if emotion == EmotionalTone.EXCITED:
            if profile.personality in [VoicePersonality.FRIDAY, VoicePersonality.JARVIS_FRIENDLY]:
                text = text + "!"
        elif emotion == EmotionalTone.APOLOGETIC:
            if "sorry" not in text.lower() and "apologize" not in text.lower():
                text = "I'm sorry, but " + text.lower()
        elif emotion == EmotionalTone.CONFIDENT:
            if profile.personality == VoicePersonality.JARVIS_TECHNICAL:
                text = text.replace("I think", "I am certain")
                text = text.replace("maybe", "definitely")
        
        return text

class EnhancedTTSEngine:
    """Enhanced Text-to-Speech engine with multiple backends"""
    
    def __init__(self):
        # Initialize TTS engines
        self.pyttsx_engine = None
        self.azure_speech_config = None
        self.neural_synthesizer = NeuralVoiceSynthesizer()
        self.audio_processor = AudioEffectProcessor()
        
        self.init_pyttsx()
        self.init_azure_tts()
        
        # Audio settings
        self.sample_rate = 22050
        self.channels = 1
        
    def init_pyttsx(self):
        """Initialize pyttsx3 engine"""
        try:
            self.pyttsx_engine = pyttsx3.init()
            
            # Set default properties
            self.pyttsx_engine.setProperty('rate', 180)
            self.pyttsx_engine.setProperty('volume', 0.8)
            
            # Get available voices
            voices = self.pyttsx_engine.getProperty('voices')
            if voices:
                # Prefer a voice with "david" or "male" in the name for JARVIS
                for voice in voices:
                    if any(keyword in voice.name.lower() for keyword in ['david', 'male', 'daniel']):
                        self.pyttsx_engine.setProperty('voice', voice.id)
                        break
            
            print("✅ Pyttsx3 TTS engine initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize pyttsx3: {e}")
    
    def init_azure_tts(self):
        """Initialize Azure Cognitive Services TTS"""
        try:
            # You would set your Azure subscription key here
            subscription_key = os.getenv('AZURE_SPEECH_KEY')
            region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
            
            if subscription_key:
                self.azure_speech_config = speechsdk.SpeechConfig(
                    subscription=subscription_key, region=region
                )
                self.azure_speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
                print("✅ Azure TTS initialized")
            else:
                print("⚠️ Azure TTS not configured (missing API key)")
        except Exception as e:
            print(f"⚠️ Could not initialize Azure TTS: {e}")
    
    def synthesize_with_pyttsx(self, text: str, profile: VoiceProfile) -> Optional[np.ndarray]:
        """Synthesize speech using pyttsx3"""
        if not self.pyttsx_engine:
            return None
        
        try:
            # Set voice properties
            self.pyttsx_engine.setProperty('rate', int(profile.speech_rate))
            self.pyttsx_engine.setProperty('volume', profile.volume)
            
            # Save to temporary file
            temp_file = "temp_speech.wav"
            self.pyttsx_engine.save_to_file(text, temp_file)
            self.pyttsx_engine.runAndWait()
            
            # Load audio data
            if os.path.exists(temp_file):
                audio_data, sr = librosa.load(temp_file, sr=self.sample_rate)
                os.remove(temp_file)
                return audio_data
            
        except Exception as e:
            print(f"Error in pyttsx synthesis: {e}")
        
        return None
    
    def synthesize_with_azure(self, text: str, profile: VoiceProfile) -> Optional[np.ndarray]:
        """Synthesize speech using Azure TTS"""
        if not self.azure_speech_config:
            return None
        
        try:
            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.azure_speech_config)
            
            # Build SSML for voice customization
            ssml = self.build_ssml(text, profile)
            
            # Synthesize
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Convert to numpy array
                audio_data = np.frombuffer(result.audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Resample if needed
                if len(audio_data) > 0:
                    # Assume Azure returns 16kHz, resample to our sample rate
                    audio_data = librosa.resample(audio_data, orig_sr=16000, target_sr=self.sample_rate)
                    return audio_data
            
        except Exception as e:
            print(f"Error in Azure synthesis: {e}")
        
        return None
    
    def build_ssml(self, text: str, profile: VoiceProfile) -> str:
        """Build SSML for advanced voice control"""
        # Map personalities to Azure voices
        voice_mapping = {
            VoicePersonality.JARVIS_CLASSIC: "en-GB-RyanNeural",
            VoicePersonality.JARVIS_FRIENDLY: "en-US-BrandonNeural", 
            VoicePersonality.FRIDAY: "en-IE-ConnorNeural",
            VoicePersonality.JARVIS_TECHNICAL: "en-US-DavisNeural",
            VoicePersonality.PROFESSIONAL: "en-US-AndrewNeural",
            VoicePersonality.AUTHORITATIVE: "en-US-BrianNeural"
        }
        
        voice_name = voice_mapping.get(profile.personality, "en-US-BrandonNeural")
        
        # Calculate prosody values
        rate_percent = int((profile.speech_rate - 180) / 180 * 100)
        pitch_percent = int((profile.base_pitch - 1.0) * 100)
        volume_percent = int(profile.volume * 100)
        
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="{voice_name}">
                <prosody rate="{rate_percent:+d}%" pitch="{pitch_percent:+d}%" volume="{volume_percent}%">
                    {text}
                </prosody>
            </voice>
        </speak>
        """
        
        return ssml.strip()
    
    def apply_personality_effects(self, audio: np.ndarray, profile: VoiceProfile) -> np.ndarray:
        """Apply personality-specific audio effects"""
        processed_audio = audio.copy()
        
        # Apply effects based on profile
        effects = profile.voice_effects
        
        if "reverb" in effects:
            processed_audio = self.audio_processor.add_reverb(
                processed_audio, room_size=effects["reverb"]
            )
        
        if "echo" in effects:
            processed_audio = self.audio_processor.add_echo(
                processed_audio, delay_ms=300, decay=effects["echo"]
            )
        
        if "chorus" in effects:
            processed_audio = self.audio_processor.add_chorus(
                processed_audio, depth=effects["chorus"]
            )
        
        if "robotic" in effects:
            processed_audio = self.audio_processor.apply_robotic_effect(
                processed_audio, intensity=effects["robotic"]
            )
        
        if "holographic" in effects:
            processed_audio = self.audio_processor.apply_holographic_effect(processed_audio)
        
        # Apply EQ
        bass_gain = effects.get("eq_bass", 0)
        mid_gain = effects.get("eq_mid", 0) 
        treble_gain = effects.get("eq_treble", 0)
        
        if any([bass_gain, mid_gain, treble_gain]):
            processed_audio = self.audio_processor.apply_eq(
                processed_audio, bass_gain, mid_gain, treble_gain
            )
        
        return processed_audio

class JarvisVoiceSynthesis:
    """Main JARVIS Voice Synthesis System"""
    
    def __init__(self):
        self.tts_engine = EnhancedTTSEngine()
        self.personality_engine = VoicePersonalityEngine()
        
        # Speech queue and processing
        self.speech_queue = queue.PriorityQueue()
        self.is_speaking = False
        self.speech_thread = None
        self.is_running = False
        
        # Audio playback
        self.audio_stream = None
        self.playback_device = None
        
        # Voice learning and adaptation
        self.conversation_history = deque(maxlen=100)
        self.user_preferences = {}
        
        print("🎤 JARVIS Voice Synthesis System initialized")
        self.start_speech_processor()
    
    def start_speech_processor(self):
        """Start speech processing thread"""
        self.is_running = True
        self.speech_thread = threading.Thread(target=self._speech_processor_loop, daemon=True)
        self.speech_thread.start()
    
    def _speech_processor_loop(self):
        """Main speech processing loop"""
        while self.is_running:
            try:
                # Get next speech request (priority queue)
                priority, request = self.speech_queue.get(timeout=1)
                
                if request is None:  # Shutdown signal
                    break
                
                self._process_speech_request(request)
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in speech processor: {e}")
    
    def _process_speech_request(self, request: SpeechRequest):
        """Process individual speech request"""
        try:
            self.is_speaking = True
            
            # Get voice profile
            profile = self.personality_engine.get_profile(request.personality)
            
            # Customize text based on personality and context
            processed_text = self.personality_engine.customize_phrase(
                request.text, profile, request.context, request.emotion
            )
            
            # Apply emotional modulation to profile
            modified_profile = self._apply_emotional_modulation(profile, request.emotion)
            
            # Synthesize speech (try multiple engines)
            audio_data = None
            
            # Try neural synthesis first
            if self.tts_engine.neural_synthesizer.models_loaded:
                audio_data = self.tts_engine.neural_synthesizer.synthesize_neural(processed_text)
            
            # Fall back to Azure TTS
            if audio_data is None and self.tts_engine.azure_speech_config:
                audio_data = self.tts_engine.synthesize_with_azure(processed_text, modified_profile)
            
            # Fall back to pyttsx3
            if audio_data is None:
                audio_data = self.tts_engine.synthesize_with_pyttsx(processed_text, modified_profile)
            
            if audio_data is not None:
                # Apply personality effects
                audio_data = self.tts_engine.apply_personality_effects(audio_data, modified_profile)
                
                # Play audio
                self._play_audio(audio_data)
                
                # Execute callback if provided
                if request.callback:
                    request.callback(True, processed_text)
                
                # Update conversation history
                self.conversation_history.append({
                    'text': processed_text,
                    'personality': request.personality,
                    'emotion': request.emotion,
                    'timestamp': datetime.now()
                })
            else:
                print(f"Failed to synthesize: {processed_text}")
                if request.callback:
                    request.callback(False, processed_text)
        
        except Exception as e:
            print(f"Error processing speech request: {e}")
            if request.callback:
                request.callback(False, str(e))
        finally:
            self.is_speaking = False
    
    def _apply_emotional_modulation(self, profile: VoiceProfile, emotion: EmotionalTone) -> VoiceProfile:
        """Apply emotional modulation to voice profile"""
        modified_profile = VoiceProfile(
            personality=profile.personality,
            base_pitch=profile.base_pitch,
            speech_rate=profile.speech_rate,
            volume=profile.volume,
            accent_strength=profile.accent_strength,
            formality_level=profile.formality_level,
            emotional_range=profile.emotional_range,
            voice_effects=profile.voice_effects.copy(),
            custom_phrases=profile.custom_phrases.copy()
        )
        
        # Modify based on emotion
        if emotion == EmotionalTone.EXCITED:
            modified_profile.speech_rate *= 1.1
            modified_profile.base_pitch *= 1.05
            modified_profile.volume = min(1.0, modified_profile.volume * 1.1)
        
        elif emotion == EmotionalTone.CALM:
            modified_profile.speech_rate *= 0.9
            modified_profile.base_pitch *= 0.95
            modified_profile.voice_effects["reverb"] = 0.3
        
        elif emotion == EmotionalTone.URGENT:
            modified_profile.speech_rate *= 1.2
            modified_profile.base_pitch *= 1.1
            modified_profile.volume = min(1.0, modified_profile.volume * 1.2)
        
        elif emotion == EmotionalTone.SERIOUS:
            modified_profile.speech_rate *= 0.85
            modified_profile.base_pitch *= 0.9
            modified_profile.voice_effects["eq_bass"] = 2
        
        elif emotion == EmotionalTone.PLAYFUL:
            modified_profile.base_pitch *= 1.1
            modified_profile.voice_effects["chorus"] = 0.1
        
        elif emotion == EmotionalTone.APOLOGETIC:
            modified_profile.speech_rate *= 0.9
            modified_profile.volume *= 0.9
            modified_profile.base_pitch *= 0.95
        
        return modified_profile
    
    def _play_audio(self, audio_data: np.ndarray):
        """Play audio data"""
        try:
            # Normalize audio
            audio_data = np.clip(audio_data, -1, 1)
            
            # Play using sounddevice
            sd.play(audio_data, samplerate=self.tts_engine.sample_rate)
            sd.wait()  # Wait until playback is finished
            
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def speak(self, text: str, personality: VoicePersonality = VoicePersonality.JARVIS_CLASSIC,
             emotion: EmotionalTone = EmotionalTone.NEUTRAL, 
             context: SpeechContext = SpeechContext.INFORMATION,
             priority: int = 1, callback: Optional[Callable] = None) -> bool:
        """Queue text for speech synthesis"""
        try:
            request = SpeechRequest(
                text=text,
                personality=personality,
                emotion=emotion,
                context=context,
                priority=priority,
                callback=callback
            )
            
            # Add to priority queue (lower number = higher priority)
            self.speech_queue.put((6 - priority, request))
            return True
            
        except Exception as e:
            print(f"Error queuing speech: {e}")
            return False
    
    def speak_async(self, text: str, **kwargs) -> asyncio.Future:
        """Asynchronous speech synthesis"""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        def callback(success: bool, processed_text: str):
            if success:
                future.set_result(processed_text)
            else:
                future.set_exception(Exception(f"Speech synthesis failed: {processed_text}"))
        
        kwargs['callback'] = callback
        success = self.speak(text, **kwargs)
        
        if not success:
            future.set_exception(Exception("Failed to queue speech request"))
        
        return future
    
    def interrupt_speech(self):
        """Interrupt current speech"""
        try:
            sd.stop()  # Stop current playback
            
            # Clear queue except high priority items
            temp_queue = queue.PriorityQueue()
            while not self.speech_queue.empty():
                try:
                    priority, request = self.speech_queue.get_nowait()
                    if priority <= 2:  # Keep high priority items
                        temp_queue.put((priority, request))
                except queue.Empty:
                    break
            
            self.speech_queue = temp_queue
            print("Speech interrupted")
            
        except Exception as e:
            print(f"Error interrupting speech: {e}")
    
    def set_user_preference(self, key: str, value: Any):
        """Set user preference for voice adaptation"""
        self.user_preferences[key] = value
        
        # Apply common preferences
        if key == "preferred_personality":
            # This could be used to adapt default personality
            pass
        elif key == "speech_speed_preference":
            # Adjust base speech rates
            pass
    
    def get_voice_status(self) -> Dict[str, Any]:
        """Get current voice system status"""
        return {
            "is_speaking": self.is_speaking,
            "queue_size": self.speech_queue.qsize(),
            "available_engines": {
                "pyttsx3": self.tts_engine.pyttsx_engine is not None,
                "azure": self.tts_engine.azure_speech_config is not None,
                "neural": self.tts_engine.neural_synthesizer.models_loaded
            },
            "personalities": list(VoicePersonality),
            "emotions": list(EmotionalTone),
            "contexts": list(SpeechContext)
        }
    
    def save_voice_profile(self, personality: VoicePersonality, filename: str):
        """Save custom voice profile"""
        profile = self.personality_engine.get_profile(personality)
        with open(filename, 'wb') as f:
            pickle.dump(profile, f)
        print(f"Voice profile saved: {filename}")
    
    def load_voice_profile(self, filename: str) -> Optional[VoicePersonality]:
        """Load custom voice profile"""
        try:
            with open(filename, 'rb') as f:
                profile = pickle.load(f)
            
            # Create custom personality enum value
            custom_personality = VoicePersonality.CUSTOM
            self.personality_engine.profiles[custom_personality] = profile
            
            print(f"Voice profile loaded: {filename}")
            return custom_personality
            
        except Exception as e:
            print(f"Error loading voice profile: {e}")
            return None
    
    def shutdown(self):
        """Shutdown voice synthesis system"""
        print("🔇 Shutting down Voice Synthesis System...")
        
        self.is_running = False
        
        # Signal speech processor to stop
        self.speech_queue.put((0, None))
        
        # Wait for speech thread to finish
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=2)
        
        # Stop any ongoing playback
        try:
            sd.stop()
        except:
            pass
        
        print("✅ Voice Synthesis System shutdown complete")

# Example usage and personality demonstrations
def demo_personalities():
    """Demonstrate different voice personalities"""
    voice_system = JarvisVoiceSynthesis()
    
    test_phrases = [
        ("Welcome to the JARVIS voice system", SpeechContext.GREETING),
        ("System diagnostics completed successfully", SpeechContext.SUCCESS),
        ("I'm sorry, but I couldn't complete that task", SpeechContext.ERROR_MESSAGE),
        ("Analyzing the data now, please wait", SpeechContext.INFORMATION),
        ("Would you like me to proceed with the operation?", SpeechContext.QUESTION)
    ]
    
    personalities = [
        VoicePersonality.JARVIS_CLASSIC,
        VoicePersonality.JARVIS_FRIENDLY, 
        VoicePersonality.FRIDAY,
        VoicePersonality.JARVIS_TECHNICAL
    ]
    
    emotions = [EmotionalTone.NEUTRAL, EmotionalTone.CONFIDENT, EmotionalTone.CONCERNED]
    
    print("🎭 Demonstrating JARVIS voice personalities...")
    
    for personality in personalities:
        print(f"\n🎪 {personality.value.upper()} Personality:")
        
        for i, (text, context) in enumerate(test_phrases):
            emotion = emotions[i % len(emotions)]
            print(f"   Speaking: '{text}' ({context.value}, {emotion.value})")
            
            voice_system.speak(
                text=text,
                personality=personality,
                emotion=emotion,
                context=context,
                priority=3
            )
            
            # Wait for speech to complete
            time.sleep(3)
    
    # Show system status
    status = voice_system.get_voice_status()
    print(f"\n📊 Voice System Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Cleanup
    time.sleep(2)
    voice_system.shutdown()

if __name__ == "__main__":
    demo_personalities()