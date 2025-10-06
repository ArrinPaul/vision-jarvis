import speech_recognition as sr
import pyttsx3
import openai
import threading
import time
from .wake_word_detector import WakeWordDetector
from .conversation_manager import ConversationManager

class VoiceAssistant:
    """
    Advanced Voice Assistant for natural language, context-aware conversation, and multi-turn dialogue.
    Features: Wake word detection, context memory, personality, and advanced TTS.
    """
    def __init__(self, openai_api_key=None, personality="friendly_assistant"):
        # Initialize core components
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.conversation_manager = ConversationManager()
        self.wake_word_detector = WakeWordDetector(callback=self.on_wake_word_detected)
        
        # Configure TTS
        self._configure_tts()
        
        # OpenAI setup
        if openai_api_key:
            openai.api_key = openai_api_key
            
        # Assistant state
        self.is_active = False
        self.personality = personality
        self.system_prompt = self._get_system_prompt()
        
    def _configure_tts(self):
        """Configure text-to-speech engine with personality."""
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Try to use a female voice if available
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
        
        # Set speech rate and volume
        self.tts_engine.setProperty('rate', 180)  # Speed
        self.tts_engine.setProperty('volume', 0.9)  # Volume
        
    def _get_system_prompt(self):
        """Get system prompt based on personality."""
        prompts = {
            "friendly_assistant": "You are JARVIS, Tony Stark's AI assistant. You are helpful, witty, sophisticated, and slightly sarcastic. You have a British accent in your responses. Keep responses concise but informative.",
            "professional": "You are a professional AI assistant. Provide clear, concise, and helpful responses.",
            "casual": "You are a casual, friendly AI assistant. Be conversational and approachable."
        }
        return prompts.get(self.personality, prompts["friendly_assistant"])
        
    def start_listening(self):
        """Start wake word detection."""
        self.wake_word_detector.start_listening()
        
    def stop_listening(self):
        """Stop wake word detection."""
        self.wake_word_detector.stop_listening()
        
    def on_wake_word_detected(self):
        """Callback when wake word is detected."""
        self.respond("Yes, how can I help you?")
        self.is_active = True
        threading.Thread(target=self.handle_conversation, daemon=True).start()
        
    def listen(self, timeout=5):
        """Capture and process user voice input with timeout."""
        try:
            with sr.Microphone() as source:
                print("Listening...")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            text = self.recognizer.recognize_google(audio)
            print(f"User said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("Listening timeout")
            return ""
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"Speech Recognition error: {e}")
            return ""

    def respond(self, text, emotion="neutral"):
        """Generate and speak a response with emotion."""
        print(f"JARVIS: {text}")
        
        # Adjust TTS based on emotion
        rate = 180
        if emotion == "excited":
            rate = 200
        elif emotion == "calm":
            rate = 160
        elif emotion == "urgent":
            rate = 220
            
        self.tts_engine.setProperty('rate', rate)
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def handle_conversation(self):
        """Handle a complete conversation session."""
        conversation_timeout = 30  # seconds of inactivity before ending session
        last_interaction = time.time()
        
        while self.is_active and (time.time() - last_interaction < conversation_timeout):
            user_input = self.listen(timeout=10)
            
            if not user_input:
                continue
                
            if user_input.lower() in ['goodbye', 'bye', 'exit', 'stop']:
                self.respond("Goodbye! I'll be here when you need me.")
                self.is_active = False
                break
                
            last_interaction = time.time()
            self.process_user_input(user_input)
            
        if self.is_active:
            self.respond("I'll be in standby mode. Say 'Hey Jarvis' to wake me up.")
            self.is_active = False

    def process_user_input(self, user_input):
        """Process user input and generate appropriate response."""
        # Add to conversation history
        self.conversation_manager.add_user_message(user_input)
        
        # Analyze intent
        intent = self.conversation_manager.analyze_intent(user_input)
        
        # Get context
        context = self.conversation_manager.get_context_prompt()
        history = self.conversation_manager.get_conversation_history(limit=5)
        
        try:
            # Prepare messages for OpenAI
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if context:
                messages.append({"role": "system", "content": context})
                
            # Add recent conversation history
            messages.extend(history)
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            
            assistant_reply = response.choices[0].message['content']
            
            # Add to conversation history
            self.conversation_manager.add_assistant_message(assistant_reply)
            
            # Determine emotion based on content
            emotion = self._detect_emotion(assistant_reply)
            
            # Respond with appropriate emotion
            self.respond(assistant_reply, emotion)
            
        except Exception as e:
            print(f"LLM error: {e}")
            error_responses = [
                "I'm experiencing some technical difficulties. Please try again.",
                "My neural networks are a bit tangled at the moment.",
                "I seem to be having a senior moment. Could you repeat that?"
            ]
            import random
            self.respond(random.choice(error_responses))
            
    def _detect_emotion(self, text):
        """Simple emotion detection based on text content."""
        text_lower = text.lower()
        if any(word in text_lower for word in ['!', 'exciting', 'great', 'amazing', 'wonderful']):
            return "excited"
        elif any(word in text_lower for word in ['urgent', 'immediately', 'quickly', 'emergency']):
            return "urgent"
        elif any(word in text_lower for word in ['calm', 'peaceful', 'relax', 'meditation']):
            return "calm"
        else:
            return "neutral"
            
    def set_personality(self, personality):
        """Change assistant personality."""
        self.personality = personality
        self.system_prompt = self._get_system_prompt()
        self.respond(f"Personality updated to {personality} mode.")
        
    def get_status(self):
        """Get assistant status."""
        return {
            "is_active": self.is_active,
            "personality": self.personality,
            "wake_word_listening": self.wake_word_detector.is_listening
        }
