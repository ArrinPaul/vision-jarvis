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
import playsound
import os
import webbrowser  # Add this at the top with other imports

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.listening = False
        self.result = ""
        self.thread = None
        self.last_result_time = 0
        self.mic_icon = load_icon("mic_icon.png")
        self.return_icon = load_icon("return_icon.png", (80, 80))
        self.llm_service = LLMService()
        self.is_speaking = False
        self.thinking = False        
        self.hover_start = None  # Add this line
        self.first_run = True
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.stop_audio = False
        self.system = SystemController()
        
        # Start audio processing thread
        self.start_audio_processor()
    
    def start_audio_processor(self):
        def audio_processor():
            while not self.stop_audio:
                try:
                    # Get audio file from queue with timeout
                    audio_file = self.audio_queue.get(timeout=0.5)
                    playsound.playsound(audio_file)
                    os.unlink(audio_file)  # Delete temp file after playing
                    self.audio_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Audio playback error: {e}")
        
        self.audio_thread = threading.Thread(target=audio_processor)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def speak(self, text):
        if not text or self.is_speaking:
            return
            
        self.is_speaking = True
        
        def tts_thread():
            try:
                # Create temp file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                    temp_filename = fp.name
                
                # Generate speech
                tts = gTTS(text=text, lang='en')
                tts.save(temp_filename)
                
                # Add to playback queue
                self.audio_queue.put(temp_filename)
            except Exception as e:
                print(f"TTS error: {e}")
            finally:
                # Set a timer to clear speaking state after estimated duration
                word_count = len(text.split())
                duration = max(2, word_count / 3)  # Estimate 3 words/second
                threading.Timer(duration, self.clear_speaking_state).start()
        
        threading.Thread(target=tts_thread).start()
    
    def clear_speaking_state(self):
        self.is_speaking = False
        self.thinking = False
    
    def start_listening(self):
        if self.listening or self.is_speaking:
            return
            
        self.listening = True
        self.result = ""
        
        def listen_thread():
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    self.result = self.recognizer.recognize_google(audio)
                    self.last_result_time = time.time()
                    
                    # Speak acknowledgment
                    self.speak("Got it")
                    self.process_command()
                except sr.WaitTimeoutError:
                    self.result = "I didn't hear anything"
                    self.speak("I didn't hear anything, please try again")
                except sr.UnknownValueError:
                    self.result = "Sorry, I didn't understand that"
                    self.speak("I didn't understand that, could you repeat?")
                except sr.RequestError as e:
                    self.result = f"Speech service error: {e}"
                    self.speak("I'm having trouble with the speech service")
                finally:
                    self.listening = False
            
        self.thread = threading.Thread(target=listen_thread)
        self.thread.daemon = True
        self.thread.start()
        
    def handle_system_command(self, command):
        # Add these search handlers before other command checks
        if "search for" in command:
            query = command.replace("search for", "").strip()
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"Searching Google for {query}"
        elif "google" in command:
            query = command.replace("google", "").strip()
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"Searching Google for {query}"

        # Application control
        if "open " in command:
            app = command.replace("open ", "").strip()
            return self.system.open_application(app)
        elif "close " in command:
            app = command.replace("close ", "").strip()
            return self.system.close_application(app)
        
        # Volume control
        elif "volume up" in command:
            return self.system.volume_up()
        elif "volume down" in command:
            return self.system.volume_down()
        elif "mute" in command:
            return self.system.mute()
        elif "set volume to " in command:
            level = command.replace("set volume to ", "").replace("%", "").strip()
            if level.isdigit():
                return self.system.set_volume(int(level))
            return "Please specify a number between 0 and 100"
        
        # Media control
        elif "play" in command or "pause" in command:
            return self.system.hotkey("playpause")
        elif "next" in command:
            return self.system.hotkey("nexttrack")
        elif "previous" in command:
            return self.system.hotkey("prevtrack")
        
        # Keyboard/mouse control
        elif "press " in command:
            key = command.replace("press ", "").strip()
            return self.system.press_key(key)
        elif "type " in command:
            text = command.replace("type ", "").strip()
            return self.system.type_text(text)
        elif "move mouse " in command:
            direction = command.replace("move mouse ", "").strip()
            return self.system.mouse_move(direction)
        elif "click" in command:
            button = "right" if "right" in command else "left"
            return self.system.click(button)
        elif "scroll " in command:
            direction = "down" if "down" in command else "up"
            return self.system.scroll(direction)
        
        # System commands
        elif "screenshot" in command:
            return self.system.take_screenshot()
        elif "lock" in command:
            return self.system.lock_pc()
        elif "shut down" in command:
            return self.system.shutdown()
        elif "restart" in command:
            return self.system.restart()
        
        return None
        
    def process_command(self):
        command = self.result.lower()
        print(f"Processing command: {command}")
        
        # First try system commands
        system_response = self.handle_system_command(command)
        if system_response:
            self.result = system_response
            self.speak(system_response)
            return
        
        # Handle special cases
        if "time" in command:
            time_str = time.strftime("%I:%M %p")
            self.result = f"Current time is {time_str}"
            self.speak(self.result)
        elif "date" in command:
            date_str = time.strftime("%B %d, %Y")
            self.result = f"Today is {date_str}"
            self.speak(self.result)
        # Handle knowledge-based queries with LLM
        else:
            self.thinking = True
            self.result = "Processing your question..."
            self.speak("Let me think about that")
            
            # Use LLM in a separate thread
            def llm_thread():
                try:
                    response = self.llm_service.generate_response(command)
                    self.result = response
                    self.speak(response)
                except Exception as e:
                    self.result = f"Error: {str(e)}"
                    self.speak("I encountered an error processing your request")
                    self.thinking = False
                
            threading.Thread(target=llm_thread).start()
    
    def run(self, img, lm_list):
        should_exit = False
        
        # Auto-start on first run
        if self.first_run:
            self.speak("Welcome to your personal assistant. How can I help you today?")
            self.first_run = False
        
        # Draw UI
        center_x, center_y = img.shape[1]//2, img.shape[0]//2
        img = overlay_image(img, self.mic_icon, center_x - 60, center_y - 60)
        
        # Draw return button
        return_rect = (20, 20, 100, 100)
        img = overlay_image(img, self.return_icon, 20, 20)
        
        # Draw status
        status = ""
        status_color = (0, 0, 255)
        if self.listening:
            status = "LISTENING..."
            status_color = (0, 0, 255)
        elif self.thinking:
            status = "THINKING..."
            status_color = (0, 255, 255)
        elif self.is_speaking:
            status = "SPEAKING"
            status_color = (0, 255, 0)
        else:
            status = "READY"
            status_color = (0, 255, 0)
            
        cv2.putText(img, f"Status: {status}", (img.shape[1] - 250, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw visual feedback for listening state
        if self.listening:
            radius = 50 + int(20 * abs(np.sin(time.time() * 5)))
            cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), 3)
        
        # Display result if available
        if self.result and not self.listening:
            # Wrap text
            words = self.result.split(' ')
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + word + " "
                text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                if text_size[0] < img.shape[1] - 40:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word + " "
            lines.append(current_line)
            
            # Draw text
            y_pos = center_y + 150
            for line in lines:
                cv2.putText(img, line, (40, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 40
        
        # Process hand gestures
        if lm_list:
            index_tip = lm_list[8]
            x, y = index_tip[1], index_tip[2]
            
            # Draw cursor
            cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)
            
            # Check return button
            if is_point_in_rect(x, y, return_rect):
                if self.hover_start is None:  # Use self.hover_start instead
                    self.hover_start = time.time()
                elif time.time() - self.hover_start >= 1.5:
                    should_exit = True
            else:
                self.hover_start = None
                
            # Activate mic when hovering over center
            distance = ((x - center_x)**2 + (y - center_y)**2)**0.5
            if (distance < 100 and not self.listening 
                and not self.is_speaking and not self.thinking):
                self.start_listening()
    
        return img, should_exit
    
    def __del__(self):
        self.stop_audio = True
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)