import cv2
import threading
import speech_recognition as sr
import webbrowser
import os
import time
import requests
import json
import numpy as np
from utils import load_icon, overlay_image, is_point_in_rect
from transformers import pipeline

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
        
        # Initialize local NLP model
        try:
            self.nlp = pipeline("text-generation", model="gpt2")
        except:
            self.nlp = None
            print("Local NLP model not available, using fallback")
        
    def start_listening(self):
        if self.listening:
            return
            
        self.listening = True
        self.result = ""
        
        def listen_thread():
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source)
            try:
                self.result = self.recognizer.recognize_google(audio)
                self.last_result_time = time.time()
                self.process_command()
            except sr.UnknownValueError:
                self.result = "Could not understand audio"
            except sr.RequestError:
                self.result = "API unavailable"
            self.listening = False
            
        self.thread = threading.Thread(target=listen_thread)
        self.thread.daemon = True
        self.thread.start()
        
    def process_command(self):
        command = self.result.lower()
        
        # Local app commands
        if "notepad" in command:
            os.system("notepad.exe")
        elif "calculator" in command:
            os.system("calc.exe")
        elif "explorer" in command:
            os.system("explorer")
        elif "search" in command:
            query = command.replace("search", "").strip()
            webbrowser.open(f"https://google.com/search?q={query}")
        elif "time" in command:
            self.result = time.strftime("%I:%M %p")
        elif "date" in command:
            self.result = time.strftime("%B %d, %Y")
        # NLP-based responses
        else:
            if self.nlp:
                response = self.nlp(command, max_length=50, num_return_sequences=1)
                self.result = response[0]['generated_text']
            else:
                # Fallback to OpenAI API (requires API key)
                try:
                    headers = {"Authorization": "Bearer YOUR_API_KEY"}
                    data = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": command}]}
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=data
                    )
                    if response.status_code == 200:
                        self.result = response.json()['choices'][0]['message']['content']
                    else:
                        self.result = "AI service unavailable"
                except:
                    self.result = "Command processed"
    
    def run(self, img, lm_list):
        should_exit = False
        hover_start = None
        
        # Draw UI
        center_x, center_y = img.shape[1]//2, img.shape[0]//2
        img = overlay_image(img, self.mic_icon, center_x - 60, center_y - 60)
        
        # Draw return button
        return_rect = (20, 20, 100, 100)
        img = overlay_image(img, self.return_icon, 20, 20)
        
        # Display result if recent
        if self.result and time.time() - self.last_result_time < 5:
            # Wrap text
            words = self.result.split(' ')
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + word + " "
                if cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] < img.shape[1] - 40:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word + " "
            lines.append(current_line)
            
            # Draw text
            y_pos = center_y + 100
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
                if hover_start is None:
                    hover_start = time.time()
                elif time.time() - hover_start >= 1.5:
                    should_exit = True
            else:
                hover_start = None
                
            # Activate mic when hovering over center
            distance = ((x - center_x)**2 + (y - center_y)**2)**0.5
            if distance < 100 and not self.listening:
                self.start_listening()
    
        return img, should_exit