import pyaudio
import struct
import pvporcupine
import threading
import time

class WakeWordDetector:
    """
    Wake word detection using Porcupine engine.
    """
    def __init__(self, wake_word="jarvis", callback=None):
        self.wake_word = wake_word
        self.callback = callback
        self.is_listening = False
        self.porcupine = None
        self.audio_stream = None
        
    def start_listening(self):
        """Start wake word detection in a separate thread."""
        self.is_listening = True
        threading.Thread(target=self._listen_for_wake_word, daemon=True).start()
        
    def stop_listening(self):
        """Stop wake word detection."""
        self.is_listening = False
        if self.porcupine:
            self.porcupine.delete()
        if self.audio_stream:
            self.audio_stream.close()
            
    def _listen_for_wake_word(self):
        """Listen for wake word in background."""
        try:
            # Initialize Porcupine with built-in wake word
            self.porcupine = pvporcupine.create(keywords=["jarvis"])
            
            # Initialize audio stream
            pa = pyaudio.PyAudio()
            self.audio_stream = pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            
            print("Wake word detection started. Say 'Jarvis' to activate...")
            
            while self.is_listening:
                pcm = self.audio_stream.read(self.porcupine.frame_length)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    print("Wake word detected!")
                    if self.callback:
                        self.callback()
                    time.sleep(1)  # Prevent multiple detections
                        
        except Exception as e:
            print(f"Wake word detection error: {e}")
        finally:
            self.stop_listening()