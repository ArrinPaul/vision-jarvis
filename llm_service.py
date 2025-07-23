import os
import openai
from dotenv import load_dotenv
import time
import re

load_dotenv()

class LLMService:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.last_used = 0
        self.last_query = ""
        self.last_response = ""
        openai.api_key = self.openai_api_key
    
    def generate_response(self, query):
        # Cache responses to avoid repeated API calls
        if query == self.last_query and time.time() - self.last_used < 60:
            return self.last_response
        
        self.last_query = query
        self.last_used = time.time()
        
        try:
            # Use more specific prompt for system commands
            prompt = f"""
            You are an AI assistant that controls a Windows computer. 
            The user can ask you to perform system actions or answer questions.
            For system commands, respond with just the command text in the format: 
            COMMAND: [command text]
            
            Available commands:
            - Open applications: "open [app name]"
            - Close applications: "close [app name]"
            - Volume control: "volume up", "volume down", "mute", "set volume to [0-100]%"
            - Media control: "play/pause", "next track", "previous track"
            - Keyboard: "press [key]", "type [text]"
            - Mouse: "move mouse [up/down/left/right]", "click", "right click", "scroll up/down"
            - System: "take screenshot", "lock pc", "shut down", "restart"
            
            If the user asks a general question, answer normally.
            
            User: {query}
            Assistant:
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=300,
                temperature=0.3
            )
            self.last_response = response.choices[0].message.content.strip()
            
            # Extract command if present
            command_match = re.search(r"COMMAND:\s*(.+)", self.last_response)
            if command_match:
                return command_match.group(1)
                
            return self.last_response
        except openai.error.RateLimitError:
            return "I'm getting too many requests. Please try again in a moment."
        except openai.error.AuthenticationError:
            return "Authentication error. Please check API configuration."
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "I'm having trouble connecting to the knowledge service. Please try again later."