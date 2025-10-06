import os
import google.generativeai as genai
from dotenv import load_dotenv
import time
import re

load_dotenv()


class LLMService:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.last_used = 0
        self.last_query = ""
        self.last_response = ""
        self.model = None
        self.enabled = False

        # Try to initialize Gemini client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Gemini client with error handling"""
        try:
            if not self.gemini_api_key:
                print("GEMINI_API_KEY not found in environment variables")
                print("Please set GEMINI_API_KEY in your .env file")
                self.enabled = False
                return

            # Configure the Gemini API
            genai.configure(api_key=self.gemini_api_key)

            # Initialize the model (using Gemini 1.5 Flash)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            self.enabled = True
            print("Gemini client initialized successfully")

        except Exception as e:
            print(f"Failed to initialize Gemini client: {e}")
            self.model = None
            self.enabled = False

    def generate_response(self, query):
        # Check if client is available
        if not self.enabled or not self.model:
            return "I'm sorry, the AI service is not available right now. Please check your internet connection and API configuration."

        # Cache responses to avoid repeated API calls
        if query == self.last_query and time.time() - self.last_used < 60:
            return self.last_response

        self.last_query = query
        self.last_used = time.time()

        try:
            # Create a more specific prompt for system commands
            prompt = f"""You are an AI assistant that controls a Windows computer.
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

If the user asks a general question, answer normally and keep it concise.

User: {query}
Assistant:"""

            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            self.last_response = response.text.strip()

            # Extract command if present
            command_match = re.search(r"COMMAND:\s*(.+)", self.last_response)
            if command_match:
                return command_match.group(1)

            return self.last_response

        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "rate limit" in error_str:
                return "I'm getting too many requests. Please try again in a moment."
            elif "api key" in error_str or "authentication" in error_str:
                return "Authentication error. Please check your Gemini API key configuration."
            elif "blocked" in error_str or "safety" in error_str:
                return "I cannot process that request due to safety filters. Please try rephrasing your question."
            else:
                print(f"Gemini API error: {e}")
                return "I'm having trouble connecting to the AI service. Please try again later."

    def generate_structured(self, query: str) -> dict:
        """Return structured output for command vs chat.
        Schema: {"type": "command"|"chat", "action": str|None, "args": dict, "reply": str}
        """
        if not self.enabled or not self.model:
            return {"type": "chat", "action": None, "args": {}, "reply": self.generate_response(query)}

        sys_prompt = (\
            """
You are an AI assistant controlling a Windows PC. When a user asks for an action, reply ONLY with JSON in this schema:
{
  "type": "command" | "chat",
  "action": "open|close|volume|mute|set_volume|playpause|next|previous|press|type|mouse_move|click|scroll|screenshot|lock|search|google",
  "args": { },
  "reply": "Short natural-language reply"
}
Rules:
- If it's a command, set type="command" and fill action+args. Keep reply short.
- If it's general conversation, set type="chat" and provide reply; leave action null.
- Examples:
  User: open notepad -> {"type":"command","action":"open","args":{"app":"notepad"},"reply":"Opening Notepad"}
  User: set volume to 40% -> {"type":"command","action":"set_volume","args":{"level":40},"reply":"Setting volume to 40%"}
  User: what's the time -> {"type":"chat","action":null,"args":{},"reply":"It's 3:14 PM."}
"""


        )
        try:
            response = self.model.generate_content(sys_prompt + "\nUser: " + query + "\nAssistant:")
            text = (response.text or "").strip()

            # Extract first JSON block
            import json, re
            m = re.search(r"\{[\s\S]*?\}", text)
            if not m:
                return {"type": "chat", "action": None, "args": {}, "reply": text or "I didn't understand that request."}

            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return {"type": "chat", "action": None, "args": {}, "reply": text or "I had trouble parsing that request."}

            # Normalize and validate
            if not isinstance(data, dict):
                return {"type": "chat", "action": None, "args": {}, "reply": text}

            data.setdefault("type", "chat")
            data.setdefault("action", None)
            data.setdefault("args", {})
            data.setdefault("reply", "")

            # Basic safety: only allow known actions
            allowed = {"open","close","volume","mute","set_volume","playpause","next","previous","press","type","mouse_move","click","scroll","screenshot","lock","search","google","windows_tile","system_status","calendar","routine","briefing","enable_briefings"}
            if data.get("type") == "command" and data.get("action") not in allowed:
                data["type"] = "chat"
                data["action"] = None
                data["reply"] = f"I cannot perform that action for security reasons. Available actions: {', '.join(sorted(allowed))}"

            return data

        except Exception as e:
            print(f"LLM structured error: {e}")
            error_msg = str(e).lower()
            if "api key" in error_msg or "authentication" in error_msg:
                return {"type": "chat", "action": None, "args": {}, "reply": "Authentication error. Please check your Gemini API key configuration."}
            else:
                return {"type": "chat", "action": None, "args": {}, "reply": self.generate_response(query)}
