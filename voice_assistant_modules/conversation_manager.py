import json
import time
from datetime import datetime

class ConversationManager:
    """
    Manages conversation context, memory, and multi-turn dialogue.
    """
    def __init__(self, memory_file="conversation_memory.json", max_history=50):
        self.memory_file = memory_file
        self.max_history = max_history
        self.conversation_history = []
        self.user_context = {}
        self.session_start = datetime.now()
        self.load_memory()
        
    def load_memory(self):
        """Load conversation memory from file."""
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                self.conversation_history = data.get('history', [])
                self.user_context = data.get('context', {})
        except FileNotFoundError:
            print("No previous conversation memory found. Starting fresh.")
        except Exception as e:
            print(f"Error loading memory: {e}")
            
    def save_memory(self):
        """Save conversation memory to file."""
        try:
            data = {
                'history': self.conversation_history[-self.max_history:],
                'context': self.user_context,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
            
    def add_user_message(self, message):
        """Add user message to conversation history."""
        entry = {
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        }
        self.conversation_history.append(entry)
        self.save_memory()
        
    def add_assistant_message(self, message):
        """Add assistant message to conversation history."""
        entry = {
            'role': 'assistant',
            'content': message,
            'timestamp': datetime.now().isoformat()
        }
        self.conversation_history.append(entry)
        self.save_memory()
        
    def update_context(self, key, value):
        """Update user context information."""
        self.user_context[key] = value
        self.save_memory()
        
    def get_context_prompt(self):
        """Generate context prompt for LLM."""
        context = f"Current session started at: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if self.user_context:
            context += "User context:\n"
            for key, value in self.user_context.items():
                context += f"- {key}: {value}\n"
                
        return context
        
    def get_conversation_history(self, limit=10):
        """Get recent conversation history for context."""
        return self.conversation_history[-limit:] if self.conversation_history else []
        
    def clear_history(self):
        """Clear conversation history but keep context."""
        self.conversation_history = []
        self.save_memory()
        
    def analyze_intent(self, message):
        """Simple intent analysis (can be enhanced with NLP)."""
        message_lower = message.lower()
        
        # Command patterns
        if any(word in message_lower for word in ['turn on', 'turn off', 'switch', 'activate', 'deactivate']):
            return 'device_control'
        elif any(word in message_lower for word in ['weather', 'temperature', 'forecast']):
            return 'weather_query'
        elif any(word in message_lower for word in ['remind', 'reminder', 'schedule', 'appointment']):
            return 'reminder'
        elif any(word in message_lower for word in ['what', 'how', 'why', 'when', 'where']):
            return 'question'
        else:
            return 'conversation'