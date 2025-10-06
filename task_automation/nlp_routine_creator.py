"""
Natural Language Routine Creator

Converts user natural language requests into structured automation routines:
Examples:
"Every weekday at 7 am turn on the office lights, start the coffee machine, and read me the weather and my first three calendar events"
"When I sit at my desk and it's after 9am, put me in focus mode and mute all notifications except urgent ones"
"If the living room temperature goes above 26 degrees, turn on the AC and close the blinds"
"At sunset, turn on ambient lights and play relaxing music"
"When I arrive home, disarm security, turn on entry lights, and set thermostat to comfort mode"

Approach:
- Parse temporal expressions
- Detect conditional clauses
- Extract device/actions/intents
- Identify triggers (time, event, condition)
- Build sequence and task chain
- Estimate parameters when unspecified
"""

import re
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from .automation_engine import RoutineBuilder, TaskType, ConditionType

logger = logging.getLogger(__name__)

class NLPIntentType:
    TIME_TRIGGER = 'time_trigger'
    EVENT_TRIGGER = 'event_trigger'
    CONDITION_TRIGGER = 'condition_trigger'
    ACTION_SEQUENCE = 'action_sequence'

class NaturalLanguageRoutineCreator:
    """Converts natural language descriptions into automation routines"""
    
    def __init__(self, routine_builder: RoutineBuilder = None):
        self.routine_builder = routine_builder or RoutineBuilder()
        
        # Simple patterns for time expressions
        self.time_patterns = [
            (r"every weekday at (\d{1,2})(?:[:.](\d{2}))? ?(am|pm)?", self._parse_weekday_time),
            (r"every day at (\d{1,2})(?:[:.](\d{2}))? ?(am|pm)?", self._parse_daily_time),
            (r"at (\d{1,2})(?:[:.](\d{2}))? ?(am|pm)", self._parse_simple_time),
            (r"(sunrise|sunset)", self._parse_sun_time)
        ]
        
        # Simple patterns for conditional expressions
        self.condition_patterns = [
            (r"if the ([a-z ]+) is (above|below|greater than|less than|over|under) (\d+)", self._parse_threshold_condition),
            (r"if the ([a-z ]+) (?:goes|gets) (hot|cold|bright|dark|noisy|quiet)", self._parse_state_condition),
            (r"when I (?:arrive|get home)", self._parse_arrival_condition),
            (r"when I (sit at|sit down at|am at|get to) my (desk|workstation|computer)", self._parse_location_condition)
        ]
        
        # Action patterns
        self.action_patterns = [
            (r"turn on the ([a-z ]+)", self._parse_turn_on_action),
            (r"turn off the ([a-z ]+)", self._parse_turn_off_action),
            (r"start the ([a-z ]+)", self._parse_start_action),
            (r"play (.+?)(?: music)?(?:\. |$| and)", self._parse_play_music_action),
            (r"set (?:the )?thermostat to (\d+)", self._parse_thermostat_action),
            (r"set .*?thermostat .*?comfort", self._parse_thermostat_comfort_action),
            (r"close the ([a-z ]+)", self._parse_close_action),
            (r"open the ([a-z ]+)", self._parse_open_action),
            (r"dim the ([a-z ]+)", self._parse_dim_action),
            (r"read me the weather", self._parse_weather_action),
            (r"(tell|read|list) me (?:my )?first (\d+) calendar events", self._parse_calendar_events_action),
            (r"put me in focus mode", self._parse_focus_mode_action),
            (r"mute all notifications", self._parse_mute_notifications_action),
            (r"disarm security", self._parse_disarm_security_action)
        ]
        
    def parse_natural_language(self, text: str, routine_name: str = None, category: str = "nlp") -> Dict[str, Any]:
        """Parse natural language description into routine structure"""
        original_text = text
        text = text.lower().strip()
        
        if not routine_name:
            routine_name = f"Routine from NL {datetime.now().strftime('%H:%M:%S')}"
        
        # Identify and extract temporal triggers
        time_triggers = self._extract_time_triggers(text)
        
        # Identify conditional triggers
        conditions = self._extract_conditions(text)
        
        # Identify actions
        actions = self._extract_actions(text)
        
        # Remove extracted parts from text and handle leftovers
        # (Keep for future NLP improvements)
        
        # Build routine
        routine = self.routine_builder.create_routine(
            routine_name,
            description=f"Created from natural language: '{original_text}'",
            category=category
        )
        
        # Add actions as tasks
        previous_task = None
        for action in actions:
            task = self.routine_builder.add_task(
                name=action["name"],
                task_type=action["task_type"],
                parameters=action["parameters"],
                timeout=action.get("timeout", 30)
            )
            
            if previous_task:
                self.routine_builder.chain_tasks(previous_task.id, task.id)
            previous_task = task
        
        # Add triggers
        for trigger in time_triggers:
            self.routine_builder.add_trigger(trigger["type"], trigger["parameters"])
        
        # Add condition task at beginning if conditions exist
        if conditions and routine.tasks:
            cond_task = self.routine_builder.add_condition_task(
                name="Check Conditions",
                condition_type=ConditionType.CUSTOM,
                condition_params={"conditions": conditions}
            )
            # Rechain
            if previous_task:
                self.routine_builder.chain_tasks(cond_task.id, routine.tasks[0].id)
                # Move condition to front
                # NOTE: For simplicity; could refine ordering logic
                routine.tasks.insert(0, routine.tasks.pop())
        
        # Heuristic triggers if none found
        if not time_triggers and not conditions:
            self.routine_builder.add_trigger("voice_command", {
                "phrases": [f"run {routine_name.lower()}", f"start {routine_name.lower()}"]
            })
        
        return {
            "routine": routine,
            "time_triggers": time_triggers,
            "conditions": conditions,
            "actions": actions,
            "original_text": original_text
        }
    
    def _extract_time_triggers(self, text: str) -> List[Dict[str, Any]]:
        """Extract time-based triggers from text"""
        triggers = []
        
        for pattern, parser in self.time_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                trigger_data = parser(match)
                if trigger_data:
                    triggers.append(trigger_data)
        
        return triggers
    
    def _extract_conditions(self, text: str) -> List[Dict[str, Any]]:
        """Extract conditional expressions"""
        conditions = []
        
        for pattern, parser in self.condition_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                condition_data = parser(match)
                if condition_data:
                    conditions.append(condition_data)
        
        return conditions
    
    def _extract_actions(self, text: str) -> List[Dict[str, Any]]:
        """Extract action sequences"""
        actions = []
        
        for pattern, parser in self.action_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                action_data = parser(match)
                if action_data:
                    actions.append(action_data)
        
        # Fallback action if none found
        if not actions:
            actions.append({
                "name": "Notify Routine Started",
                "task_type": TaskType.NOTIFICATION,
                "parameters": {
                    "message": "Routine started",
                    "type": "info"
                }
            })
        
        return actions
    
    # Time parsers
    def _parse_weekday_time(self, match) -> Dict[str, Any]:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        ampm = match.group(3)
        
        if ampm == 'pm' and hour < 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        
        return {
            "type": "time_based",
            "parameters": {
                "time": f"{hour:02d}:{minute:02d}",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
            }
        }
    
    def _parse_daily_time(self, match) -> Dict[str, Any]:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        ampm = match.group(3)
        
        if ampm == 'pm' and hour < 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        
        return {
            "type": "time_based",
            "parameters": {
                "time": f"{hour:02d}:{minute:02d}",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            }
        }
    
    def _parse_simple_time(self, match) -> Dict[str, Any]:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        ampm = match.group(3)
        
        if ampm == 'pm' and hour < 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        
        return {
            "type": "time_based",
            "parameters": {
                "time": f"{hour:02d}:{minute:02d}",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            }
        }
    
    def _parse_sun_time(self, match) -> Dict[str, Any]:
        sun_event = match.group(1)
        return {
            "type": "time_based",
            "parameters": {
                "sun_event": sun_event,
                "offset_minutes": 0
            }
        }
    
    # Condition parsers
    def _parse_threshold_condition(self, match) -> Dict[str, Any]:
        metric = match.group(1).strip()
        comparison = match.group(2)
        threshold = int(match.group(3))
        
        operator = "greater_than" if comparison in ["above", "greater than", "over"] else "less_than"
        
        return {
            "type": "threshold",
            "metric": metric,
            "operator": operator,
            "value": threshold
        }
    
    def _parse_state_condition(self, match) -> Dict[str, Any]:
        metric = match.group(1).strip()
        state = match.group(2)
        
        return {
            "type": "state_change",
            "metric": metric,
            "state": state
        }
    
    def _parse_arrival_condition(self, match) -> Dict[str, Any]:
        return {
            "type": "user_arrival",
            "location": "home"
        }
    
    def _parse_location_condition(self, match) -> Dict[str, Any]:
        location = match.group(2)
        return {
            "type": "user_location",
            "location": f"{location}"
        }
    
    # Action parsers
    def _parse_turn_on_action(self, match) -> Dict[str, Any]:
        device = match.group(1).strip()
        return {
            "name": f"Turn On {device.title()}",
            "task_type": TaskType.SMART_HOME_CONTROL,
            "parameters": {
                "device_type": device,
                "action": "turn_on"
            }
        }
    
    def _parse_turn_off_action(self, match) -> Dict[str, Any]:
        device = match.group(1).strip()
        return {
            "name": f"Turn Off {device.title()}",
            "task_type": TaskType.SMART_HOME_CONTROL,
            "parameters": {
                "device_type": device,
                "action": "turn_off"
            }
        }
    
    def _parse_start_action(self, match) -> Dict[str, Any]:
        device = match.group(1).strip()
        return {
            "name": f"Start {device.title()}",
            "task_type": TaskType.SMART_HOME_CONTROL,
            "parameters": {
                "device_type": device,
                "action": "start"
            }
        }
    
    def _parse_play_music_action(self, match) -> Dict[str, Any]:
        music_type = match.group(1).strip()
        # Clean trailing conjunctions
        music_type = re.sub(r" and$", "", music_type)
        music_type = re.sub(r"\.\s*$", "", music_type)
        
        return {
            "name": f"Play {music_type.title()} Music",
            "task_type": TaskType.VOICE_COMMAND,
            "parameters": {
                "command": f"Play {music_type} music"
            }
        }
    
    def _parse_thermostat_action(self, match) -> Dict[str, Any]:
        temperature = int(match.group(1))
        return {
            "name": f"Set Thermostat to {temperature}",
            "task_type": TaskType.SMART_HOME_CONTROL,
            "parameters": {
                "device_type": "thermostat",
                "action": "set_temperature",
                "temperature": temperature
            }
        }
    
    def _parse_thermostat_comfort_action(self, match) -> Dict[str, Any]:
        return {
            "name": "Set Thermostat to Comfort Mode",
            "task_type": TaskType.SMART_HOME_CONTROL,
            "parameters": {
                "device_type": "thermostat",
                "action": "set_mode",
                "mode": "comfort"
            }
        }
    
    def _parse_close_action(self, match) -> Dict[str, Any]:
        target = match.group(1).strip()
        return {
            "name": f"Close {target.title()}",
            "task_type": TaskType.SMART_HOME_CONTROL,
            "parameters": {
                "device_type": target,
                "action": "close"
            }
        }
    
    def _parse_open_action(self, match) -> Dict[str, Any]:
        target = match.group(1).strip()
        return {
            "name": f"Open {target.title()}",
            "task_type": TaskType.SMART_HOME_CONTROL,
            "parameters": {
                "device_type": target,
                "action": "open"
            }
        }
    
    def _parse_dim_action(self, match) -> Dict[str, Any]:
        target = match.group(1).strip()
        return {
            "name": f"Dim {target.title()}",
            "task_type": TaskType.SMART_HOME_CONTROL,
            "parameters": {
                "device_type": target,
                "action": "dim",
                "brightness": 0.4
            }
        }
    
    def _parse_weather_action(self, match) -> Dict[str, Any]:
        return {
            "name": "Get Weather",
            "task_type": TaskType.VOICE_COMMAND,
            "parameters": {
                "command": "What's the weather today?"
            }
        }
    
    def _parse_calendar_events_action(self, match) -> Dict[str, Any]:
        count = int(match.group(2)) if match.group(2) else 3
        return {
            "name": f"Read First {count} Calendar Events",
            "task_type": TaskType.VOICE_COMMAND,
            "parameters": {
                "command": f"Read my first {count} calendar events"
            }
        }
    
    def _parse_focus_mode_action(self, match) -> Dict[str, Any]:
        return {
            "name": "Enable Focus Mode",
            "task_type": TaskType.SYSTEM_COMMAND,
            "parameters": {
                "command": "enable_dnd",
                "duration_minutes": 120
            }
        }
    
    def _parse_mute_notifications_action(self, match) -> Dict[str, Any]:
        return {
            "name": "Mute Notifications",
            "task_type": TaskType.SYSTEM_COMMAND,
            "parameters": {
                "command": "mute_notifications",
                "duration_minutes": 120
            }
        }
    
    def _parse_disarm_security_action(self, match) -> Dict[str, Any]:
        return {
            "name": "Disarm Security System",
            "task_type": TaskType.SYSTEM_COMMAND,
            "parameters": {
                "command": "disarm_security"
            }
        }

# Example usage
if __name__ == "__main__":
    creator = NaturalLanguageRoutineCreator()
    
    examples = [
        "Every weekday at 7 am turn on the office lights, start the coffee machine, and read me the weather and my first three calendar events",
        "When I sit at my desk and it's after 9am, put me in focus mode and mute all notifications except urgent ones",
        "If the living room temperature goes above 26 degrees, turn on the AC and close the blinds",
        "At sunset, turn on ambient lights and play relaxing music",
        "When I arrive home, disarm security, turn on entry lights, and set thermostat to comfort mode"
    ]
    
    for text in examples:
        print(f"\nParsing: {text}")
        result = creator.parse_natural_language(text)
        print(f"Created routine: {result['routine'].name}")
        print(f"Tasks: {len(result['routine'].tasks)}")
        print(f"Triggers: {len(result['time_triggers'])}")
        print(f"Conditions: {len(result['conditions'])}")
