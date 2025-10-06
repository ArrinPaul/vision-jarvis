"""
JARVIS Trigger Management System

Provides multi-modal trigger support for automation routines including:
- Voice command triggers
- Gesture-based triggers  
- Time-based scheduling
- Event-driven triggers
- Context-aware triggers
- Sensor-based triggers
"""

import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum
import json
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TriggerType(Enum):
    """Types of automation triggers"""
    VOICE_COMMAND = "voice_command"
    GESTURE = "gesture"
    TIME_BASED = "time_based"
    DEVICE_EVENT = "device_event"
    USER_PRESENCE = "user_presence"
    SYSTEM_EVENT = "system_event"
    SENSOR_DATA = "sensor_data"
    WEATHER_EVENT = "weather_event"
    LOCATION_BASED = "location_based"
    CONTEXT_CHANGE = "context_change"
    API_WEBHOOK = "api_webhook"
    FILE_CHANGE = "file_change"
    APPLICATION_EVENT = "application_event"
    BIOMETRIC_EVENT = "biometric_event"

class TriggerPriority(Enum):
    """Priority levels for trigger execution"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TriggerEvent:
    """Event that can trigger automation"""
    trigger_id: str
    trigger_type: TriggerType
    routine_id: str
    event_data: Dict[str, Any]
    timestamp: datetime
    priority: TriggerPriority = TriggerPriority.NORMAL
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

@dataclass
class TriggerDefinition:
    """Definition of a trigger condition"""
    id: str
    name: str
    trigger_type: TriggerType
    routine_id: str
    parameters: Dict[str, Any]
    enabled: bool = True
    priority: TriggerPriority = TriggerPriority.NORMAL
    conditions: List[Dict[str, Any]] = None
    cooldown_seconds: int = 0
    max_triggers_per_hour: int = 0
    created_at: datetime = None
    last_triggered: datetime = None
    trigger_count: int = 0
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.created_at is None:
            self.created_at = datetime.now()

class TriggerManager:
    """Manages all automation triggers and their execution"""
    
    def __init__(self, data_dir: str = "data/triggers"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.triggers: Dict[str, TriggerDefinition] = {}
        self.active_listeners: Dict[TriggerType, bool] = {}
        self.trigger_handlers: Dict[TriggerType, Callable] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        
        # Callback for routine execution
        self.routine_executor: Optional[Callable] = None
        
        # Initialize trigger handlers
        self._initialize_trigger_handlers()
        
        # Load existing triggers
        self._load_all_triggers()
        
        logger.info("TriggerManager initialized")
    
    def set_routine_executor(self, executor: Callable):
        """Set the routine executor callback"""
        self.routine_executor = executor
    
    def add_trigger(self, 
                   name: str,
                   trigger_type: TriggerType,
                   routine_id: str,
                   parameters: Dict[str, Any],
                   priority: TriggerPriority = TriggerPriority.NORMAL,
                   conditions: List[Dict[str, Any]] = None,
                   cooldown_seconds: int = 0) -> TriggerDefinition:
        """Add a new trigger definition"""
        
        trigger_id = f"{routine_id}_{trigger_type.value}_{len(self.triggers)}"
        
        trigger = TriggerDefinition(
            id=trigger_id,
            name=name,
            trigger_type=trigger_type,
            routine_id=routine_id,
            parameters=parameters,
            priority=priority,
            conditions=conditions or [],
            cooldown_seconds=cooldown_seconds
        )
        
        self.triggers[trigger_id] = trigger
        self._save_trigger(trigger)
        
        # Start listener if needed
        self._ensure_listener_active(trigger_type)
        
        logger.info(f"Added trigger: {name} ({trigger_type.value}) for routine {routine_id}")
        return trigger
    
    def add_voice_trigger(self, 
                         name: str,
                         routine_id: str,
                         phrases: List[str],
                         confidence_threshold: float = 0.8,
                         exact_match: bool = False) -> TriggerDefinition:
        """Add a voice command trigger"""
        
        parameters = {
            "phrases": phrases,
            "confidence_threshold": confidence_threshold,
            "exact_match": exact_match,
            "case_sensitive": False
        }
        
        return self.add_trigger(
            name=name,
            trigger_type=TriggerType.VOICE_COMMAND,
            routine_id=routine_id,
            parameters=parameters,
            cooldown_seconds=2  # Prevent rapid re-triggering
        )
    
    def add_gesture_trigger(self,
                           name: str,
                           routine_id: str,
                           gesture_name: str,
                           confidence_threshold: float = 0.8,
                           hold_duration: float = 0.0) -> TriggerDefinition:
        """Add a gesture-based trigger"""
        
        parameters = {
            "gesture_name": gesture_name,
            "confidence_threshold": confidence_threshold,
            "hold_duration": hold_duration,
            "require_confirmation": False
        }
        
        return self.add_trigger(
            name=name,
            trigger_type=TriggerType.GESTURE,
            routine_id=routine_id,
            parameters=parameters,
            cooldown_seconds=1
        )
    
    def add_time_trigger(self,
                        name: str,
                        routine_id: str,
                        time: str = None,
                        days: List[str] = None,
                        interval_minutes: int = None,
                        cron_expression: str = None) -> TriggerDefinition:
        """Add a time-based trigger"""
        
        parameters = {}
        
        if time and days:
            # Specific time on specific days
            parameters.update({
                "type": "scheduled",
                "time": time,
                "days": days or ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            })
        elif interval_minutes:
            # Recurring interval
            parameters.update({
                "type": "interval",
                "interval_minutes": interval_minutes
            })
        elif cron_expression:
            # Cron-style scheduling
            parameters.update({
                "type": "cron",
                "cron_expression": cron_expression
            })
        else:
            raise ValueError("Must specify either time/days, interval, or cron expression")
        
        return self.add_trigger(
            name=name,
            trigger_type=TriggerType.TIME_BASED,
            routine_id=routine_id,
            parameters=parameters
        )
    
    def add_device_event_trigger(self,
                                name: str,
                                routine_id: str,
                                device_type: str,
                                event_type: str,
                                device_id: str = None,
                                condition_value: Any = None) -> TriggerDefinition:
        """Add a device event trigger"""
        
        parameters = {
            "device_type": device_type,
            "event_type": event_type,
            "device_id": device_id,
            "condition_value": condition_value
        }
        
        return self.add_trigger(
            name=name,
            trigger_type=TriggerType.DEVICE_EVENT,
            routine_id=routine_id,
            parameters=parameters
        )
    
    def add_presence_trigger(self,
                           name: str,
                           routine_id: str,
                           presence_type: str,
                           detection_method: str = "face_detection",
                           sensitivity: float = 0.8) -> TriggerDefinition:
        """Add a user presence trigger"""
        
        parameters = {
            "presence_type": presence_type,  # "arrived", "left", "detected"
            "detection_method": detection_method,
            "sensitivity": sensitivity,
            "timeout_seconds": 30
        }
        
        return self.add_trigger(
            name=name,
            trigger_type=TriggerType.USER_PRESENCE,
            routine_id=routine_id,
            parameters=parameters
        )
    
    def add_context_trigger(self,
                          name: str,
                          routine_id: str,
                          context_type: str,
                          context_value: Any,
                          comparison: str = "equals") -> TriggerDefinition:
        """Add a context change trigger"""
        
        parameters = {
            "context_type": context_type,  # "location", "activity", "mood", etc.
            "context_value": context_value,
            "comparison": comparison  # "equals", "contains", "greater_than", etc.
        }
        
        return self.add_trigger(
            name=name,
            trigger_type=TriggerType.CONTEXT_CHANGE,
            routine_id=routine_id,
            parameters=parameters
        )
    
    def add_sensor_trigger(self,
                          name: str,
                          routine_id: str,
                          sensor_type: str,
                          threshold: float,
                          comparison: str = "greater_than",
                          duration_seconds: int = 0) -> TriggerDefinition:
        """Add a sensor-based trigger"""
        
        parameters = {
            "sensor_type": sensor_type,  # "temperature", "light", "motion", etc.
            "threshold": threshold,
            "comparison": comparison,
            "duration_seconds": duration_seconds
        }
        
        return self.add_trigger(
            name=name,
            trigger_type=TriggerType.SENSOR_DATA,
            routine_id=routine_id,
            parameters=parameters
        )
    
    async def start_monitoring(self):
        """Start monitoring for triggers"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting trigger monitoring")
        
        # Start event processing loop
        asyncio.create_task(self._process_events())
        
        # Start time-based trigger monitoring
        asyncio.create_task(self._monitor_time_triggers())
        
        # Initialize active listeners
        for trigger_type in TriggerType:
            if any(t.trigger_type == trigger_type and t.enabled for t in self.triggers.values()):
                self._ensure_listener_active(trigger_type)
    
    async def stop_monitoring(self):
        """Stop monitoring for triggers"""
        self.running = False
        logger.info("Stopping trigger monitoring")
        
        # Stop all listeners
        for trigger_type in self.active_listeners:
            self.active_listeners[trigger_type] = False
    
    async def process_voice_command(self, command: str, confidence: float = 1.0, context: Dict[str, Any] = None):
        """Process a voice command for trigger matching"""
        
        matching_triggers = []
        
        for trigger in self.triggers.values():
            if (trigger.trigger_type == TriggerType.VOICE_COMMAND and 
                trigger.enabled and
                self._check_cooldown(trigger)):
                
                if self._match_voice_command(trigger, command, confidence):
                    matching_triggers.append(trigger)
        
        # Sort by priority and execute
        matching_triggers.sort(key=lambda t: t.priority.value, reverse=True)
        
        for trigger in matching_triggers:
            await self._queue_trigger_event(trigger, {
                "command": command,
                "confidence": confidence
            }, context)
    
    async def process_gesture(self, gesture_name: str, confidence: float, context: Dict[str, Any] = None):
        """Process a gesture for trigger matching"""
        
        matching_triggers = []
        
        for trigger in self.triggers.values():
            if (trigger.trigger_type == TriggerType.GESTURE and
                trigger.enabled and
                self._check_cooldown(trigger)):
                
                if self._match_gesture(trigger, gesture_name, confidence):
                    matching_triggers.append(trigger)
        
        matching_triggers.sort(key=lambda t: t.priority.value, reverse=True)
        
        for trigger in matching_triggers:
            await self._queue_trigger_event(trigger, {
                "gesture_name": gesture_name,
                "confidence": confidence
            }, context)
    
    async def process_device_event(self, device_type: str, event_type: str, event_data: Dict[str, Any], context: Dict[str, Any] = None):
        """Process a device event for trigger matching"""
        
        matching_triggers = []
        
        for trigger in self.triggers.values():
            if (trigger.trigger_type == TriggerType.DEVICE_EVENT and
                trigger.enabled and
                self._check_cooldown(trigger)):
                
                if self._match_device_event(trigger, device_type, event_type, event_data):
                    matching_triggers.append(trigger)
        
        matching_triggers.sort(key=lambda t: t.priority.value, reverse=True)
        
        for trigger in matching_triggers:
            await self._queue_trigger_event(trigger, {
                "device_type": device_type,
                "event_type": event_type,
                "event_data": event_data
            }, context)
    
    async def process_presence_event(self, presence_type: str, detection_data: Dict[str, Any], context: Dict[str, Any] = None):
        """Process a user presence event"""
        
        matching_triggers = []
        
        for trigger in self.triggers.values():
            if (trigger.trigger_type == TriggerType.USER_PRESENCE and
                trigger.enabled and
                self._check_cooldown(trigger)):
                
                if self._match_presence_event(trigger, presence_type, detection_data):
                    matching_triggers.append(trigger)
        
        matching_triggers.sort(key=lambda t: t.priority.value, reverse=True)
        
        for trigger in matching_triggers:
            await self._queue_trigger_event(trigger, {
                "presence_type": presence_type,
                "detection_data": detection_data
            }, context)
    
    async def process_context_change(self, context_type: str, old_value: Any, new_value: Any, context: Dict[str, Any] = None):
        """Process a context change event"""
        
        matching_triggers = []
        
        for trigger in self.triggers.values():
            if (trigger.trigger_type == TriggerType.CONTEXT_CHANGE and
                trigger.enabled and
                self._check_cooldown(trigger)):
                
                if self._match_context_change(trigger, context_type, new_value):
                    matching_triggers.append(trigger)
        
        matching_triggers.sort(key=lambda t: t.priority.value, reverse=True)
        
        for trigger in matching_triggers:
            await self._queue_trigger_event(trigger, {
                "context_type": context_type,
                "old_value": old_value,
                "new_value": new_value
            }, context)
    
    async def process_sensor_data(self, sensor_type: str, value: float, context: Dict[str, Any] = None):
        """Process sensor data for trigger matching"""
        
        matching_triggers = []
        
        for trigger in self.triggers.values():
            if (trigger.trigger_type == TriggerType.SENSOR_DATA and
                trigger.enabled and
                self._check_cooldown(trigger)):
                
                if self._match_sensor_data(trigger, sensor_type, value):
                    matching_triggers.append(trigger)
        
        matching_triggers.sort(key=lambda t: t.priority.value, reverse=True)
        
        for trigger in matching_triggers:
            await self._queue_trigger_event(trigger, {
                "sensor_type": sensor_type,
                "value": value
            }, context)
    
    def _match_voice_command(self, trigger: TriggerDefinition, command: str, confidence: float) -> bool:
        """Check if voice command matches trigger"""
        phrases = trigger.parameters.get("phrases", [])
        threshold = trigger.parameters.get("confidence_threshold", 0.8)
        exact_match = trigger.parameters.get("exact_match", False)
        case_sensitive = trigger.parameters.get("case_sensitive", False)
        
        if confidence < threshold:
            return False
        
        if not case_sensitive:
            command = command.lower()
            phrases = [p.lower() for p in phrases]
        
        for phrase in phrases:
            if exact_match:
                if command.strip() == phrase.strip():
                    return True
            else:
                if phrase in command or self._fuzzy_match(phrase, command):
                    return True
        
        return False
    
    def _match_gesture(self, trigger: TriggerDefinition, gesture_name: str, confidence: float) -> bool:
        """Check if gesture matches trigger"""
        target_gesture = trigger.parameters.get("gesture_name")
        threshold = trigger.parameters.get("confidence_threshold", 0.8)
        
        return (gesture_name == target_gesture and confidence >= threshold)
    
    def _match_device_event(self, trigger: TriggerDefinition, device_type: str, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Check if device event matches trigger"""
        trigger_device_type = trigger.parameters.get("device_type")
        trigger_event_type = trigger.parameters.get("event_type")
        device_id = trigger.parameters.get("device_id")
        condition_value = trigger.parameters.get("condition_value")
        
        # Check device type and event type
        if device_type != trigger_device_type or event_type != trigger_event_type:
            return False
        
        # Check specific device ID if specified
        if device_id and event_data.get("device_id") != device_id:
            return False
        
        # Check condition value if specified
        if condition_value is not None:
            event_value = event_data.get("value")
            if event_value != condition_value:
                return False
        
        return True
    
    def _match_presence_event(self, trigger: TriggerDefinition, presence_type: str, detection_data: Dict[str, Any]) -> bool:
        """Check if presence event matches trigger"""
        trigger_presence_type = trigger.parameters.get("presence_type")
        detection_method = trigger.parameters.get("detection_method")
        sensitivity = trigger.parameters.get("sensitivity", 0.8)
        
        if presence_type != trigger_presence_type:
            return False
        
        # Check detection confidence
        confidence = detection_data.get("confidence", 1.0)
        if confidence < sensitivity:
            return False
        
        return True
    
    def _match_context_change(self, trigger: TriggerDefinition, context_type: str, new_value: Any) -> bool:
        """Check if context change matches trigger"""
        trigger_context_type = trigger.parameters.get("context_type")
        trigger_value = trigger.parameters.get("context_value")
        comparison = trigger.parameters.get("comparison", "equals")
        
        if context_type != trigger_context_type:
            return False
        
        return self._compare_values(new_value, trigger_value, comparison)
    
    def _match_sensor_data(self, trigger: TriggerDefinition, sensor_type: str, value: float) -> bool:
        """Check if sensor data matches trigger"""
        trigger_sensor_type = trigger.parameters.get("sensor_type")
        threshold = trigger.parameters.get("threshold")
        comparison = trigger.parameters.get("comparison", "greater_than")
        
        if sensor_type != trigger_sensor_type:
            return False
        
        return self._compare_values(value, threshold, comparison)
    
    def _compare_values(self, value1: Any, value2: Any, comparison: str) -> bool:
        """Compare two values based on comparison type"""
        try:
            if comparison == "equals":
                return value1 == value2
            elif comparison == "not_equals":
                return value1 != value2
            elif comparison == "greater_than":
                return float(value1) > float(value2)
            elif comparison == "less_than":
                return float(value1) < float(value2)
            elif comparison == "greater_equal":
                return float(value1) >= float(value2)
            elif comparison == "less_equal":
                return float(value1) <= float(value2)
            elif comparison == "contains":
                return str(value2) in str(value1)
            elif comparison == "starts_with":
                return str(value1).startswith(str(value2))
            elif comparison == "ends_with":
                return str(value1).endswith(str(value2))
            else:
                return False
        except (ValueError, TypeError):
            return False
    
    def _fuzzy_match(self, phrase: str, command: str, threshold: float = 0.7) -> bool:
        """Perform fuzzy matching between phrase and command"""
        # Simple fuzzy matching using word overlap
        phrase_words = set(phrase.split())
        command_words = set(command.split())
        
        if not phrase_words:
            return False
        
        overlap = len(phrase_words.intersection(command_words))
        similarity = overlap / len(phrase_words.union(command_words))
        
        return similarity >= threshold
    
    def _check_cooldown(self, trigger: TriggerDefinition) -> bool:
        """Check if trigger is within cooldown period"""
        if trigger.cooldown_seconds <= 0:
            return True
        
        if not trigger.last_triggered:
            return True
        
        cooldown_end = trigger.last_triggered + timedelta(seconds=trigger.cooldown_seconds)
        return datetime.now() >= cooldown_end
    
    async def _queue_trigger_event(self, trigger: TriggerDefinition, event_data: Dict[str, Any], context: Dict[str, Any] = None):
        """Queue a trigger event for processing"""
        
        # Update trigger stats
        trigger.last_triggered = datetime.now()
        trigger.trigger_count += 1
        
        # Create trigger event
        event = TriggerEvent(
            trigger_id=trigger.id,
            trigger_type=trigger.trigger_type,
            routine_id=trigger.routine_id,
            event_data=event_data,
            timestamp=datetime.now(),
            priority=trigger.priority,
            context=context or {}
        )
        
        await self.event_queue.put(event)
        logger.info(f"Queued trigger event: {trigger.name} -> {trigger.routine_id}")
    
    async def _process_events(self):
        """Process trigger events from the queue"""
        while self.running:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Execute routine if executor is available
                if self.routine_executor:
                    try:
                        await self.routine_executor(event.routine_id, event.context)
                        logger.info(f"Executed routine {event.routine_id} from trigger {event.trigger_id}")
                    except Exception as e:
                        logger.error(f"Failed to execute routine {event.routine_id}: {e}")
                else:
                    logger.warning("No routine executor set, skipping event")
                
                # Mark event as done
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing trigger event: {e}")
    
    async def _monitor_time_triggers(self):
        """Monitor time-based triggers"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for trigger in self.triggers.values():
                    if (trigger.trigger_type == TriggerType.TIME_BASED and
                        trigger.enabled and
                        self._check_cooldown(trigger)):
                        
                        if self._should_trigger_time_based(trigger, current_time):
                            await self._queue_trigger_event(trigger, {
                                "trigger_time": current_time.isoformat()
                            })
                
                # Sleep for 30 seconds before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in time trigger monitoring: {e}")
                await asyncio.sleep(60)
    
    def _should_trigger_time_based(self, trigger: TriggerDefinition, current_time: datetime) -> bool:
        """Check if time-based trigger should fire"""
        params = trigger.parameters
        trigger_type = params.get("type")
        
        if trigger_type == "scheduled":
            # Check specific time and days
            target_time = params.get("time")
            target_days = params.get("days", [])
            
            if not target_time:
                return False
            
            # Parse target time
            try:
                target_hour, target_minute = map(int, target_time.split(":"))
            except ValueError:
                return False
            
            # Check if current day matches
            current_day = current_time.strftime("%A").lower()
            if target_days and current_day not in [d.lower() for d in target_days]:
                return False
            
            # Check if time matches (within 1 minute window)
            if (current_time.hour == target_hour and 
                abs(current_time.minute - target_minute) <= 1):
                return True
        
        elif trigger_type == "interval":
            # Check interval-based trigger
            interval_minutes = params.get("interval_minutes", 60)
            
            if not trigger.last_triggered:
                return True
            
            next_trigger = trigger.last_triggered + timedelta(minutes=interval_minutes)
            return current_time >= next_trigger
        
        elif trigger_type == "cron":
            # TODO: Implement cron expression parsing
            pass
        
        return False
    
    def _initialize_trigger_handlers(self):
        """Initialize trigger type handlers"""
        self.trigger_handlers = {
            TriggerType.VOICE_COMMAND: self._handle_voice_trigger,
            TriggerType.GESTURE: self._handle_gesture_trigger,
            TriggerType.TIME_BASED: self._handle_time_trigger,
            TriggerType.DEVICE_EVENT: self._handle_device_trigger,
            TriggerType.USER_PRESENCE: self._handle_presence_trigger,
            TriggerType.SENSOR_DATA: self._handle_sensor_trigger,
            TriggerType.CONTEXT_CHANGE: self._handle_context_trigger
        }
    
    def _ensure_listener_active(self, trigger_type: TriggerType):
        """Ensure listener is active for trigger type"""
        if trigger_type not in self.active_listeners:
            self.active_listeners[trigger_type] = True
            logger.info(f"Activated listener for {trigger_type.value}")
    
    async def _handle_voice_trigger(self, trigger: TriggerDefinition):
        """Handle voice trigger activation"""
        # Voice triggers are handled by process_voice_command
        pass
    
    async def _handle_gesture_trigger(self, trigger: TriggerDefinition):
        """Handle gesture trigger activation"""
        # Gesture triggers are handled by process_gesture
        pass
    
    async def _handle_time_trigger(self, trigger: TriggerDefinition):
        """Handle time-based trigger activation"""
        # Time triggers are handled by _monitor_time_triggers
        pass
    
    async def _handle_device_trigger(self, trigger: TriggerDefinition):
        """Handle device event trigger activation"""
        # Device triggers are handled by process_device_event
        pass
    
    async def _handle_presence_trigger(self, trigger: TriggerDefinition):
        """Handle presence trigger activation"""
        # Presence triggers are handled by process_presence_event
        pass
    
    async def _handle_sensor_trigger(self, trigger: TriggerDefinition):
        """Handle sensor trigger activation"""
        # Sensor triggers are handled by process_sensor_data
        pass
    
    async def _handle_context_trigger(self, trigger: TriggerDefinition):
        """Handle context change trigger activation"""
        # Context triggers are handled by process_context_change
        pass
    
    def get_trigger(self, trigger_id: str) -> Optional[TriggerDefinition]:
        """Get trigger by ID"""
        return self.triggers.get(trigger_id)
    
    def list_triggers(self, routine_id: str = None, trigger_type: TriggerType = None) -> List[TriggerDefinition]:
        """List triggers with optional filtering"""
        triggers = list(self.triggers.values())
        
        if routine_id:
            triggers = [t for t in triggers if t.routine_id == routine_id]
        
        if trigger_type:
            triggers = [t for t in triggers if t.trigger_type == trigger_type]
        
        return triggers
    
    def enable_trigger(self, trigger_id: str):
        """Enable a trigger"""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = True
            self._save_trigger(self.triggers[trigger_id])
            logger.info(f"Enabled trigger: {trigger_id}")
    
    def disable_trigger(self, trigger_id: str):
        """Disable a trigger"""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = False
            self._save_trigger(self.triggers[trigger_id])
            logger.info(f"Disabled trigger: {trigger_id}")
    
    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a trigger"""
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
            
            # Remove file
            filepath = self.data_dir / f"{trigger_id}.json"
            if filepath.exists():
                filepath.unlink()
            
            logger.info(f"Removed trigger: {trigger_id}")
            return True
        
        return False
    
    def get_trigger_stats(self) -> Dict[str, Any]:
        """Get trigger statistics"""
        stats = {
            "total_triggers": len(self.triggers),
            "enabled_triggers": len([t for t in self.triggers.values() if t.enabled]),
            "by_type": {},
            "most_triggered": None,
            "recent_triggers": []
        }
        
        # Count by type
        for trigger in self.triggers.values():
            trigger_type = trigger.trigger_type.value
            if trigger_type not in stats["by_type"]:
                stats["by_type"][trigger_type] = 0
            stats["by_type"][trigger_type] += 1
        
        # Find most triggered
        if self.triggers:
            most_triggered = max(self.triggers.values(), key=lambda t: t.trigger_count)
            stats["most_triggered"] = {
                "name": most_triggered.name,
                "count": most_triggered.trigger_count
            }
        
        # Recent triggers
        recent = sorted(
            [t for t in self.triggers.values() if t.last_triggered],
            key=lambda t: t.last_triggered,
            reverse=True
        )[:5]
        
        stats["recent_triggers"] = [
            {
                "name": t.name,
                "last_triggered": t.last_triggered.isoformat() if t.last_triggered else None
            }
            for t in recent
        ]
        
        return stats
    
    def _save_trigger(self, trigger: TriggerDefinition):
        """Save trigger to file"""
        filepath = self.data_dir / f"{trigger.id}.json"
        
        trigger_data = {
            "id": trigger.id,
            "name": trigger.name,
            "trigger_type": trigger.trigger_type.value,
            "routine_id": trigger.routine_id,
            "parameters": trigger.parameters,
            "enabled": trigger.enabled,
            "priority": trigger.priority.value,
            "conditions": trigger.conditions,
            "cooldown_seconds": trigger.cooldown_seconds,
            "max_triggers_per_hour": trigger.max_triggers_per_hour,
            "created_at": trigger.created_at.isoformat(),
            "last_triggered": trigger.last_triggered.isoformat() if trigger.last_triggered else None,
            "trigger_count": trigger.trigger_count
        }
        
        with open(filepath, 'w') as f:
            json.dump(trigger_data, f, indent=2)
    
    def _load_all_triggers(self):
        """Load all triggers from files"""
        if not self.data_dir.exists():
            return
        
        for filepath in self.data_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                trigger = TriggerDefinition(
                    id=data["id"],
                    name=data["name"],
                    trigger_type=TriggerType(data["trigger_type"]),
                    routine_id=data["routine_id"],
                    parameters=data["parameters"],
                    enabled=data.get("enabled", True),
                    priority=TriggerPriority(data.get("priority", TriggerPriority.NORMAL.value)),
                    conditions=data.get("conditions", []),
                    cooldown_seconds=data.get("cooldown_seconds", 0),
                    max_triggers_per_hour=data.get("max_triggers_per_hour", 0),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_triggered=datetime.fromisoformat(data["last_triggered"]) if data.get("last_triggered") else None,
                    trigger_count=data.get("trigger_count", 0)
                )
                
                self.triggers[trigger.id] = trigger
                
            except Exception as e:
                logger.error(f"Failed to load trigger from {filepath}: {e}")
        
        logger.info(f"Loaded {len(self.triggers)} triggers")

# Example usage
if __name__ == "__main__":
    async def test_trigger_manager():
        manager = TriggerManager()
        
        # Add sample triggers
        voice_trigger = manager.add_voice_trigger(
            "Morning Greeting",
            "morning_routine_123",
            ["good morning jarvis", "start my day", "morning routine"],
            confidence_threshold=0.7
        )
        
        gesture_trigger = manager.add_gesture_trigger(
            "Focus Gesture",
            "focus_routine_456", 
            "focus_gesture",
            confidence_threshold=0.8
        )
        
        time_trigger = manager.add_time_trigger(
            "Morning Alarm",
            "wake_up_routine_789",
            time="07:00",
            days=["monday", "tuesday", "wednesday", "thursday", "friday"]
        )
        
        # Start monitoring
        await manager.start_monitoring()
        
        # Simulate voice command
        await manager.process_voice_command("good morning jarvis", confidence=0.9)
        
        # Simulate gesture
        await manager.process_gesture("focus_gesture", confidence=0.85)
        
        # Get stats
        stats = manager.get_trigger_stats()
        print(f"Trigger stats: {stats}")
        
        await manager.stop_monitoring()
    
    # Run test
    asyncio.run(test_trigger_manager())