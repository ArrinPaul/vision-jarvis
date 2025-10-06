"""
JARVIS Context-Aware Automation

Provides intelligent context awareness for automation including:
- Environmental context detection
- User behavior analysis
- Situational awareness
- Adaptive automation based on context
- Context prediction and learning
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from pathlib import Path
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of context information"""
    LOCATION = "location"
    TIME = "time"
    WEATHER = "weather"
    USER_ACTIVITY = "user_activity"
    DEVICE_STATUS = "device_status"
    CALENDAR = "calendar"
    ENVIRONMENT = "environment"
    BIOMETRIC = "biometric"
    SOCIAL = "social"
    APPLICATION = "application"

class ContextPriority(Enum):
    """Priority levels for context factors"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ContextData:
    """Individual context data point"""
    type: ContextType
    key: str
    value: Any
    confidence: float
    timestamp: datetime
    source: str
    priority: ContextPriority = ContextPriority.MEDIUM
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextRule:
    """Rule for context-based automation decisions"""
    id: str
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    priority: int = 1
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

class ContextAwareAutomation:
    """Context-aware automation engine"""
    
    def __init__(self, data_dir: str = "data/context"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Context storage
        self.current_context: Dict[str, ContextData] = {}
        self.context_history: List[ContextData] = []
        self.context_rules: Dict[str, ContextRule] = {}
        
        # Context sources
        self.context_sources: Dict[str, Callable] = {}
        self.active_sources: Set[str] = set()
        
        # Learning and prediction
        self.user_patterns: Dict[str, Any] = {}
        self.context_predictions: Dict[str, Any] = {}
        
        # Callbacks
        self.automation_callbacks: List[Callable] = []
        self.context_change_callbacks: List[Callable] = []
        
        # Settings
        self.context_retention_hours = 168  # 1 week
        self.learning_enabled = True
        self.prediction_enabled = True
        
        self.running = False
        
        # Initialize built-in context sources
        self._initialize_builtin_sources()
        
        # Load existing data
        self._load_context_rules()
        self._load_user_patterns()
        
        logger.info("ContextAwareAutomation initialized")
    
    def add_automation_callback(self, callback: Callable):
        """Add callback for automation execution"""
        self.automation_callbacks.append(callback)
    
    def add_context_change_callback(self, callback: Callable):
        """Add callback for context changes"""
        self.context_change_callbacks.append(callback)
    
    def register_context_source(self, source_name: str, source_func: Callable):
        """Register a context data source"""
        self.context_sources[source_name] = source_func
        logger.info(f"Registered context source: {source_name}")
    
    async def start_monitoring(self):
        """Start context monitoring"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting context monitoring")
        
        # Start context collection loop
        asyncio.create_task(self._context_collection_loop())
        
        # Start context analysis loop
        asyncio.create_task(self._context_analysis_loop())
        
        # Start learning loop
        if self.learning_enabled:
            asyncio.create_task(self._learning_loop())
        
        # Start prediction loop
        if self.prediction_enabled:
            asyncio.create_task(self._prediction_loop())
    
    async def stop_monitoring(self):
        """Stop context monitoring"""
        self.running = False
        logger.info("Stopping context monitoring")
    
    async def _context_collection_loop(self):
        """Main context collection loop"""
        while self.running:
            try:
                # Collect context from all active sources
                for source_name in self.active_sources:
                    source_func = self.context_sources.get(source_name)
                    if source_func:
                        try:
                            context_data = await source_func()
                            if context_data:
                                await self._process_context_data(context_data)
                        except Exception as e:
                            logger.error(f"Error collecting from {source_name}: {e}")
                
                # Clean up expired context
                await self._cleanup_expired_context()
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in context collection loop: {e}")
                await asyncio.sleep(60)
    
    async def _context_analysis_loop(self):
        """Context analysis and rule evaluation loop"""
        while self.running:
            try:
                # Evaluate context rules
                await self._evaluate_context_rules()
                
                # Sleep for 15 seconds
                await asyncio.sleep(15)
                
            except Exception as e:
                logger.error(f"Error in context analysis loop: {e}")
                await asyncio.sleep(30)
    
    async def _learning_loop(self):
        """Machine learning loop for pattern detection"""
        while self.running:
            try:
                # Update user patterns based on context history
                await self._update_user_patterns()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(600)
    
    async def _prediction_loop(self):
        """Context prediction loop"""
        while self.running:
            try:
                # Generate context predictions
                await self._generate_context_predictions()
                
                # Sleep for 2 minutes
                await asyncio.sleep(120)
                
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(300)
    
    async def update_context(self, 
                           context_type: ContextType,
                           key: str,
                           value: Any,
                           confidence: float = 1.0,
                           source: str = "manual",
                           priority: ContextPriority = ContextPriority.MEDIUM,
                           expires_in_minutes: int = None,
                           metadata: Dict[str, Any] = None):
        """Update context information"""
        
        expires_at = None
        if expires_in_minutes:
            expires_at = datetime.now() + timedelta(minutes=expires_in_minutes)
        
        context_data = ContextData(
            type=context_type,
            key=key,
            value=value,
            confidence=confidence,
            timestamp=datetime.now(),
            source=source,
            priority=priority,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        await self._process_context_data(context_data)
    
    async def _process_context_data(self, context_data: ContextData):
        """Process new context data"""
        context_key = f"{context_data.type.value}_{context_data.key}"
        
        # Check if this is a significant change
        old_context = self.current_context.get(context_key)
        is_significant_change = self._is_significant_change(old_context, context_data)
        
        # Update current context
        self.current_context[context_key] = context_data
        
        # Add to history
        self.context_history.append(context_data)
        
        # Trim history if too large
        if len(self.context_history) > 10000:
            self.context_history = self.context_history[-5000:]
        
        # Notify callbacks if significant change
        if is_significant_change:
            for callback in self.context_change_callbacks:
                try:
                    await callback(context_data, old_context)
                except Exception as e:
                    logger.error(f"Context change callback failed: {e}")
        
        logger.debug(f"Updated context: {context_key} = {context_data.value}")
    
    def _is_significant_change(self, old_context: Optional[ContextData], new_context: ContextData) -> bool:
        """Determine if context change is significant"""
        if not old_context:
            return True
        
        # For numeric values, check percentage change
        if isinstance(old_context.value, (int, float)) and isinstance(new_context.value, (int, float)):
            if old_context.value == 0:
                return new_context.value != 0
            change_percent = abs((new_context.value - old_context.value) / old_context.value)
            return change_percent > 0.1  # 10% change threshold
        
        # For other types, check equality
        return old_context.value != new_context.value
    
    async def _cleanup_expired_context(self):
        """Remove expired context data"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, context_data in self.current_context.items():
            if context_data.expires_at and current_time > context_data.expires_at:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.current_context[key]
        
        # Clean up old history
        cutoff_time = current_time - timedelta(hours=self.context_retention_hours)
        self.context_history = [
            ctx for ctx in self.context_history 
            if ctx.timestamp > cutoff_time
        ]
    
    def add_context_rule(self,
                        name: str,
                        description: str,
                        conditions: List[Dict[str, Any]],
                        actions: List[Dict[str, Any]],
                        priority: int = 1) -> ContextRule:
        """Add a context-based automation rule"""
        
        rule_id = f"rule_{len(self.context_rules)}_{int(time.time())}"
        
        rule = ContextRule(
            id=rule_id,
            name=name,
            description=description,
            conditions=conditions,
            actions=actions,
            priority=priority
        )
        
        self.context_rules[rule_id] = rule
        self._save_context_rule(rule)
        
        logger.info(f"Added context rule: {name}")
        return rule
    
    async def _evaluate_context_rules(self):
        """Evaluate all context rules"""
        # Sort rules by priority
        sorted_rules = sorted(
            self.context_rules.values(),
            key=lambda r: r.priority,
            reverse=True
        )
        
        for rule in sorted_rules:
            if rule.enabled:
                try:
                    if await self._evaluate_rule_conditions(rule):
                        await self._execute_rule_actions(rule)
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    async def _evaluate_rule_conditions(self, rule: ContextRule) -> bool:
        """Evaluate if rule conditions are met"""
        for condition in rule.conditions:
            if not await self._evaluate_condition(condition):
                return False
        return True
    
    async def _evaluate_condition(self, condition: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        condition_type = condition.get("type")
        
        if condition_type == "context_value":
            return self._evaluate_context_value_condition(condition)
        elif condition_type == "context_change":
            return self._evaluate_context_change_condition(condition)
        elif condition_type == "time_based":
            return self._evaluate_time_condition(condition)
        elif condition_type == "pattern_based":
            return self._evaluate_pattern_condition(condition)
        else:
            logger.warning(f"Unknown condition type: {condition_type}")
            return False
    
    def _evaluate_context_value_condition(self, condition: Dict[str, Any]) -> bool:
        """Evaluate context value condition"""
        context_key = condition.get("context_key")
        operator = condition.get("operator", "equals")
        expected_value = condition.get("value")
        min_confidence = condition.get("min_confidence", 0.5)
        
        if not context_key:
            return False
        
        context_data = self.current_context.get(context_key)
        if not context_data or context_data.confidence < min_confidence:
            return False
        
        actual_value = context_data.value
        
        if operator == "equals":
            return actual_value == expected_value
        elif operator == "not_equals":
            return actual_value != expected_value
        elif operator == "greater_than":
            return float(actual_value) > float(expected_value)
        elif operator == "less_than":
            return float(actual_value) < float(expected_value)
        elif operator == "contains":
            return str(expected_value) in str(actual_value)
        elif operator == "in_range":
            min_val, max_val = expected_value
            return min_val <= float(actual_value) <= max_val
        else:
            return False
    
    def _evaluate_context_change_condition(self, condition: Dict[str, Any]) -> bool:
        """Evaluate context change condition"""
        context_key = condition.get("context_key")
        change_type = condition.get("change_type", "any")
        time_window_minutes = condition.get("time_window_minutes", 5)
        
        if not context_key:
            return False
        
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        # Find recent changes to this context
        recent_changes = [
            ctx for ctx in self.context_history
            if (f"{ctx.type.value}_{ctx.key}" == context_key and
                ctx.timestamp > cutoff_time)
        ]
        
        if change_type == "any":
            return len(recent_changes) > 0
        elif change_type == "increased":
            if len(recent_changes) >= 2:
                return recent_changes[-1].value > recent_changes[0].value
        elif change_type == "decreased":
            if len(recent_changes) >= 2:
                return recent_changes[-1].value < recent_changes[0].value
        
        return False
    
    def _evaluate_time_condition(self, condition: Dict[str, Any]) -> bool:
        """Evaluate time-based condition"""
        time_type = condition.get("time_type")
        
        current_time = datetime.now()
        
        if time_type == "time_range":
            start_time = condition.get("start_time")
            end_time = condition.get("end_time")
            
            current_time_str = current_time.strftime("%H:%M")
            return start_time <= current_time_str <= end_time
        
        elif time_type == "day_of_week":
            allowed_days = condition.get("days", [])
            current_day = current_time.strftime("%A").lower()
            return current_day in [d.lower() for d in allowed_days]
        
        return False
    
    def _evaluate_pattern_condition(self, condition: Dict[str, Any]) -> bool:
        """Evaluate pattern-based condition"""
        pattern_type = condition.get("pattern_type")
        
        if pattern_type == "user_routine":
            # Check if user is following typical routine
            return self._check_user_routine_pattern(condition)
        
        return False
    
    def _check_user_routine_pattern(self, condition: Dict[str, Any]) -> bool:
        """Check user routine pattern"""
        # TODO: Implement sophisticated pattern matching
        # For now, return true for demonstration
        return True
    
    async def _execute_rule_actions(self, rule: ContextRule):
        """Execute actions for a triggered rule"""
        logger.info(f"Executing actions for rule: {rule.name}")
        
        # Update rule stats
        rule.last_triggered = datetime.now()
        rule.trigger_count += 1
        
        for action in rule.actions:
            try:
                await self._execute_action(action)
            except Exception as e:
                logger.error(f"Failed to execute action: {e}")
        
        # Save updated rule
        self._save_context_rule(rule)
    
    async def _execute_action(self, action: Dict[str, Any]):
        """Execute a single action"""
        action_type = action.get("type")
        
        if action_type == "trigger_routine":
            routine_id = action.get("routine_id")
            context = action.get("context", {})
            
            # Execute through automation callbacks
            for callback in self.automation_callbacks:
                await callback(routine_id, context)
        
        elif action_type == "update_context":
            context_type = ContextType(action.get("context_type"))
            key = action.get("key")
            value = action.get("value")
            
            await self.update_context(context_type, key, value, source="context_rule")
        
        elif action_type == "send_notification":
            message = action.get("message")
            notification_type = action.get("notification_type", "info")
            
            # TODO: Send notification through notification system
            logger.info(f"NOTIFICATION [{notification_type}]: {message}")
    
    async def _update_user_patterns(self):
        """Update user behavior patterns based on context history"""
        if not self.learning_enabled or len(self.context_history) < 50:
            return
        
        try:
            # Analyze time-based patterns
            self._analyze_time_patterns()
            
            # Analyze activity patterns
            self._analyze_activity_patterns()
            
            # Analyze environmental patterns
            self._analyze_environment_patterns()
            
            # Save updated patterns
            self._save_user_patterns()
            
        except Exception as e:
            logger.error(f"Error updating user patterns: {e}")
    
    def _analyze_time_patterns(self):
        """Analyze time-based user patterns"""
        # Group context by hour of day
        hourly_activity = {}
        
        for ctx in self.context_history[-1000:]:  # Last 1000 entries
            if ctx.type == ContextType.USER_ACTIVITY:
                hour = ctx.timestamp.hour
                activity = ctx.value
                
                if hour not in hourly_activity:
                    hourly_activity[hour] = {}
                
                hourly_activity[hour][activity] = hourly_activity[hour].get(activity, 0) + 1
        
        self.user_patterns["hourly_activity"] = hourly_activity
    
    def _analyze_activity_patterns(self):
        """Analyze user activity patterns"""
        # Find most common activities
        activity_counts = {}
        
        for ctx in self.context_history[-500:]:
            if ctx.type == ContextType.USER_ACTIVITY:
                activity = ctx.value
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
        
        # Sort by frequency
        sorted_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)
        self.user_patterns["common_activities"] = sorted_activities[:10]
    
    def _analyze_environment_patterns(self):
        """Analyze environmental context patterns"""
        # Analyze location-based patterns
        location_contexts = {}
        
        for ctx in self.context_history[-300:]:
            if ctx.type == ContextType.LOCATION:
                location = ctx.value
                
                if location not in location_contexts:
                    location_contexts[location] = {
                        "activities": {},
                        "times": [],
                        "duration": 0
                    }
                
                location_contexts[location]["times"].append(ctx.timestamp)
        
        self.user_patterns["location_patterns"] = location_contexts
    
    async def _generate_context_predictions(self):
        """Generate predictions about future context"""
        if not self.prediction_enabled:
            return
        
        try:
            # Predict next likely activity
            next_activity = self._predict_next_activity()
            if next_activity:
                self.context_predictions["next_activity"] = next_activity
            
            # Predict optimal times for routines
            optimal_times = self._predict_optimal_routine_times()
            if optimal_times:
                self.context_predictions["optimal_routine_times"] = optimal_times
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
    
    def _predict_next_activity(self) -> Optional[Dict[str, Any]]:
        """Predict user's next likely activity"""
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Get hourly patterns
        hourly_patterns = self.user_patterns.get("hourly_activity", {})
        hour_patterns = hourly_patterns.get(current_hour, {})
        
        if not hour_patterns:
            return None
        
        # Find most likely activity for this hour
        most_likely = max(hour_patterns.items(), key=lambda x: x[1])
        
        return {
            "activity": most_likely[0],
            "confidence": most_likely[1] / sum(hour_patterns.values()),
            "predicted_at": current_time.isoformat()
        }
    
    def _predict_optimal_routine_times(self) -> Dict[str, Any]:
        """Predict optimal times for different routine types"""
        # TODO: Implement sophisticated time optimization
        # For now, return basic suggestions based on patterns
        
        return {
            "focus_work": ["09:00", "14:00"],
            "exercise": ["07:00", "18:00"],
            "relaxation": ["20:00", "21:00"]
        }
    
    def get_current_context(self, context_type: ContextType = None) -> Dict[str, Any]:
        """Get current context information"""
        if context_type:
            filtered_context = {
                k: v for k, v in self.current_context.items()
                if v.type == context_type
            }
            return {k: v.value for k, v in filtered_context.items()}
        else:
            return {k: v.value for k, v in self.current_context.items()}
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context"""
        summary = {
            "total_context_points": len(self.current_context),
            "by_type": {},
            "recent_changes": len([
                ctx for ctx in self.context_history
                if ctx.timestamp > datetime.now() - timedelta(minutes=30)
            ]),
            "active_rules": len([r for r in self.context_rules.values() if r.enabled]),
            "patterns_learned": len(self.user_patterns),
            "predictions_available": len(self.context_predictions)
        }
        
        # Count by type
        for context_data in self.current_context.values():
            ctx_type = context_data.type.value
            summary["by_type"][ctx_type] = summary["by_type"].get(ctx_type, 0) + 1
        
        return summary
    
    def _initialize_builtin_sources(self):
        """Initialize built-in context sources"""
        
        async def time_context_source():
            """Provide time-based context"""
            now = datetime.now()
            return [
                ContextData(
                    type=ContextType.TIME,
                    key="hour",
                    value=now.hour,
                    confidence=1.0,
                    timestamp=now,
                    source="builtin_time"
                ),
                ContextData(
                    type=ContextType.TIME,
                    key="day_of_week",
                    value=now.strftime("%A").lower(),
                    confidence=1.0,
                    timestamp=now,
                    source="builtin_time"
                ),
                ContextData(
                    type=ContextType.TIME,
                    key="time_period",
                    value=self._get_time_period(now.hour),
                    confidence=1.0,
                    timestamp=now,
                    source="builtin_time"
                )
            ]
        
        self.register_context_source("time", time_context_source)
        self.active_sources.add("time")
    
    def _get_time_period(self, hour: int) -> str:
        """Get time period based on hour"""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def _save_context_rule(self, rule: ContextRule):
        """Save context rule to file"""
        filepath = self.data_dir / "rules" / f"{rule.id}.json"
        filepath.parent.mkdir(exist_ok=True)
        
        rule_data = {
            "id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "conditions": rule.conditions,
            "actions": rule.actions,
            "priority": rule.priority,
            "enabled": rule.enabled,
            "created_at": rule.created_at.isoformat(),
            "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
            "trigger_count": rule.trigger_count
        }
        
        with open(filepath, 'w') as f:
            json.dump(rule_data, f, indent=2)
    
    def _load_context_rules(self):
        """Load context rules from files"""
        rules_dir = self.data_dir / "rules"
        if not rules_dir.exists():
            return
        
        for filepath in rules_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                rule = ContextRule(
                    id=data["id"],
                    name=data["name"],
                    description=data["description"],
                    conditions=data["conditions"],
                    actions=data["actions"],
                    priority=data.get("priority", 1),
                    enabled=data.get("enabled", True),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_triggered=datetime.fromisoformat(data["last_triggered"]) if data.get("last_triggered") else None,
                    trigger_count=data.get("trigger_count", 0)
                )
                
                self.context_rules[rule.id] = rule
                
            except Exception as e:
                logger.error(f"Failed to load context rule from {filepath}: {e}")
        
        logger.info(f"Loaded {len(self.context_rules)} context rules")
    
    def _save_user_patterns(self):
        """Save user patterns to file"""
        filepath = self.data_dir / "user_patterns.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.user_patterns, f, indent=2, default=str)
    
    def _load_user_patterns(self):
        """Load user patterns from file"""
        filepath = self.data_dir / "user_patterns.json"
        
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    self.user_patterns = json.load(f)
                logger.info("Loaded user patterns")
            except Exception as e:
                logger.error(f"Failed to load user patterns: {e}")

# Example usage
if __name__ == "__main__":
    async def test_context_automation():
        context_system = ContextAwareAutomation()
        
        # Add a sample context rule
        rule = context_system.add_context_rule(
            name="Morning Focus Time",
            description="Start focus routine when it's morning and user is working",
            conditions=[
                {
                    "type": "context_value",
                    "context_key": "time_time_period",
                    "operator": "equals",
                    "value": "morning"
                },
                {
                    "type": "context_value",
                    "context_key": "user_activity_current",
                    "operator": "equals",
                    "value": "working"
                }
            ],
            actions=[
                {
                    "type": "trigger_routine",
                    "routine_id": "focus_routine",
                    "context": {"source": "morning_focus_rule"}
                }
            ]
        )
        
        # Start monitoring
        await context_system.start_monitoring()
        
        # Simulate context updates
        await context_system.update_context(
            ContextType.USER_ACTIVITY,
            "current",
            "working",
            confidence=0.9,
            source="activity_detector"
        )
        
        await context_system.update_context(
            ContextType.LOCATION,
            "current",
            "home_office",
            confidence=0.95,
            source="location_service"
        )
        
        # Wait a bit for processing
        await asyncio.sleep(2)
        
        # Get context summary
        summary = context_system.get_context_summary()
        print(f"Context summary: {summary}")
        
        await context_system.stop_monitoring()
    
    # Run test
    asyncio.run(test_context_automation())