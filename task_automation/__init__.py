"""
JARVIS Task Automation & Routines Module

This module provides comprehensive task automation capabilities including:
- Intelligent workflow management
- Conditional logic and decision trees
- Multi-modal triggers (voice, gesture, time, events)
- Context-aware automation
- Natural language routine creation
- Integration with all JARVIS modules

Components:
- AutomationEngine: Core automation processing
- RoutineBuilder: Visual workflow designer
- TriggerManager: Multi-modal trigger system
- TaskExecutor: Task execution and monitoring
- ScheduleManager: Time-based automation
- ContextAwareAutomation: Intelligent context handling
"""

from .automation_engine import AutomationEngine, RoutineBuilder, TaskType, ConditionType
from .trigger_manager import TriggerManager
from .task_executor import TaskExecutor
from .schedule_manager import ScheduleManager
from .context_automation import ContextAwareAutomation
from .nlp_routine_creator import NaturalLanguageRoutineCreator

__all__ = [
    'AutomationEngine',
    'RoutineBuilder', 
    'TaskType',
    'ConditionType',
    'TriggerManager',
    'TaskExecutor',
    'ScheduleManager',
    'ContextAwareAutomation',
    'NaturalLanguageRoutineCreator'
]

__version__ = "1.0.0"
__author__ = "JARVIS AI Assistant"