# Task Automation Module

Provides intelligent workflow automation for the Jarvis system including:

## Key Components
- AutomationEngine: Executes routines and manages task chains
- RoutineBuilder: Fluent builder for constructing routines (sample morning/focus/evening routines)
- NaturalLanguageRoutineCreator: Converts natural language sentences into structured routines with triggers, conditions, and tasks
- TriggerManager (planned integration): Multi-modal triggers (voice, gesture, time, device, presence, context, sensor)
- ScheduleManager (planned integration): Cron, interval, recurring, smart scheduling
- ContextAwareAutomation (planned integration): Context rules, pattern learning, predictive triggers
- TaskExecutor (extended behaviors) for richer task types and retries (separate file)

## Features Implemented
- Routine definition with tasks, triggers, conditional branches
- Task types: voice command, gesture action, smart home control, system command, AR interface update, notification, delay, condition, loop, API call, file operation, user input
- Conditional logic with time/user/device/weather/custom evaluators (stubs extendable)
- Task chaining (on_success/on_failure) plus sequential fallback ordering
- Execution tracking: status, results, failures, history
- Persistence: routines stored in `data/automation/*.json`
- Natural Language routine creation (pattern-based) with heuristics for:
  - Temporal triggers: specific time, everyday/weekday, sunrise/sunset
  - Conditions: threshold (above/below), state, arrival/location, simple context
  - Actions: turn on/off/start devices, play music, set thermostat, close/open/dim, weather, calendar events, focus mode, mute notifications, disarm security
  - Fallback voice trigger if no explicit trigger/condition found

## Natural Language Examples
Input: "Every weekday at 7 am turn on the office lights, start the coffee machine, and read me the weather and my first three calendar events"
→ Creates a routine with time trigger 07:00 (Mon–Fri) and tasks: Turn On Office Lights → Start Coffee Machine → Get Weather → Read First 3 Calendar Events.

Input: "If the living room temperature goes above 26 degrees, turn on the AC and close the blinds"
→ Threshold condition + actions: Turn On AC → Close Blinds.

Input: "At sunset, turn on ambient lights and play relaxing music"
→ Time trigger (sunset) + actions: Turn On Ambient Lights → Play Relaxing Music.

## Voice Assistant Integration
The voice assistant now supports automation commands:
- "Create a routine ..." / "Make a routine ..." / "Build an automation ..." → Parses natural language and persists a new routine.
- "List routines" → Speaks up to 10 existing routines.
- "Run focus mode routine" / "Start morning routine" → Finds partial name match and executes asynchronously.

## Internal Execution Flow
1. Voice input recognized → `_process_command` in `voice_assistant.py`.
2. `_maybe_handle_automation` detects routine-related command.
3. Creation: `NaturalLanguageRoutineCreator.parse_natural_language` builds routine via shared `RoutineBuilder` instance.
4. Routine persisted by `AutomationEngine.add_routine` (JSON file saved).
5. Execution: `AutomationEngine.execute_routine` walks task chain asynchronously.
6. Results stored in execution context with history appended.

## Extending NLP Parsing
Enhance `nlp_routine_creator.py` by:
- Adding richer temporal parsing (cron-like phrases, offsets)
- Integrating a semantic LLM for intent classification & slot filling
- Entity normalization (devices, zones, activities)
- Confidence scoring & user confirmation for ambiguous routines

## Planned / Next Steps
- Wire TriggerManager to automatically invoke routines from real events
- Integrate ScheduleManager loop to schedule time-based triggers
- Hook ContextAwareAutomation predictions to proactive routine suggestions
- Add test suite `tests/test_automation.py` for parsing, execution, persistence
- Expand task executors to call real modules (voice assistant, smart home, AR UI)
- Add security/auth checks before running sensitive routines (e.g., disarm security)
- Provide a GUI/AR visual workflow editor & status dashboard

## Data Layout
```
data/
  automation/
    <routine_id>.json   # persisted routines
  triggers/             # (future) persisted trigger definitions
  schedules/            # (future) schedule metadata
  context/              # (future) context snapshots & learned patterns
```

## Error Handling & Safeguards
- Graceful fallback if automation modules fail to import
- Timeouts per task (default 30s) with exception wrapping
- Condition checks short-circuit unmet tasks
- Routine disabled flag (future toggle UI)
- Partial name match heuristic avoids executing completely unrelated routine names (future: use fuzzy ratio threshold)

## Minimal Programmatic Example
```python
from task_automation import AutomationEngine
engine = AutomationEngine()
routine = engine.routine_builder.create_morning_routine()
engine.add_routine(routine)
import asyncio
asyncio.run(engine.execute_routine(routine.id))
```

## Contributing
1. Add new TaskType in `automation_engine.py`
2. Implement executor method `_execute_<type>`
3. Register in `self.task_executors`
4. (Optional) Extend NLP patterns to generate this new task

## License
Internal project module (adjust as appropriate).
