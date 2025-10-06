"""
JARVIS Schedule Manager

Handles time-based automation scheduling including:
- Cron-style scheduling
- Recurring tasks
- Calendar integration
- Smart scheduling with context awareness
- Timezone handling
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
import pytz
from pathlib import Path

# Optional import for cron functionality
try:
    from croniter import croniter
    CRONITER_AVAILABLE = True
except ImportError:
    CRONITER_AVAILABLE = False
    croniter = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScheduleType(Enum):
    """Types of scheduling"""
    ONE_TIME = "one_time"
    RECURRING = "recurring"
    CRON = "cron"
    INTERVAL = "interval"
    SMART = "smart"

class ScheduleStatus(Enum):
    """Status of scheduled tasks"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

@dataclass
class ScheduleEntry:
    """Scheduled task entry"""
    id: str
    name: str
    routine_id: str
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    timezone: str = "UTC"
    created_at: datetime = None
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    conditions: List[Dict[str, Any]] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.conditions is None:
            self.conditions = []
        if self.context is None:
            self.context = {}

class ScheduleManager:
    """Manages time-based automation scheduling"""
    
    def __init__(self, data_dir: str = "data/schedules", default_timezone: str = "UTC"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.schedules: Dict[str, ScheduleEntry] = {}
        self.default_timezone = default_timezone
        self.running = False
        
        # Callback for routine execution
        self.routine_executor: Optional[Callable] = None
        
        # Smart scheduling data
        self.user_patterns: Dict[str, Any] = {}
        self.context_data: Dict[str, Any] = {}
        
        # Load existing schedules
        self._load_all_schedules()
        
        logger.info("ScheduleManager initialized")
    
    def set_routine_executor(self, executor: Callable):
        """Set the routine executor callback"""
        self.routine_executor = executor
    
    def add_schedule(self,
                    name: str,
                    routine_id: str,
                    schedule_type: ScheduleType,
                    schedule_config: Dict[str, Any],
                    timezone: str = None,
                    max_runs: int = None,
                    conditions: List[Dict[str, Any]] = None,
                    context: Dict[str, Any] = None) -> ScheduleEntry:
        """Add a new scheduled task"""
        
        schedule_id = f"schedule_{len(self.schedules)}_{int(datetime.now().timestamp())}"
        
        schedule = ScheduleEntry(
            id=schedule_id,
            name=name,
            routine_id=routine_id,
            schedule_type=schedule_type,
            schedule_config=schedule_config,
            timezone=timezone or self.default_timezone,
            max_runs=max_runs,
            conditions=conditions or [],
            context=context or {}
        )
        
        # Calculate next run time
        schedule.next_run = self._calculate_next_run(schedule)
        
        self.schedules[schedule_id] = schedule
        self._save_schedule(schedule)
        
        logger.info(f"Added schedule: {name} ({schedule_type.value}) - Next run: {schedule.next_run}")
        return schedule
    
    def add_one_time_schedule(self,
                             name: str,
                             routine_id: str,
                             run_time: datetime,
                             timezone: str = None,
                             conditions: List[Dict[str, Any]] = None) -> ScheduleEntry:
        """Add a one-time scheduled task"""
        
        config = {"run_time": run_time.isoformat()}
        
        return self.add_schedule(
            name=name,
            routine_id=routine_id,
            schedule_type=ScheduleType.ONE_TIME,
            schedule_config=config,
            timezone=timezone,
            max_runs=1,
            conditions=conditions
        )
    
    def add_recurring_schedule(self,
                              name: str,
                              routine_id: str,
                              days: List[str],
                              time: str,
                              timezone: str = None,
                              end_date: datetime = None,
                              conditions: List[Dict[str, Any]] = None) -> ScheduleEntry:
        """Add a recurring scheduled task"""
        
        config = {
            "days": days,  # ["monday", "tuesday", etc.]
            "time": time,  # "07:30"
            "end_date": end_date.isoformat() if end_date else None
        }
        
        return self.add_schedule(
            name=name,
            routine_id=routine_id,
            schedule_type=ScheduleType.RECURRING,
            schedule_config=config,
            timezone=timezone,
            conditions=conditions
        )
    
    def add_cron_schedule(self,
                         name: str,
                         routine_id: str,
                         cron_expression: str,
                         timezone: str = None,
                         max_runs: int = None,
                         conditions: List[Dict[str, Any]] = None) -> ScheduleEntry:
        """Add a cron-based scheduled task"""
        
        # Validate cron expression
        try:
            if CRONITER_AVAILABLE:
                croniter(cron_expression)
            else:
                # Basic validation for common cron patterns
                self._validate_cron_fallback(cron_expression)
        except ValueError as e:
            raise ValueError(f"Invalid cron expression: {cron_expression} - {e}")
        
        config = {"cron_expression": cron_expression}
        
        return self.add_schedule(
            name=name,
            routine_id=routine_id,
            schedule_type=ScheduleType.CRON,
            schedule_config=config,
            timezone=timezone,
            max_runs=max_runs,
            conditions=conditions
        )
    
    def add_interval_schedule(self,
                             name: str,
                             routine_id: str,
                             interval_minutes: int,
                             start_time: datetime = None,
                             end_time: datetime = None,
                             max_runs: int = None,
                             conditions: List[Dict[str, Any]] = None) -> ScheduleEntry:
        """Add an interval-based scheduled task"""
        
        config = {
            "interval_minutes": interval_minutes,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None
        }
        
        return self.add_schedule(
            name=name,
            routine_id=routine_id,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config=config,
            max_runs=max_runs,
            conditions=conditions
        )
    
    def add_smart_schedule(self,
                          name: str,
                          routine_id: str,
                          preferred_times: List[str],
                          context_conditions: Dict[str, Any],
                          timezone: str = None,
                          conditions: List[Dict[str, Any]] = None) -> ScheduleEntry:
        """Add a smart scheduled task that adapts to user patterns"""
        
        config = {
            "preferred_times": preferred_times,
            "context_conditions": context_conditions,
            "learning_enabled": True
        }
        
        return self.add_schedule(
            name=name,
            routine_id=routine_id,
            schedule_type=ScheduleType.SMART,
            schedule_config=config,
            timezone=timezone,
            conditions=conditions
        )
    
    async def start_scheduler(self):
        """Start the scheduler"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting schedule manager")
        
        # Start scheduling loop
        asyncio.create_task(self._scheduling_loop())
        
        # Start smart scheduling updates
        asyncio.create_task(self._smart_scheduling_loop())
    
    async def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("Stopping schedule manager")
    
    async def _scheduling_loop(self):
        """Main scheduling loop"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check all active schedules
                for schedule in list(self.schedules.values()):
                    if (schedule.status == ScheduleStatus.ACTIVE and
                        schedule.next_run and
                        current_time >= schedule.next_run):
                        
                        # Check conditions
                        if await self._check_schedule_conditions(schedule):
                            await self._execute_schedule(schedule)
                        else:
                            logger.info(f"Schedule conditions not met: {schedule.name}")
                            # Reschedule for next opportunity
                            schedule.next_run = self._calculate_next_run(schedule)
                
                # Sleep for 30 seconds before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                await asyncio.sleep(60)
    
    async def _smart_scheduling_loop(self):
        """Smart scheduling optimization loop"""
        while self.running:
            try:
                # Update smart schedules based on user patterns
                for schedule in self.schedules.values():
                    if (schedule.schedule_type == ScheduleType.SMART and
                        schedule.status == ScheduleStatus.ACTIVE):
                        
                        await self._optimize_smart_schedule(schedule)
                
                # Sleep for 5 minutes before next optimization
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in smart scheduling loop: {e}")
                await asyncio.sleep(600)
    
    async def _execute_schedule(self, schedule: ScheduleEntry):
        """Execute a scheduled task"""
        try:
            logger.info(f"Executing scheduled task: {schedule.name}")
            
            # Update schedule stats
            schedule.last_run = datetime.now()
            schedule.run_count += 1
            
            # Execute routine if executor is available
            if self.routine_executor:
                await self.routine_executor(schedule.routine_id, schedule.context)
                logger.info(f"Executed routine {schedule.routine_id} from schedule {schedule.id}")
            else:
                logger.warning("No routine executor set, skipping schedule execution")
            
            # Update next run time
            if schedule.max_runs and schedule.run_count >= schedule.max_runs:
                schedule.status = ScheduleStatus.COMPLETED
                schedule.next_run = None
            else:
                schedule.next_run = self._calculate_next_run(schedule)
            
            # Save updated schedule
            self._save_schedule(schedule)
            
        except Exception as e:
            logger.error(f"Failed to execute schedule {schedule.id}: {e}")
    
    def _calculate_next_run(self, schedule: ScheduleEntry) -> Optional[datetime]:
        """Calculate the next run time for a schedule"""
        
        tz = pytz.timezone(schedule.timezone)
        current_time = datetime.now(tz)
        
        if schedule.schedule_type == ScheduleType.ONE_TIME:
            run_time = datetime.fromisoformat(schedule.schedule_config["run_time"])
            if not run_time.tzinfo:
                run_time = tz.localize(run_time)
            return run_time if run_time > current_time else None
        
        elif schedule.schedule_type == ScheduleType.RECURRING:
            return self._calculate_recurring_next_run(schedule, current_time)
        
        elif schedule.schedule_type == ScheduleType.CRON:
            cron_expr = schedule.schedule_config["cron_expression"]
            if CRONITER_AVAILABLE:
                cron = croniter(cron_expr, current_time)
                return cron.get_next(datetime)
            else:
                return self._calculate_cron_next_run_fallback(cron_expr, current_time)
        
        elif schedule.schedule_type == ScheduleType.INTERVAL:
            interval_minutes = schedule.schedule_config["interval_minutes"]
            
            if schedule.last_run:
                return schedule.last_run + timedelta(minutes=interval_minutes)
            else:
                start_time = schedule.schedule_config.get("start_time")
                if start_time:
                    start_dt = datetime.fromisoformat(start_time)
                    if not start_dt.tzinfo:
                        start_dt = tz.localize(start_dt)
                    return max(start_dt, current_time)
                else:
                    return current_time + timedelta(minutes=interval_minutes)
        
        elif schedule.schedule_type == ScheduleType.SMART:
            return self._calculate_smart_next_run(schedule, current_time)
        
        return None
    
    def _calculate_recurring_next_run(self, schedule: ScheduleEntry, current_time: datetime) -> Optional[datetime]:
        """Calculate next run time for recurring schedule"""
        config = schedule.schedule_config
        target_days = [d.lower() for d in config["days"]]
        target_time = config["time"]
        end_date = config.get("end_date")
        
        # Parse target time
        try:
            hour, minute = map(int, target_time.split(":"))
        except ValueError:
            logger.error(f"Invalid time format: {target_time}")
            return None
        
        # Check end date
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
            if not end_dt.tzinfo:
                end_dt = current_time.tzinfo.localize(end_dt)
            if current_time >= end_dt:
                return None
        
        # Find next occurrence
        next_run = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If time has passed today, start from tomorrow
        if next_run <= current_time:
            next_run += timedelta(days=1)
        
        # Find next matching day
        while True:
            weekday = next_run.strftime("%A").lower()
            if weekday in target_days:
                break
            next_run += timedelta(days=1)
            
            # Safety break (shouldn't need more than 7 days)
            if (next_run - current_time).days > 7:
                break
        
        return next_run
    
    def _calculate_smart_next_run(self, schedule: ScheduleEntry, current_time: datetime) -> Optional[datetime]:
        """Calculate next run time for smart schedule"""
        config = schedule.schedule_config
        preferred_times = config.get("preferred_times", ["09:00"])
        
        # Start with preferred times
        for time_str in preferred_times:
            try:
                hour, minute = map(int, time_str.split(":"))
                next_run = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # If time has passed today, try tomorrow
                if next_run <= current_time:
                    next_run += timedelta(days=1)
                
                # TODO: Apply smart optimizations based on user patterns
                return next_run
                
            except ValueError:
                continue
        
        # Fallback to 9 AM tomorrow
        return current_time.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    async def _check_schedule_conditions(self, schedule: ScheduleEntry) -> bool:
        """Check if schedule conditions are met"""
        if not schedule.conditions:
            return True
        
        for condition in schedule.conditions:
            condition_type = condition.get("type")
            params = condition.get("parameters", {})
            
            if condition_type == "weather":
                # TODO: Check weather conditions
                pass
            elif condition_type == "user_presence":
                # TODO: Check user presence
                pass
            elif condition_type == "device_status":
                # TODO: Check device status
                pass
            elif condition_type == "context":
                # TODO: Check context conditions
                pass
        
        return True  # Default to true for now
    
    async def _optimize_smart_schedule(self, schedule: ScheduleEntry):
        """Optimize smart schedule based on user patterns"""
        # TODO: Implement smart scheduling optimization
        # This would analyze user behavior patterns, calendar events,
        # environmental factors, etc. to optimize scheduling times
        pass
    
    def update_user_patterns(self, patterns: Dict[str, Any]):
        """Update user behavior patterns for smart scheduling"""
        self.user_patterns.update(patterns)
        logger.info("Updated user patterns for smart scheduling")
    
    def update_context_data(self, context: Dict[str, Any]):
        """Update context data for smart scheduling"""
        self.context_data.update(context)
    
    def get_schedule(self, schedule_id: str) -> Optional[ScheduleEntry]:
        """Get schedule by ID"""
        return self.schedules.get(schedule_id)
    
    def list_schedules(self, status: ScheduleStatus = None) -> List[ScheduleEntry]:
        """List schedules with optional status filter"""
        schedules = list(self.schedules.values())
        
        if status:
            schedules = [s for s in schedules if s.status == status]
        
        return schedules
    
    def pause_schedule(self, schedule_id: str):
        """Pause a schedule"""
        if schedule_id in self.schedules:
            self.schedules[schedule_id].status = ScheduleStatus.PAUSED
            self._save_schedule(self.schedules[schedule_id])
            logger.info(f"Paused schedule: {schedule_id}")
    
    def resume_schedule(self, schedule_id: str):
        """Resume a paused schedule"""
        if schedule_id in self.schedules:
            schedule = self.schedules[schedule_id]
            schedule.status = ScheduleStatus.ACTIVE
            schedule.next_run = self._calculate_next_run(schedule)
            self._save_schedule(schedule)
            logger.info(f"Resumed schedule: {schedule_id}")
    
    def cancel_schedule(self, schedule_id: str):
        """Cancel a schedule"""
        if schedule_id in self.schedules:
            self.schedules[schedule_id].status = ScheduleStatus.CANCELLED
            self.schedules[schedule_id].next_run = None
            self._save_schedule(self.schedules[schedule_id])
            logger.info(f"Cancelled schedule: {schedule_id}")
    
    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule completely"""
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            
            # Remove file
            filepath = self.data_dir / f"{schedule_id}.json"
            if filepath.exists():
                filepath.unlink()
            
            logger.info(f"Removed schedule: {schedule_id}")
            return True
        
        return False
    
    def get_upcoming_schedules(self, hours: int = 24) -> List[ScheduleEntry]:
        """Get schedules that will run in the next N hours"""
        cutoff_time = datetime.now() + timedelta(hours=hours)
        
        upcoming = []
        for schedule in self.schedules.values():
            if (schedule.status == ScheduleStatus.ACTIVE and
                schedule.next_run and
                schedule.next_run <= cutoff_time):
                upcoming.append(schedule)
        
        # Sort by next run time
        upcoming.sort(key=lambda s: s.next_run)
        return upcoming
    
    def get_schedule_statistics(self) -> Dict[str, Any]:
        """Get schedule statistics"""
        total_schedules = len(self.schedules)
        if total_schedules == 0:
            return {"total_schedules": 0}
        
        active_schedules = len([s for s in self.schedules.values() if s.status == ScheduleStatus.ACTIVE])
        completed_schedules = len([s for s in self.schedules.values() if s.status == ScheduleStatus.COMPLETED])
        
        # Calculate total runs
        total_runs = sum(s.run_count for s in self.schedules.values())
        
        # Count by type
        by_type = {}
        for schedule in self.schedules.values():
            schedule_type = schedule.schedule_type.value
            by_type[schedule_type] = by_type.get(schedule_type, 0) + 1
        
        return {
            "total_schedules": total_schedules,
            "active_schedules": active_schedules,
            "completed_schedules": completed_schedules,
            "total_runs": total_runs,
            "by_type": by_type,
            "upcoming_24h": len(self.get_upcoming_schedules(24))
        }
    
    def _save_schedule(self, schedule: ScheduleEntry):
        """Save schedule to file"""
        filepath = self.data_dir / f"{schedule.id}.json"
        
        schedule_data = asdict(schedule)
        # Convert datetime objects to ISO strings
        if schedule_data["created_at"]:
            schedule_data["created_at"] = schedule_data["created_at"].isoformat()
        if schedule_data["next_run"]:
            schedule_data["next_run"] = schedule_data["next_run"].isoformat()
        if schedule_data["last_run"]:
            schedule_data["last_run"] = schedule_data["last_run"].isoformat()
        
        # Convert enums to strings
        schedule_data["schedule_type"] = schedule_data["schedule_type"].value
        schedule_data["status"] = schedule_data["status"].value
        
        with open(filepath, 'w') as f:
            json.dump(schedule_data, f, indent=2, default=str)
    
    def _load_all_schedules(self):
        """Load all schedules from files"""
        if not self.data_dir.exists():
            return
        
        for filepath in self.data_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Convert strings back to objects
                if data.get("created_at"):
                    data["created_at"] = datetime.fromisoformat(data["created_at"])
                if data.get("next_run"):
                    data["next_run"] = datetime.fromisoformat(data["next_run"])
                if data.get("last_run"):
                    data["last_run"] = datetime.fromisoformat(data["last_run"])
                
                data["schedule_type"] = ScheduleType(data["schedule_type"])
                data["status"] = ScheduleStatus(data["status"])
                
                schedule = ScheduleEntry(**data)
                self.schedules[schedule.id] = schedule
                
            except Exception as e:
                logger.error(f"Failed to load schedule from {filepath}: {e}")
        
        logger.info(f"Loaded {len(self.schedules)} schedules")
    
    def _validate_cron_fallback(self, cron_expression: str):
        """Fallback cron validation for basic patterns"""
        # Basic validation - check if it has 5 parts (minute, hour, day, month, weekday)
        parts = cron_expression.strip().split()
        if len(parts) != 5:
            raise ValueError("Cron expression must have 5 parts")
        
        # Basic range validation
        minute, hour, day, month, weekday = parts
        
        for part, max_val, name in [(minute, 59, 'minute'), (hour, 23, 'hour'), 
                                   (day, 31, 'day'), (month, 12, 'month'), 
                                   (weekday, 7, 'weekday')]:
            if part != '*' and not part.startswith('*/'):
                try:
                    val = int(part)
                    if val < 0 or val > max_val:
                        raise ValueError(f"Invalid {name} value: {val}")
                except ValueError:
                    # Allow ranges and lists for basic compatibility
                    if '-' not in part and ',' not in part:
                        raise ValueError(f"Invalid {name} format: {part}")
    
    def _calculate_cron_next_run_fallback(self, cron_expr: str, current_time: datetime) -> Optional[datetime]:
        """Fallback cron calculation for basic patterns"""
        try:
            parts = cron_expr.strip().split()
            if len(parts) != 5:
                return None
            
            minute, hour, day, month, weekday = parts
            
            # Start with next minute
            next_time = current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
            
            # Very basic implementation - only handles simple cases
            if minute != '*':
                try:
                    target_minute = int(minute)
                    if next_time.minute != target_minute:
                        next_time = next_time.replace(minute=target_minute)
                        if next_time <= current_time:
                            next_time += timedelta(hours=1)
                except ValueError:
                    pass
            
            if hour != '*':
                try:
                    target_hour = int(hour)
                    if next_time.hour != target_hour:
                        next_time = next_time.replace(hour=target_hour, minute=0)
                        if next_time <= current_time:
                            next_time += timedelta(days=1)
                except ValueError:
                    pass
            
            return next_time
            
        except Exception as e:
            logger.error(f"Fallback cron calculation error: {e}")
            # Return next hour as fallback
            return current_time + timedelta(hours=1)

# Example usage
if __name__ == "__main__":
    async def test_schedule_manager():
        manager = ScheduleManager()
        
        # Add sample schedules
        
        # One-time schedule
        one_time = manager.add_one_time_schedule(
            "Morning Meeting Reminder",
            "meeting_reminder_routine",
            datetime.now() + timedelta(minutes=5)
        )
        
        # Recurring schedule
        daily_standup = manager.add_recurring_schedule(
            "Daily Standup",
            "standup_routine",
            ["monday", "tuesday", "wednesday", "thursday", "friday"],
            "09:00"
        )
        
        # Cron schedule
        weekly_report = manager.add_cron_schedule(
            "Weekly Report",
            "report_routine",
            "0 17 * * 5"  # Every Friday at 5 PM
        )
        
        # Interval schedule
        status_check = manager.add_interval_schedule(
            "System Status Check",
            "status_check_routine",
            30  # Every 30 minutes
        )
        
        # Smart schedule
        focus_time = manager.add_smart_schedule(
            "Focus Time",
            "focus_routine",
            ["09:00", "14:00"],
            {"activity": "work", "calendar_free": True}
        )
        
        # Start scheduler
        await manager.start_scheduler()
        
        # List schedules
        print("Active schedules:")
        for schedule in manager.list_schedules(ScheduleStatus.ACTIVE):
            print(f"- {schedule.name}: Next run at {schedule.next_run}")
        
        # Get statistics
        stats = manager.get_schedule_statistics()
        print(f"Schedule statistics: {stats}")
        
        # Simulate running for a bit
        await asyncio.sleep(2)
        
        await manager.stop_scheduler()
    
    # Run test
    asyncio.run(test_schedule_manager())