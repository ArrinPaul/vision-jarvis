"""
JARVIS Task Executor

Handles execution of different types of automated tasks with monitoring,
error handling, and integration with all JARVIS modules.
"""

import asyncio
import logging
import subprocess
import threading
from datetime import datetime, timedelta  
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json
import time
import os
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Status of task execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"

class ExecutionPriority(Enum):
    """Priority levels for task execution"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TaskExecution:
    """Task execution instance"""
    execution_id: str
    task_id: str
    routine_id: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    result: Dict[str, Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.result is None:
            self.result = {}
        if self.context is None:
            self.context = {}

class TaskExecutor:
    """Executes automation tasks with monitoring and error handling"""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running_tasks: Dict[str, TaskExecution] = {}
        self.completed_tasks: List[TaskExecution] = []
        
        # Module integrations
        self.voice_assistant = None
        self.smart_home = None
        self.gesture_recognition = None
        self.ar_interface = None
        self.object_detection = None
        self.user_profiles = None
        
        # Task executors for different types
        self.task_handlers: Dict[str, Callable] = {
            "voice_command": self._execute_voice_command,
            "gesture_action": self._execute_gesture_action,
            "smart_home_control": self._execute_smart_home_control,
            "system_command": self._execute_system_command,
            "ar_interface_update": self._execute_ar_interface_update,
            "notification": self._execute_notification,
            "delay": self._execute_delay,
            "condition": self._execute_condition,
            "loop": self._execute_loop,
            "api_call": self._execute_api_call,
            "file_operation": self._execute_file_operation,
            "user_input": self._execute_user_input,
            "email": self._execute_email,
            "web_automation": self._execute_web_automation,
            "data_processing": self._execute_data_processing,
            "ml_inference": self._execute_ml_inference
        }
        
        # Notification callbacks
        self.notification_callbacks: List[Callable] = []
        
        logger.info("TaskExecutor initialized")
    
    def set_module_integrations(self, 
                              voice_assistant=None,
                              smart_home=None, 
                              gesture_recognition=None,
                              ar_interface=None,
                              object_detection=None,
                              user_profiles=None):
        """Set module integrations for task execution"""
        self.voice_assistant = voice_assistant
        self.smart_home = smart_home
        self.gesture_recognition = gesture_recognition
        self.ar_interface = ar_interface
        self.object_detection = object_detection
        self.user_profiles = user_profiles
        
        logger.info("Module integrations configured")
    
    def add_notification_callback(self, callback: Callable):
        """Add callback for task completion notifications"""
        self.notification_callbacks.append(callback)
    
    async def execute_task(self, 
                          task_id: str,
                          routine_id: str,
                          task_type: str,
                          parameters: Dict[str, Any],
                          timeout: int = 30,
                          retry_count: int = 3,
                          priority: ExecutionPriority = ExecutionPriority.NORMAL,
                          context: Dict[str, Any] = None) -> TaskExecution:
        """Execute a single task"""
        
        # Check concurrent task limit
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            raise Exception("Maximum concurrent tasks reached")
        
        execution_id = f"{task_id}_{int(time.time())}"
        start_time = datetime.now()
        
        execution = TaskExecution(
            execution_id=execution_id,
            task_id=task_id,
            routine_id=routine_id,
            status=TaskStatus.PENDING,
            start_time=start_time,
            context=context or {}
        )
        
        self.running_tasks[execution_id] = execution
        
        try:
            logger.info(f"Starting task execution: {task_type} ({execution_id})")
            execution.status = TaskStatus.RUNNING
            
            # Get task handler
            handler = self.task_handlers.get(task_type)
            if not handler:
                raise ValueError(f"No handler for task type: {task_type}")
            
            # Execute with retry logic
            for attempt in range(retry_count + 1):
                try:
                    execution.retry_count = attempt
                    
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        handler(parameters, execution),
                        timeout=timeout
                    )
                    
                    # Success
                    execution.status = TaskStatus.COMPLETED
                    execution.result = result
                    break
                    
                except asyncio.TimeoutError:
                    if attempt == retry_count:
                        execution.status = TaskStatus.TIMEOUT
                        execution.error = f"Task timed out after {timeout} seconds"
                        break
                    else:
                        execution.status = TaskStatus.RETRYING
                        logger.warning(f"Task {execution_id} timed out, retrying ({attempt + 1}/{retry_count})")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        
                except Exception as e:
                    if attempt == retry_count:
                        execution.status = TaskStatus.FAILED
                        execution.error = str(e)
                        break
                    else:
                        execution.status = TaskStatus.RETRYING
                        logger.warning(f"Task {execution_id} failed, retrying ({attempt + 1}/{retry_count}): {e}")
                        await asyncio.sleep(2 ** attempt)
            
        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error = str(e)
            logger.error(f"Task execution failed: {execution_id} - {e}")
        
        finally:
            # Finalize execution
            execution.end_time = datetime.now()
            execution.duration = (execution.end_time - execution.start_time).total_seconds()
            
            # Move to completed tasks
            if execution_id in self.running_tasks:
                del self.running_tasks[execution_id]
            
            self.completed_tasks.append(execution)
            
            # Keep only last 1000 completed tasks
            if len(self.completed_tasks) > 1000:
                self.completed_tasks = self.completed_tasks[-1000:]
            
            # Notify callbacks
            for callback in self.notification_callbacks:
                try:
                    await callback(execution)
                except Exception as e:
                    logger.error(f"Notification callback failed: {e}")
            
            logger.info(f"Task execution completed: {execution_id} - {execution.status.value}")
        
        return execution
    
    # Task Handlers
    async def _execute_voice_command(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute voice command task"""
        command = parameters.get("command")
        wait_for_response = parameters.get("wait_for_response", True)
        timeout = parameters.get("timeout", 10)
        
        if not command:
            raise ValueError("No command specified")
        
        logger.info(f"Executing voice command: {command}")
        
        if self.voice_assistant:
            try:
                # Use actual voice assistant if available
                response = await self.voice_assistant.process_command(command)
                return {
                    "command": command,
                    "response": response,
                    "success": True,
                    "execution_time": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Voice assistant error: {e}")
                # Fall back to simulation
        
        # Simulate voice command execution
        await asyncio.sleep(0.5)
        return {
            "command": command,
            "response": f"Executed: {command}",
            "success": True,
            "simulated": True
        }
    
    async def _execute_gesture_action(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute gesture action task"""
        gesture_name = parameters.get("gesture_name")
        confirmation = parameters.get("confirmation_required", False)
        
        if not gesture_name:
            raise ValueError("No gesture specified")
        
        logger.info(f"Executing gesture action: {gesture_name}")
        
        if self.gesture_recognition:
            try:
                # Use actual gesture recognition if available
                result = await self.gesture_recognition.execute_gesture_action(gesture_name)
                return {
                    "gesture": gesture_name,
                    "result": result,
                    "success": True
                }
            except Exception as e:
                logger.error(f"Gesture recognition error: {e}")
        
        # Simulate gesture execution
        await asyncio.sleep(0.3)
        return {
            "gesture": gesture_name,
            "executed": True,
            "simulated": True
        }
    
    async def _execute_smart_home_control(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute smart home control task"""
        device_type = parameters.get("device_type")
        action = parameters.get("action")
        device_id = parameters.get("device_id")
        
        if not device_type or not action:
            raise ValueError("Device type and action required")
        
        logger.info(f"Smart home control: {action} {device_type}")
        
        if self.smart_home:
            try:
                # Use actual smart home integration
                result = await self.smart_home.control_device(
                    device_type=device_type,
                    action=action,
                    device_id=device_id,
                    **{k: v for k, v in parameters.items() if k not in ['device_type', 'action', 'device_id']}
                )
                return {
                    "device_type": device_type,
                    "action": action,
                    "result": result,
                    "success": True
                }
            except Exception as e:
                logger.error(f"Smart home error: {e}")
        
        # Simulate smart home control
        await asyncio.sleep(0.5)
        return {
            "device_type": device_type,
            "action": action,
            "success": True,
            "simulated": True
        }
    
    async def _execute_system_command(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute system command task"""
        command = parameters.get("command")
        args = parameters.get("args", [])
        shell = parameters.get("shell", False)
        
        if not command:
            raise ValueError("No command specified")
        
        logger.info(f"Executing system command: {command}")
        
        try:
            # Handle special commands
            if command == "close_apps":
                apps = parameters.get("apps", [])
                return await self._close_applications(apps)
            elif command == "enable_dnd":
                duration = parameters.get("duration_minutes", 60)
                return await self._enable_do_not_disturb(duration)
            elif command == "set_sleep_timer":
                duration = parameters.get("duration_minutes", 60)
                return await self._set_sleep_timer(duration)
            else:
                # Execute actual system command
                if isinstance(args, list):
                    full_command = [command] + args
                else:
                    full_command = command
                
                process = await asyncio.create_subprocess_exec(
                    *full_command if isinstance(full_command, list) else full_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    shell=shell
                )
                
                stdout, stderr = await process.communicate()
                
                return {
                    "command": command,
                    "return_code": process.returncode,
                    "stdout": stdout.decode() if stdout else "",
                    "stderr": stderr.decode() if stderr else "",
                    "success": process.returncode == 0
                }
                
        except Exception as e:
            logger.error(f"System command error: {e}")
            return {
                "command": command,
                "error": str(e),
                "success": False
            }
    
    async def _execute_ar_interface_update(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute AR interface update task"""
        update_type = parameters.get("update_type")
        data = parameters.get("data", {})
        
        if not update_type:
            raise ValueError("No update type specified")
        
        logger.info(f"Updating AR interface: {update_type}")
        
        if self.ar_interface:
            try:
                result = await self.ar_interface.update_interface(update_type, data)
                return {
                    "update_type": update_type,
                    "result": result,
                    "success": True
                }
            except Exception as e:
                logger.error(f"AR interface error: {e}")
        
        # Simulate AR update
        await asyncio.sleep(0.3)
        return {
            "update_type": update_type,
            "success": True,
            "simulated": True
        }
    
    async def _execute_notification(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute notification task"""
        message = parameters.get("message")
        notification_type = parameters.get("type", "info")
        duration = parameters.get("duration", 5)
        show_avatar = parameters.get("show_avatar", True)
        
        if not message:
            raise ValueError("No message specified")
        
        logger.info(f"Showing notification: {message}")
        
        # TODO: Integrate with actual notification system
        # For now, just log the notification
        print(f"NOTIFICATION [{notification_type.upper()}]: {message}")
        
        # Simulate notification display
        await asyncio.sleep(0.1)
        
        return {
            "message": message,
            "type": notification_type,
            "duration": duration,
            "displayed": True
        }
    
    async def _execute_delay(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute delay task"""
        duration = parameters.get("duration_seconds", 1)
        
        logger.info(f"Waiting for {duration} seconds")
        await asyncio.sleep(duration)
        
        return {
            "duration": duration,
            "completed": True
        }
    
    async def _execute_condition(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute condition evaluation task"""
        condition_type = parameters.get("condition_type")
        condition_params = parameters.get("condition_params", {})
        
        logger.info(f"Evaluating condition: {condition_type}")
        
        # Evaluate condition based on type
        result = await self._evaluate_condition(condition_type, condition_params, execution.context)
        
        return {
            "condition_type": condition_type,
            "condition_result": result,
            "evaluated": True
        }
    
    async def _execute_loop(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute loop task"""
        task_ids = parameters.get("task_ids", [])
        condition = parameters.get("condition", {})
        max_iterations = parameters.get("max_iterations", 10)
        
        logger.info(f"Executing loop with {len(task_ids)} tasks, max {max_iterations} iterations")
        
        iterations = 0
        results = []
        
        while iterations < max_iterations:
            # Check loop condition
            should_continue = await self._evaluate_loop_condition(condition, execution.context, iterations)
            if not should_continue:
                break
            
            # Execute tasks in loop
            for task_id in task_ids:
                # TODO: Execute subtasks
                await asyncio.sleep(0.1)  # Simulate task execution
            
            iterations += 1
            results.append(f"Iteration {iterations} completed")
        
        return {
            "iterations": iterations,
            "results": results,
            "completed": True
        }
    
    async def _execute_api_call(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute API call task"""
        url = parameters.get("url")
        method = parameters.get("method", "GET").upper()
        headers = parameters.get("headers", {})
        data = parameters.get("data")
        json_data = parameters.get("json")
        timeout = parameters.get("timeout", 10)
        
        if not url:
            raise ValueError("No URL specified")
        
        logger.info(f"Making {method} API call to {url}")
        
        try:
            # Make the actual API call
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                json=json_data,
                timeout=timeout
            )
            
            # Try to parse JSON response
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = response.text
            
            return {
                "url": url,
                "method": method,
                "status_code": response.status_code,
                "response_data": response_data,
                "success": response.status_code < 400
            }
            
        except requests.RequestException as e:
            logger.error(f"API call failed: {e}")
            return {
                "url": url,
                "method": method,
                "error": str(e),
                "success": False
            }
    
    async def _execute_file_operation(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute file operation task"""
        operation = parameters.get("operation")
        filepath = parameters.get("filepath")
        content = parameters.get("content")
        destination = parameters.get("destination")
        
        if not operation or not filepath:
            raise ValueError("Operation and filepath required")
        
        logger.info(f"File operation: {operation} on {filepath}")
        
        try:
            if operation == "read":
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {
                    "operation": operation,
                    "filepath": filepath,
                    "content": content,
                    "success": True
                }
            
            elif operation == "write":
                if content is None:
                    raise ValueError("Content required for write operation")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(str(content))
                return {
                    "operation": operation,
                    "filepath": filepath,
                    "success": True
                }
            
            elif operation == "copy":
                if not destination:
                    raise ValueError("Destination required for copy operation")
                import shutil
                shutil.copy2(filepath, destination)
                return {
                    "operation": operation,
                    "filepath": filepath,
                    "destination": destination,
                    "success": True
                }
            
            elif operation == "move":
                if not destination:
                    raise ValueError("Destination required for move operation")
                import shutil
                shutil.move(filepath, destination)
                return {
                    "operation": operation,
                    "filepath": filepath,
                    "destination": destination,
                    "success": True
                }
            
            elif operation == "delete":
                os.remove(filepath)
                return {
                    "operation": operation,
                    "filepath": filepath,
                    "success": True
                }
            
            else:
                raise ValueError(f"Unsupported file operation: {operation}")
                
        except Exception as e:
            logger.error(f"File operation failed: {e}")
            return {
                "operation": operation,
                "filepath": filepath,
                "error": str(e),
                "success": False
            }
    
    async def _execute_user_input(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute user input task"""
        prompt = parameters.get("prompt", "Please provide input:")
        input_type = parameters.get("input_type", "text")
        timeout = parameters.get("timeout", 30)
        
        logger.info(f"Requesting user input: {prompt}")
        
        # TODO: Integrate with actual UI for user input
        # For now, simulate user input
        await asyncio.sleep(1.0)
        
        simulated_input = "user provided input"
        
        return {
            "prompt": prompt,
            "input_type": input_type,
            "user_input": simulated_input,
            "success": True,
            "simulated": True
        }
    
    async def _execute_email(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute email task"""
        to = parameters.get("to")
        subject = parameters.get("subject")
        body = parameters.get("body")
        cc = parameters.get("cc")
        bcc = parameters.get("bcc")
        
        if not to or not subject:
            raise ValueError("Recipient and subject required")
        
        logger.info(f"Sending email to {to}: {subject}")
        
        # TODO: Integrate with actual email service
        # For now, simulate email sending
        await asyncio.sleep(0.5)
        
        return {
            "to": to,
            "subject": subject,
            "sent": True,
            "success": True,
            "simulated": True
        }
    
    async def _execute_web_automation(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute web automation task"""
        action = parameters.get("action")
        url = parameters.get("url")
        element = parameters.get("element")
        value = parameters.get("value")
        
        logger.info(f"Web automation: {action} on {url}")
        
        # TODO: Integrate with web automation framework (like Selenium)
        # For now, simulate web automation
        await asyncio.sleep(1.0)
        
        return {
            "action": action,
            "url": url,
            "success": True,
            "simulated": True
        }
    
    async def _execute_data_processing(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute data processing task"""
        operation = parameters.get("operation")
        data_source = parameters.get("data_source")
        output_format = parameters.get("output_format", "json")
        
        logger.info(f"Data processing: {operation} on {data_source}")
        
        # TODO: Implement actual data processing
        # For now, simulate data processing
        await asyncio.sleep(0.8)
        
        return {
            "operation": operation,
            "data_source": data_source,
            "output_format": output_format,
            "processed": True,
            "success": True,
            "simulated": True
        }
    
    async def _execute_ml_inference(self, parameters: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Execute ML inference task"""
        model_name = parameters.get("model_name")
        input_data = parameters.get("input_data")
        confidence_threshold = parameters.get("confidence_threshold", 0.5)
        
        logger.info(f"ML inference: {model_name}")
        
        # TODO: Integrate with actual ML models
        # For now, simulate ML inference
        await asyncio.sleep(0.6)
        
        return {
            "model_name": model_name,
            "confidence": 0.87,
            "prediction": "sample_prediction",
            "success": True,
            "simulated": True
        }
    
    # Helper methods
    async def _close_applications(self, apps: List[str]) -> Dict[str, Any]:
        """Close specified applications"""
        logger.info(f"Closing applications: {apps}")
        
        closed_apps = []
        for app in apps:
            try:
                # Platform-specific app closing logic
                if os.name == 'nt':  # Windows
                    subprocess.run(['taskkill', '/f', '/im', f'{app}.exe'], 
                                 capture_output=True, check=False)
                else:  # Unix-like
                    subprocess.run(['pkill', '-f', app], 
                                 capture_output=True, check=False)
                closed_apps.append(app)
            except Exception as e:
                logger.error(f"Failed to close {app}: {e}")
        
        return {
            "apps": apps,
            "closed_apps": closed_apps,
            "success": True
        }
    
    async def _enable_do_not_disturb(self, duration_minutes: int) -> Dict[str, Any]:
        """Enable do not disturb mode"""
        logger.info(f"Enabling DND for {duration_minutes} minutes")
        
        # TODO: Integrate with OS notification settings
        # For now, simulate DND activation
        await asyncio.sleep(0.2)
        
        return {
            "duration_minutes": duration_minutes,
            "enabled": True,
            "success": True,
            "simulated": True
        }
    
    async def _set_sleep_timer(self, duration_minutes: int) -> Dict[str, Any]:
        """Set system sleep timer"""
        logger.info(f"Setting sleep timer for {duration_minutes} minutes")
        
        # TODO: Integrate with OS power management
        # For now, simulate sleep timer
        await asyncio.sleep(0.1)
        
        return {
            "duration_minutes": duration_minutes,
            "timer_set": True,
            "success": True,
            "simulated": True
        }
    
    async def _evaluate_condition(self, condition_type: str, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a condition"""
        # Basic condition evaluation
        if condition_type == "time_based":
            # Check time range
            time_range = params.get("time_range", [])
            if len(time_range) == 2:
                current_time = datetime.now().time()
                start_time = datetime.strptime(time_range[0], "%H:%M").time()
                end_time = datetime.strptime(time_range[1], "%H:%M").time()
                return start_time <= current_time <= end_time
        
        elif condition_type == "user_present":
            # TODO: Check user presence
            return True  # Simulate user present
        
        elif condition_type == "device_status":
            # TODO: Check device status
            return True  # Simulate device check
        
        # Default to true for unknown conditions
        return True
    
    async def _evaluate_loop_condition(self, condition: Dict[str, Any], context: Dict[str, Any], iteration: int) -> bool:
        """Evaluate loop continuation condition"""
        condition_type = condition.get("type", "count")
        
        if condition_type == "count":
            max_count = condition.get("max_count", 5)
            return iteration < max_count
        
        elif condition_type == "while":
            # TODO: Evaluate while condition
            return iteration < 3  # Simulate condition
        
        return False
    
    def cancel_task(self, execution_id: str) -> bool:
        """Cancel a running task"""
        if execution_id in self.running_tasks:
            execution = self.running_tasks[execution_id]
            execution.status = TaskStatus.CANCELLED
            execution.end_time = datetime.now()
            execution.duration = (execution.end_time - execution.start_time).total_seconds()
            
            # Move to completed tasks
            del self.running_tasks[execution_id]
            self.completed_tasks.append(execution)
            
            logger.info(f"Cancelled task: {execution_id}")
            return True
        
        return False
    
    def get_running_tasks(self) -> List[TaskExecution]:
        """Get list of currently running tasks"""
        return list(self.running_tasks.values())
    
    def get_completed_tasks(self, limit: int = 50) -> List[TaskExecution]:
        """Get list of completed tasks"""
        return self.completed_tasks[-limit:]
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get task execution statistics"""
        total_tasks = len(self.completed_tasks)
        if total_tasks == 0:
            return {"total_tasks": 0}
        
        successful_tasks = len([t for t in self.completed_tasks if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in self.completed_tasks if t.status == TaskStatus.FAILED])
        
        durations = [t.duration for t in self.completed_tasks if t.duration]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_tasks": total_tasks,
            "running_tasks": len(self.running_tasks),
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "average_duration": avg_duration,
            "current_load": len(self.running_tasks) / self.max_concurrent_tasks
        }

# Example usage
if __name__ == "__main__":
    async def test_task_executor():
        executor = TaskExecutor(max_concurrent_tasks=5)
        
        # Execute sample tasks
        tasks = [
            executor.execute_task(
                task_id="voice_test",
                routine_id="test_routine",
                task_type="voice_command",
                parameters={"command": "Hello JARVIS"},
                timeout=10
            ),
            executor.execute_task(
                task_id="delay_test",
                routine_id="test_routine",
                task_type="delay",
                parameters={"duration_seconds": 2}
            ),
            executor.execute_task(
                task_id="notification_test",
                routine_id="test_routine",
                task_type="notification",
                parameters={
                    "message": "Task execution test completed",
                    "type": "success"
                }
            )
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Print results
        for result in results:
            print(f"Task {result.task_id}: {result.status.value}")
        
        # Get statistics
        stats = executor.get_task_statistics()
        print(f"Task statistics: {stats}")
    
    # Run test
    asyncio.run(test_task_executor())