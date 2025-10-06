#!/usr/bin/env python3
"""
Sample Runner for Task Automation System

Demonstrates the automation features with interactive examples and provides
a simple CLI for testing automation functionality.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from task_automation import AutomationEngine, NaturalLanguageRoutineCreator
from task_automation.automation_engine import RoutineStatus

class AutomationSampleRunner:
    """Interactive sample runner for automation features"""
    
    def __init__(self):
        self.engine = AutomationEngine(data_dir="sample_data/automation")
        self.nlp_creator = NaturalLanguageRoutineCreator(self.engine.routine_builder)
        
    def print_header(self, title):
        """Print formatted header"""
        print("\n" + "=" * 60)
        print(f" {title}")
        print("=" * 60)
    
    def print_section(self, title):
        """Print formatted section header"""
        print(f"\n--- {title} ---")
    
    async def run_samples(self):
        """Run all sample demonstrations"""
        self.print_header("JARVIS Task Automation - Sample Runner")
        
        await self.sample_nlp_routine_creation()
        await self.sample_predefined_routines()
        await self.sample_routine_execution()
        await self.sample_routine_management()
        
        self.print_header("Sample Run Complete")
        print("Check 'sample_data/automation/' for saved routine files.")
    
    async def sample_nlp_routine_creation(self):
        """Demonstrate natural language routine creation"""
        self.print_section("Natural Language Routine Creation")
        
        examples = [
            "Every weekday at 7 am turn on the office lights, start the coffee machine, and read me the weather and my first three calendar events",
            "When I sit at my desk and it's after 9am, put me in focus mode and mute all notifications",
            "If the living room temperature goes above 26 degrees, turn on the AC and close the blinds",
            "At sunset, turn on ambient lights and play relaxing music",
            "When I arrive home, disarm security, turn on entry lights, and set thermostat to comfort mode"
        ]
        
        print("Creating routines from natural language descriptions...\n")
        
        for i, example in enumerate(examples, 1):
            print(f"{i}. Input: \"{example}\"")
            
            try:
                result = self.nlp_creator.parse_natural_language(example)
                routine = result["routine"]
                
                # Add to engine
                self.engine.add_routine(routine)
                
                print(f"   ✓ Created: {routine.name}")
                print(f"   ✓ Tasks: {len(routine.tasks)}")
                print(f"   ✓ Triggers: {len(routine.triggers)}")
                print(f"   ✓ Time triggers: {len(result['time_triggers'])}")
                print(f"   ✓ Conditions: {len(result['conditions'])}")
                
                # Show parsed actions
                if result['actions']:
                    action_names = [a['name'] for a in result['actions'][:3]]
                    more = f" (and {len(result['actions'])-3} more)" if len(result['actions']) > 3 else ""
                    print(f"   ✓ Actions: {', '.join(action_names)}{more}")
                
            except Exception as e:
                print(f"   ✗ Error: {e}")
            
            print()
    
    async def sample_predefined_routines(self):
        """Demonstrate predefined routine templates"""
        self.print_section("Predefined Routine Templates")
        
        print("Creating sample routine templates...\n")
        
        templates = [
            ("Morning Routine", self.engine.routine_builder.create_morning_routine),
            ("Work Focus Mode", self.engine.routine_builder.create_work_focus_routine),
            ("Evening Wind Down", self.engine.routine_builder.create_evening_routine)
        ]
        
        for name, creator_func in templates:
            try:
                routine = creator_func()
                self.engine.add_routine(routine)
                
                print(f"✓ {name}")
                print(f"   Category: {routine.category}")
                print(f"   Tasks: {len(routine.tasks)}")
                print(f"   Triggers: {len(routine.triggers)}")
                
                # Show first few tasks
                if routine.tasks:
                    task_names = [t.name for t in routine.tasks[:3]]
                    more = f" (and {len(routine.tasks)-3} more)" if len(routine.tasks) > 3 else ""
                    print(f"   Sample tasks: {', '.join(task_names)}{more}")
                
            except Exception as e:
                print(f"✗ Error creating {name}: {e}")
            
            print()
    
    async def sample_routine_execution(self):
        """Demonstrate routine execution"""
        self.print_section("Routine Execution")
        
        routines = list(self.engine.routines.values())[:3]  # Execute first 3 routines
        
        if not routines:
            print("No routines available for execution.")
            return
        
        print("Executing sample routines...\n")
        
        for routine in routines:
            print(f"Executing: {routine.name}")
            print(f"Description: {routine.description}")
            
            try:
                start_time = asyncio.get_event_loop().time()
                result = await self.engine.execute_routine(routine.id)
                end_time = asyncio.get_event_loop().time()
                
                print(f"   ✓ Status: {result['status'].value}")
                print(f"   ✓ Duration: {end_time - start_time:.2f}s")
                print(f"   ✓ Completed tasks: {len(result['completed_tasks'])}")
                print(f"   ✓ Failed tasks: {len(result['failed_tasks'])}")
                
                if result.get('failed_tasks'):
                    print(f"   ⚠ Failures: {result['failed_tasks']}")
                
            except Exception as e:
                print(f"   ✗ Execution failed: {e}")
            
            print()
    
    async def sample_routine_management(self):
        """Demonstrate routine management features"""
        self.print_section("Routine Management")
        
        # List all routines
        print("All routines:")
        routine_list = self.engine.list_routines()
        
        for routine_info in routine_list:
            status = "✓ Enabled" if routine_info['enabled'] else "✗ Disabled"
            print(f"   • {routine_info['name']} ({routine_info['category']}) - {status}")
            print(f"     Tasks: {routine_info['task_count']}, Triggers: {routine_info['trigger_count']}")
        
        print(f"\nTotal routines: {len(routine_list)}")
        
        # Group by category
        print("\nRoutines by category:")
        categories = set(r['category'] for r in routine_list)
        for category in sorted(categories):
            category_routines = self.engine.get_routines_by_category(category)
            print(f"   {category}: {len(category_routines)} routines")
            for routine in category_routines:
                print(f"     - {routine.name}")
        
        # Execution history
        print("\nRecent execution history:")
        history = self.engine.get_execution_history(limit=5)
        
        if history:
            for execution in history[-5:]:  # Last 5 executions
                routine_id = execution.get('routine_id', 'Unknown')
                routine = self.engine.get_routine(routine_id)
                routine_name = routine.name if routine else 'Unknown'
                
                status = execution.get('status', 'Unknown')
                duration = execution.get('duration', 0)
                
                print(f"   • {routine_name}: {status} ({duration:.2f}s)")
        else:
            print("   No execution history available.")
    
    async def interactive_mode(self):
        """Run interactive mode for testing"""
        self.print_header("Interactive Automation Testing")
        
        while True:
            print("\nOptions:")
            print("1. Create routine from natural language")
            print("2. List all routines")
            print("3. Execute routine by name")
            print("4. Run all samples")
            print("5. Exit")
            
            try:
                choice = input("\nEnter choice (1-5): ").strip()
                
                if choice == "1":
                    await self.interactive_create_routine()
                elif choice == "2":
                    await self.interactive_list_routines()
                elif choice == "3":
                    await self.interactive_execute_routine()
                elif choice == "4":
                    await self.run_samples()
                elif choice == "5":
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def interactive_create_routine(self):
        """Interactive routine creation"""
        print("\n--- Create Routine from Natural Language ---")
        text = input("Describe your routine: ").strip()
        
        if not text:
            print("Empty input, skipping.")
            return
        
        try:
            result = self.nlp_creator.parse_natural_language(text)
            routine = result["routine"]
            self.engine.add_routine(routine)
            
            print(f"\n✓ Created routine: {routine.name}")
            print(f"✓ Tasks: {len(routine.tasks)}")
            print(f"✓ Triggers: {len(routine.triggers)}")
            
            # Show details
            if routine.tasks:
                print("\nTasks:")
                for i, task in enumerate(routine.tasks, 1):
                    print(f"   {i}. {task.name} ({task.task_type.value})")
            
            if routine.triggers:
                print("\nTriggers:")
                for i, trigger in enumerate(routine.triggers, 1):
                    print(f"   {i}. {trigger['type']}: {trigger['parameters']}")
            
        except Exception as e:
            print(f"✗ Error creating routine: {e}")
    
    async def interactive_list_routines(self):
        """Interactive routine listing"""
        print("\n--- All Routines ---")
        
        routine_list = self.engine.list_routines()
        
        if not routine_list:
            print("No routines found.")
            return
        
        for i, routine_info in enumerate(routine_list, 1):
            status = "Enabled" if routine_info['enabled'] else "Disabled"
            print(f"{i}. {routine_info['name']} ({routine_info['category']}) - {status}")
            print(f"   Description: {routine_info['description']}")
            print(f"   Tasks: {routine_info['task_count']}, Triggers: {routine_info['trigger_count']}")
            print()
    
    async def interactive_execute_routine(self):
        """Interactive routine execution"""
        print("\n--- Execute Routine ---")
        
        routine_list = self.engine.list_routines()
        
        if not routine_list:
            print("No routines available.")
            return
        
        print("Available routines:")
        for i, routine_info in enumerate(routine_list, 1):
            print(f"{i}. {routine_info['name']}")
        
        try:
            choice = input(f"\nEnter routine number (1-{len(routine_list)}): ").strip()
            index = int(choice) - 1
            
            if 0 <= index < len(routine_list):
                routine_info = routine_list[index]
                routine_id = routine_info['id']
                
                print(f"\nExecuting: {routine_info['name']}")
                
                start_time = asyncio.get_event_loop().time()
                result = await self.engine.execute_routine(routine_id)
                end_time = asyncio.get_event_loop().time()
                
                print(f"✓ Status: {result['status'].value}")
                print(f"✓ Duration: {end_time - start_time:.2f}s")
                print(f"✓ Completed tasks: {len(result['completed_tasks'])}")
                
                if result.get('failed_tasks'):
                    print(f"⚠ Failed tasks: {len(result['failed_tasks'])}")
                    for failure in result['failed_tasks']:
                        print(f"   - {failure}")
            else:
                print("Invalid routine number.")
                
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error executing routine: {e}")


async def main():
    """Main entry point"""
    runner = AutomationSampleRunner()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "samples":
            await runner.run_samples()
        elif mode == "interactive":
            await runner.interactive_mode()
        elif mode == "nlp":
            await runner.sample_nlp_routine_creation()
        elif mode == "execute":
            await runner.sample_routine_execution()
        else:
            print("Usage: python sample_runner.py [samples|interactive|nlp|execute]")
            sys.exit(1)
    else:
        # Default: run interactive mode
        await runner.interactive_mode()


if __name__ == "__main__":
    print("JARVIS Task Automation - Sample Runner")
    print("Available modes:")
    print("  samples     - Run all demonstration samples")
    print("  interactive - Interactive testing mode")
    print("  nlp        - Natural language parsing demo only")
    print("  execute    - Routine execution demo only")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)