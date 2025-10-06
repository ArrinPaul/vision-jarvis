#!/usr/bin/env python3
"""
Jarvis Demo Script - Showcase all new Jarvis-like features

This script demonstrates the enhanced voice assistant capabilities:
- Structured LLM commands
- Memory and context awareness
- Performance metrics
- Proactive briefings and routines
- Advanced backends (ASR, TTS, wake word)

Usage:
    python jarvis_demo.py [--config voice_config.json]
"""

import argparse
import time
import json
import logging
from typing import Dict, Any

# Import Jarvis components
try:
    from memory_store import create_memory_store
    from context_oracle import create_context_oracle
    from metrics import get_metrics_collector, get_jarvis_readiness_report
    from briefing_system import create_briefing_system, create_routine_manager
    from action_registry import ActionRegistry
    from llm_service import LLMService
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all Jarvis modules are available")
    exit(1)


def setup_logging():
    """Setup demo logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("jarvis_demo.log"),
            logging.StreamHandler()
        ]
    )


def demo_memory_system():
    """Demonstrate memory and context features"""
    print("\n" + "="*60)
    print("DEMO: Memory and Context System")
    print("="*60)
    
    # Create memory store
    memory = create_memory_store(max_entries=50, persistence_file="demo_memory.json")
    
    # Simulate some actions
    print("Simulating user actions...")
    memory.remember_action("set_volume", {"level": 50}, "Volume set to 50%", True)
    memory.remember_action("open", {"app": "notepad"}, "Opened Notepad", True)
    memory.remember_preference("preferred_volume", 50)
    memory.remember_preference("morning_app", "outlook")
    
    # Test follow-up handling
    print("\nTesting follow-up queries:")
    test_queries = [
        "same as before",
        "undo that",
        "what did I do last"
    ]
    
    for query in test_queries:
        response = memory.handle_follow_up(query)
        print(f"Query: '{query}' -> Response: {response}")
    
    # Show recent actions
    print(f"\nRecent actions: {len(memory.get_recent_actions())} stored")
    for action in memory.get_recent_actions(3):
        print(f"  - {action['action']}: {action['result']}")
    
    print(f"Context summary: {memory.get_context_summary()}")


def demo_context_oracle():
    """Demonstrate context awareness"""
    print("\n" + "="*60)
    print("DEMO: Context Oracle (Situational Awareness)")
    print("="*60)
    
    oracle = create_context_oracle(update_interval=1.0)
    oracle.start_monitoring()
    
    print("Starting context monitoring...")
    time.sleep(2)
    
    # Show current context
    context = oracle.get_current_context()
    print(f"Active window: {context['active_window']['title']}")
    print(f"System state: CPU {context['system_state']['cpu_usage']:.1f}%, "
          f"Memory {context['system_state']['memory_usage']:.1f}%")
    
    # Test contextual responses
    print(f"Should interrupt user: {oracle.should_interrupt_user()}")
    print(f"Contextual greeting: {oracle.get_contextual_greeting()}")
    print(f"Status summary: {oracle.get_status_summary()}")
    
    # Test focus mode
    oracle.set_focus_mode(True)
    print(f"Focus mode enabled - Should interrupt: {oracle.should_interrupt_user()}")
    
    oracle.stop_monitoring()


def demo_performance_metrics():
    """Demonstrate performance tracking"""
    print("\n" + "="*60)
    print("DEMO: Performance Metrics and Jarvis Readiness")
    print("="*60)
    
    metrics = get_metrics_collector()
    
    # Simulate some performance data
    print("Simulating performance measurements...")
    
    # Simulate various latencies
    metrics.record_metric("wake_to_listen", 120, {"backend": "openwakeword"})
    metrics.record_metric("asr_latency", 250, {"backend": "faster_whisper"})
    metrics.record_metric("llm_response", 800, {"model": "gemini"})
    metrics.record_metric("tts_start", 180, {"backend": "edge_tts"})
    metrics.record_metric("gesture_fps", 28, {"resolution": "640x480"})
    metrics.record_metric("command_dispatch", 35, {"action": "set_volume"})
    
    # Simulate quality metrics
    metrics.record_metric("asr_accuracy", 0.96, {"language": "en"})
    metrics.record_metric("gesture_confidence", 0.85, {"gesture": "pinch"})
    metrics.record_metric("command_success_rate", 0.92, {"session": "demo"})
    
    # Generate readiness report
    print("\nJarvis Readiness Report:")
    print("-" * 40)
    report = get_jarvis_readiness_report()
    print(report)
    
    # Export metrics
    metrics.export_metrics("demo_metrics.json")
    print("\nMetrics exported to demo_metrics.json")


def demo_briefing_system():
    """Demonstrate proactive briefings"""
    print("\n" + "="*60)
    print("DEMO: Briefing System")
    print("="*60)
    
    # Create briefing system
    config = {
        "briefings_enabled": True,
        "briefing_interval_hours": 4,
        "morning_briefing_hour": 8
    }
    
    briefing_system = create_briefing_system(config)
    
    # Register some info providers
    briefing_system.register_info_provider("time", lambda: "3:45 PM")
    briefing_system.register_info_provider("weather", lambda: "Sunny, 72°F")
    briefing_system.register_info_provider("system_status", lambda: "All systems normal")
    
    # Generate different types of briefings
    print("Morning briefing:")
    morning = briefing_system.generate_morning_briefing()
    print(f"  {morning}")
    
    print("\nStatus briefing:")
    status = briefing_system.generate_status_briefing()
    print(f"  {status}")
    
    print("\nFull briefing:")
    full = briefing_system.generate_on_demand_briefing("full")
    print(f"  {full}")


def demo_routine_system():
    """Demonstrate routine automation"""
    print("\n" + "="*60)
    print("DEMO: Routine System")
    print("="*60)
    
    # Create action registry and routine manager
    action_registry = ActionRegistry()
    routine_manager = create_routine_manager(action_registry)
    
    # List available routines
    routines = routine_manager.list_routines()
    print(f"Available routines: {', '.join(routines)}")
    
    # Execute some routines
    test_routines = [
        ("start_meeting", {"app": "zoom"}),
        ("focus_mode", {"apps": ["code", "notepad"]}),
        ("morning_setup", {"apps": ["outlook", "calendar"]})
    ]
    
    for routine_name, args in test_routines:
        print(f"\nExecuting routine: {routine_name}")
        result = routine_manager.execute_routine(routine_name, args)
        print(f"  Result: {result}")
        
        # Show description
        description = routine_manager.get_routine_description(routine_name)
        print(f"  Description: {description}")


def demo_structured_commands():
    """Demonstrate structured LLM commands"""
    print("\n" + "="*60)
    print("DEMO: Structured LLM Commands")
    print("="*60)
    
    try:
        llm_service = LLMService()
        if not llm_service.enabled:
            print("LLM service not available - skipping structured command demo")
            return
        
        # Test structured command generation
        test_commands = [
            "set volume to 75",
            "open notepad",
            "start meeting routine",
            "what's the weather like",
            "take a screenshot"
        ]
        
        for command in test_commands:
            print(f"\nCommand: '{command}'")
            try:
                result = llm_service.generate_structured(command)
                print(f"  Type: {result.get('type', 'unknown')}")
                if result.get('action'):
                    print(f"  Action: {result['action']}")
                    print(f"  Args: {result.get('args', {})}")
                if result.get('reply'):
                    print(f"  Reply: {result['reply']}")
            except Exception as e:
                print(f"  Error: {e}")
    
    except Exception as e:
        print(f"LLM service initialization failed: {e}")


def demo_integration():
    """Demonstrate full integration"""
    print("\n" + "="*60)
    print("DEMO: Full Integration Test")
    print("="*60)
    
    print("This would normally start the full voice assistant with all features enabled.")
    print("For safety, we're just showing the configuration that would be used:")
    
    # Show the configuration
    try:
        with open("voice_config.json", "r") as f:
            config = json.load(f)
        
        print("\nCurrent Jarvis Configuration:")
        print(f"  Persona: {config.get('persona_name', 'Assistant')}")
        print(f"  Wake word: {config.get('wake_word', 'none')}")
        print(f"  ASR backend: {config.get('asr_backend', 'speech_recognition')}")
        print(f"  TTS backend: {config.get('tts_backend', 'sapi')}")
        print(f"  Features enabled: {list(config.get('features', {}).keys())}")
        
    except FileNotFoundError:
        print("voice_config.json not found - using defaults")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Jarvis Demo - Showcase AI Assistant Features")
    parser.add_argument("--config", default="voice_config.json", help="Configuration file")
    parser.add_argument("--quick", action="store_true", help="Run quick demo only")
    args = parser.parse_args()
    
    setup_logging()
    
    print("🤖 JARVIS DEMONSTRATION")
    print("Advanced AI Voice Assistant Features")
    print("=" * 60)
    
    if args.quick:
        print("Running quick demo...")
        demo_performance_metrics()
        demo_structured_commands()
    else:
        print("Running full feature demonstration...")
        
        # Run all demos
        demo_memory_system()
        demo_context_oracle()
        demo_performance_metrics()
        demo_briefing_system()
        demo_routine_system()
        demo_structured_commands()
        demo_integration()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("All Jarvis features have been demonstrated.")
    print("Check the generated files:")
    print("  - jarvis_demo.log (demo log)")
    print("  - demo_memory.json (memory persistence)")
    print("  - demo_metrics.json (performance data)")
    print("  - briefing_log.jsonl (briefing history)")
    print("\nTo start the full voice assistant:")
    print("  python main.py --config voice_config.json")


if __name__ == "__main__":
    main()
