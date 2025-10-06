#!/usr/bin/env python3
"""
Jarvis Startup Script - Production Ready

This script provides a robust startup sequence for the Jarvis system
with comprehensive error handling, dependency checking, and graceful fallbacks.
"""

import sys
import os
import logging
import argparse
import time
from pathlib import Path


def setup_logging(log_level="INFO"):
    """Setup comprehensive logging"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler("logs/jarvis_startup.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress verbose third-party logging
    logging.getLogger("comtypes").setLevel(logging.WARNING)
    logging.getLogger("pygame").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def check_critical_dependencies():
    """Check if critical dependencies are available"""
    critical_deps = [
        ("cv2", "opencv-python"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
    ]
    
    missing = []
    for module, package in critical_deps:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    return missing


def create_default_config():
    """Create default configuration files if they don't exist"""
    logger = logging.getLogger(__name__)
    
    # Create default voice_config.json
    if not os.path.exists("voice_config.json"):
        logger.info("Creating default voice_config.json")
        default_config = {
            "speech_timeout": 5.0,
            "phrase_time_limit": 10.0,
            "hover_timeout": 1.5,
            "activation_radius": 100,
            "tts_language": "en",
            "max_audio_queue": 10,
            "tts_cache_size": 50,
            "log_level": "INFO",
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "continuous_listening": False,
            "listen_pause_duration": 2.0,
            "prefer_sapi_tts": True,
            "enable_audio_feedback": True,
            "persona_name": "Jarvis",
            "wake_word": "jarvis",
            "require_wake_word": False,
            "wake_word_backend": "openwakeword",
            "asr_backend": "speech_recognition",
            "tts_backend": "edge_tts",
            "features": {
                "vision_object_detection": False,
                "event_logging": True,
                "structured_commands": True,
                "wake_word_detection": False,
                "context_monitoring": True
            },
            "whisper_model_size": "base",
            "whisper_device": "cpu",
            "whisper_compute_type": "int8",
            "edge_voice": "en-US-GuyNeural"
        }
        
        import json
        with open("voice_config.json", "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        
        logger.info("Default configuration created")


def check_environment():
    """Check environment and create necessary directories"""
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    directories = ["logs", "temp", "exports", "tests"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Check for icon files and create placeholders if missing
    icons = ["assets/mic_icon.png", "assets/camera_icon.png", "assets/paint_icon.png"]
    for icon in icons:
        if not os.path.exists(icon):
            logger.warning(f"Icon {icon} not found - UI may show placeholders")
        else:
            logger.info(f"Icon {icon} found")
    
    logger.info("Environment check completed")


def safe_import_main():
    """Safely import and initialize the main application"""
    logger = logging.getLogger(__name__)
    
    try:
        # Import main application
        from main import MainApp
        logger.info("Main application imported successfully")
        return MainApp
    except ImportError as e:
        logger.error(f"Failed to import main application: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error importing main application: {e}")
        return None


def run_health_check():
    """Run system health check"""
    logger = logging.getLogger(__name__)
    
    try:
        # Import and run health check
        import system_health_check
        logger.info("Running system health check...")
        
        # Capture health check output
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            system_health_check.main()
        
        health_output = f.getvalue()
        logger.info("Health check completed")
        
        return "✅ ALL DEPENDENCIES SATISFIED!" in health_output
        
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        return False


def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description="Start Jarvis AI Assistant")
    parser.add_argument("--config", default="voice_config.json", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--skip-health-check", action="store_true", help="Skip system health check")
    parser.add_argument("--safe-mode", action="store_true", help="Start in safe mode (minimal features)")
    parser.add_argument("--demo", action="store_true", help="Run demo instead of full application")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    print("🤖 JARVIS AI ASSISTANT")
    print("=" * 50)
    logger.info("Starting Jarvis AI Assistant...")
    
    # Check critical dependencies
    missing_deps = check_critical_dependencies()
    if missing_deps:
        logger.error(f"Critical dependencies missing: {missing_deps}")
        print(f"❌ Missing critical dependencies: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        return 1
    
    # Create default configuration
    create_default_config()
    
    # Check environment
    check_environment()
    
    # Run health check unless skipped
    if not args.skip_health_check:
        logger.info("Performing system health check...")
        health_ok = run_health_check()
        if not health_ok:
            logger.warning("Health check indicates potential issues")
            print("⚠️  Some dependencies may be missing. Run with --skip-health-check to proceed anyway.")
            
            response = input("Continue anyway? (y/N): ").lower().strip()
            if response != 'y':
                logger.info("Startup cancelled by user")
                return 0
    
    # Demo mode
    if args.demo:
        logger.info("Starting in demo mode...")
        try:
            import subprocess
            result = subprocess.run([sys.executable, "jarvis_demo.py", "--quick"],
                                  capture_output=False, text=True)
            return result.returncode
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            return 1
    
    # Safe mode configuration
    if args.safe_mode:
        logger.info("Starting in safe mode...")
        # Modify config for safe mode
        import json
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # Disable advanced features in safe mode
            config["features"] = {
                "vision_object_detection": False,
                "event_logging": True,
                "structured_commands": False,
                "wake_word_detection": False,
                "context_monitoring": False
            }
            config["asr_backend"] = "speech_recognition"
            config["tts_backend"] = "sapi"
            
            # Save safe mode config
            with open("safe_mode_config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            
            args.config = "safe_mode_config.json"
            logger.info("Safe mode configuration created")
            
        except Exception as e:
            logger.warning(f"Failed to create safe mode config: {e}")
    
    # Import and start main application
    logger.info("Initializing main application...")
    MainApp = safe_import_main()
    
    if MainApp is None:
        logger.error("Failed to import main application")
        print("❌ Failed to start Jarvis. Check logs for details.")
        return 1
    
    try:
        logger.info("Starting Jarvis main application...")
        print("✅ Starting Jarvis AI Assistant...")
        print("Press 'q' to quit, 'h' for help")
        
        # Initialize and run the application
        app = MainApp()
        app.run()
        
        logger.info("Jarvis application ended normally")
        print("👋 Jarvis shutdown complete")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Jarvis interrupted by user")
        print("\n👋 Jarvis shutdown by user")
        return 0
        
    except Exception as e:
        logger.error(f"Jarvis application error: {e}", exc_info=True)
        print(f"❌ Jarvis encountered an error: {e}")
        print("Check logs/jarvis_startup.log for details")
        return 1
    
    finally:
        # Cleanup
        try:
            if 'app' in locals():
                app.cleanup()
        except:
            pass


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
