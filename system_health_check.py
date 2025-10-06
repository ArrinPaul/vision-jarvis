#!/usr/bin/env python3
"""
System Health Check and Dependency Validator

This script validates all dependencies, checks for common issues,
and provides recommendations for optimal Jarvis performance.
"""

import sys
import importlib
import subprocess
import os
from typing import List, Tuple, Dict


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro} ✓"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} ✗ (Requires Python 3.8+)"


def check_dependency(package: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a dependency is available"""
    try:
        module_name = import_name or package
        importlib.import_module(module_name)
        return True, f"{package} ✓"
    except ImportError:
        return False, f"{package} ✗ (Not installed)"


def check_core_dependencies() -> List[Tuple[bool, str]]:
    """Check core dependencies required for basic functionality"""
    core_deps = [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("speechrecognition", "speech_recognition"),
        ("pyautogui", "pyautogui"),
        ("pygame", "pygame"),
        ("requests", "requests"),
    ]
    
    results = []
    for package, import_name in core_deps:
        results.append(check_dependency(package, import_name))
    
    return results


def check_jarvis_dependencies() -> List[Tuple[bool, str]]:
    """Check Jarvis-specific advanced dependencies"""
    jarvis_deps = [
        ("openwakeword", "openwakeword"),
        ("faster-whisper", "faster_whisper"),
        ("edge-tts", "edge_tts"),
        ("onnxruntime", "onnxruntime"),
        ("psutil", "psutil"),
        ("soundfile", "soundfile"),
        ("google-generativeai", "google.generativeai"),
    ]
    
    results = []
    for package, import_name in jarvis_deps:
        results.append(check_dependency(package, import_name))
    
    return results


def check_windows_dependencies() -> List[Tuple[bool, str]]:
    """Check Windows-specific dependencies"""
    if sys.platform != "win32":
        return [(True, "Non-Windows system - Windows deps not required")]
    
    win_deps = [
        ("pyttsx3", "pyttsx3"),
        ("comtypes", "comtypes"),
        ("pywin32", "win32gui"),
        ("pycaw", "pycaw.pycaw"),
    ]
    
    results = []
    for package, import_name in win_deps:
        results.append(check_dependency(package, import_name))
    
    return results


def check_config_files() -> List[Tuple[bool, str]]:
    """Check if configuration files exist and are valid"""
    results = []
    
    # Check voice_config.json
    try:
        import json
        with open("voice_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        results.append((True, "voice_config.json ✓"))
    except FileNotFoundError:
        results.append((False, "voice_config.json ✗ (Missing)"))
    except json.JSONDecodeError:
        results.append((False, "voice_config.json ✗ (Invalid JSON)"))
    
    # Check for required icons
    icons = ["mic_icon.png", "camera_icon.png", "paint_icon.png"]
    for icon in icons:
        if os.path.exists(icon):
            results.append((True, f"{icon} ✓"))
        else:
            results.append((False, f"{icon} ✗ (Missing)"))
    
    return results


def check_permissions() -> List[Tuple[bool, str]]:
    """Check system permissions"""
    results = []
    
    # Check camera access
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            results.append((True, "Camera access ✓"))
            cap.release()
        else:
            results.append((False, "Camera access ✗"))
    except Exception:
        results.append((False, "Camera access ✗ (Error)"))
    
    # Check microphone access (basic check)
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.1)
        results.append((True, "Microphone access ✓"))
    except Exception:
        results.append((False, "Microphone access ✗"))
    
    return results


def get_performance_recommendations() -> List[str]:
    """Get performance optimization recommendations"""
    recommendations = []
    
    # Check if running on battery
    try:
        import psutil
        battery = psutil.sensors_battery()
        if battery and not battery.power_plugged:
            recommendations.append("⚡ Running on battery - consider plugging in for better performance")
    except:
        pass
    
    # Check CPU usage
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            recommendations.append(f"🔥 High CPU usage ({cpu_percent}%) - close unnecessary applications")
    except:
        pass
    
    # Check memory usage
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            recommendations.append(f"💾 High memory usage ({memory.percent}%) - consider closing applications")
    except:
        pass
    
    return recommendations


def install_missing_dependencies(missing_deps: List[str]):
    """Install missing dependencies"""
    if not missing_deps:
        print("✓ All dependencies are installed!")
        return
    
    print(f"\n📦 Installing {len(missing_deps)} missing dependencies...")
    
    for dep in missing_deps:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✓ {dep} installed successfully")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {dep}")


def main():
    """Main health check function"""
    print("🤖 JARVIS SYSTEM HEALTH CHECK")
    print("=" * 50)
    
    # Python version check
    py_ok, py_msg = check_python_version()
    print(f"Python Version: {py_msg}")
    if not py_ok:
        print("❌ Incompatible Python version. Please upgrade to Python 3.8+")
        return
    
    print("\n📋 DEPENDENCY CHECK")
    print("-" * 30)
    
    # Core dependencies
    print("Core Dependencies:")
    core_results = check_core_dependencies()
    missing_core = []
    for ok, msg in core_results:
        print(f"  {msg}")
        if not ok:
            missing_core.append(msg.split()[0])
    
    # Jarvis dependencies
    print("\nJarvis Advanced Features:")
    jarvis_results = check_jarvis_dependencies()
    missing_jarvis = []
    for ok, msg in jarvis_results:
        print(f"  {msg}")
        if not ok:
            missing_jarvis.append(msg.split()[0])
    
    # Windows dependencies
    print("\nWindows-Specific:")
    win_results = check_windows_dependencies()
    missing_win = []
    for ok, msg in win_results:
        print(f"  {msg}")
        if not ok and sys.platform == "win32":
            missing_win.append(msg.split()[0])
    
    print("\n🔧 CONFIGURATION CHECK")
    print("-" * 30)
    config_results = check_config_files()
    for ok, msg in config_results:
        print(f"  {msg}")
    
    print("\n🔐 PERMISSIONS CHECK")
    print("-" * 30)
    perm_results = check_permissions()
    for ok, msg in perm_results:
        print(f"  {msg}")
    
    print("\n⚡ PERFORMANCE RECOMMENDATIONS")
    print("-" * 30)
    recommendations = get_performance_recommendations()
    if recommendations:
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("  ✓ System performance looks good!")
    
    # Summary
    all_missing = missing_core + missing_jarvis + missing_win
    if all_missing:
        print(f"\n❌ {len(all_missing)} dependencies missing")
        print("Run: pip install " + " ".join(all_missing))
    else:
        print("\n✅ ALL DEPENDENCIES SATISFIED!")
        print("Your Jarvis system is ready to run!")


if __name__ == "__main__":
    main()
