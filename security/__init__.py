"""
Security Module Initialization

Makes the security package importable and provides easy access to all security components.
"""

from .security_manager import SecurityManager
from .secure_communications import SecureCommunicator
from .access_control import AccessControlManager, PermissionLevel, ResourceType
from .security_monitoring import SecurityMonitor, ThreatLevel, SecurityEvent
from .security_integration import JarvisSecuritySystem

__all__ = [
    'SecurityManager',
    'SecureCommunicator', 
    'AccessControlManager',
    'SecurityMonitor',
    'JarvisSecuritySystem',
    'PermissionLevel',
    'ResourceType',
    'ThreatLevel',
    'SecurityEvent'
]

__version__ = "1.0.0"
__author__ = "JARVIS Security Team"
__description__ = "Comprehensive security system for JARVIS"
