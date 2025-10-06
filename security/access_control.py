"""
Access Control System for JARVIS

Implements role-based access control, permission management,
and resource protection mechanisms.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any
from enum import Enum
from collections import defaultdict

class PermissionLevel(Enum):
    """Permission levels for system access."""
    NONE = 0
    READ = 1
    WRITE = 2
    EXECUTE = 3
    ADMIN = 4
    SYSTEM = 5

class ResourceType(Enum):
    """Types of system resources."""
    CAMERA = "camera"
    MICROPHONE = "microphone"
    VOICE_ASSISTANT = "voice_assistant"
    GESTURE_CONTROL = "gesture_control"
    AR_INTERFACE = "ar_interface"
    SYSTEM_SETTINGS = "system_settings"
    USER_DATA = "user_data"
    SECURITY_LOGS = "security_logs"
    NETWORK_ACCESS = "network_access"
    FILE_SYSTEM = "file_system"

class AccessControlManager:
    """Comprehensive access control system."""
    
    def __init__(self, config_file: str = "access_control.json"):
        self.config_file = config_file
        self.roles = {}
        self.users = {}
        self.permissions = {}
        self.resource_locks = {}
        self.access_history = []
        self.active_sessions = {}
        
        # Load configuration
        self._load_configuration()
        
        # Initialize default roles and permissions
        self._initialize_default_roles()
    
    def _load_configuration(self):
        """Load access control configuration."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.roles = config.get("roles", {})
                    self.users = config.get("users", {})
                    self.permissions = config.get("permissions", {})
            except Exception as e:
                print(f"Failed to load access control configuration: {e}")
                self._initialize_default_roles()
        else:
            self._initialize_default_roles()
    
    def _save_configuration(self):
        """Save access control configuration."""
        config = {
            "roles": self.roles,
            "users": self.users,
            "permissions": self.permissions,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Failed to save access control configuration: {e}")
    
    def _initialize_default_roles(self):
        """Initialize default roles and permissions."""
        # Define default roles
        self.roles = {
            "guest": {
                "description": "Limited access guest user",
                "permissions": [
                    "camera:read",
                    "ar_interface:read"
                ],
                "resource_limits": {
                    "session_duration_minutes": 30,
                    "max_concurrent_sessions": 1
                }
            },
            "user": {
                "description": "Standard user with basic access",
                "permissions": [
                    "camera:read",
                    "camera:write",
                    "microphone:read",
                    "voice_assistant:read",
                    "voice_assistant:execute",
                    "gesture_control:read",
                    "gesture_control:execute",
                    "ar_interface:read",
                    "ar_interface:write",
                    "user_data:read",
                    "user_data:write"
                ],
                "resource_limits": {
                    "session_duration_minutes": 240,
                    "max_concurrent_sessions": 2
                }
            },
            "power_user": {
                "description": "Advanced user with extended privileges",
                "permissions": [
                    "camera:read",
                    "camera:write",
                    "camera:execute",
                    "microphone:read",
                    "microphone:write",
                    "voice_assistant:read",
                    "voice_assistant:write",
                    "voice_assistant:execute",
                    "gesture_control:read",
                    "gesture_control:write",
                    "gesture_control:execute",
                    "ar_interface:read",
                    "ar_interface:write",
                    "ar_interface:execute",
                    "user_data:read",
                    "user_data:write",
                    "system_settings:read",
                    "network_access:read"
                ],
                "resource_limits": {
                    "session_duration_minutes": 480,
                    "max_concurrent_sessions": 3
                }
            },
            "admin": {
                "description": "Administrator with full system access",
                "permissions": [
                    "*:*"  # Wildcard for all permissions
                ],
                "resource_limits": {
                    "session_duration_minutes": -1,  # Unlimited
                    "max_concurrent_sessions": -1    # Unlimited
                }
            }
        }
        
        # Define default users
        self.users = {
            "system": {
                "role": "admin",
                "created_at": datetime.now().isoformat(),
                "active": True,
                "description": "System administrator account"
            },
            "guest": {
                "role": "guest",
                "created_at": datetime.now().isoformat(),
                "active": True,
                "description": "Default guest account"
            }
        }
        
        # Save configuration
        self._save_configuration()
    
    def create_role(self, role_name: str, description: str, permissions: List[str], 
                   resource_limits: Dict[str, Any] = None) -> bool:
        """Create a new role."""
        if role_name in self.roles:
            return False
        
        self.roles[role_name] = {
            "description": description,
            "permissions": permissions,
            "resource_limits": resource_limits or {
                "session_duration_minutes": 120,
                "max_concurrent_sessions": 2
            },
            "created_at": datetime.now().isoformat()
        }
        
        self._save_configuration()
        self._log_access_event("role_created", f"Role created: {role_name}")
        return True
    
    def create_user(self, username: str, role: str, description: str = "") -> bool:
        """Create a new user."""
        if username in self.users or role not in self.roles:
            return False
        
        self.users[username] = {
            "role": role,
            "created_at": datetime.now().isoformat(),
            "active": True,
            "description": description,
            "last_login": None,
            "failed_attempts": 0
        }
        
        self._save_configuration()
        self._log_access_event("user_created", f"User created: {username} with role: {role}")
        return True
    
    def check_permission(self, username: str, resource: str, action: str) -> Dict[str, Any]:
        """Check if user has permission for specific resource and action."""
        # Get user info
        user_info = self.users.get(username)
        if not user_info or not user_info.get("active", False):
            return {
                "allowed": False,
                "reason": "User not found or inactive"
            }
        
        # Get user role
        role_name = user_info.get("role")
        role_info = self.roles.get(role_name)
        if not role_info:
            return {
                "allowed": False,
                "reason": "Invalid user role"
            }
        
        # Check permissions
        permissions = role_info.get("permissions", [])
        
        # Check for wildcard permission
        if "*:*" in permissions:
            return {
                "allowed": True,
                "reason": "Admin privileges"
            }
        
        # Check specific permission
        required_permission = f"{resource}:{action}"
        resource_wildcard = f"{resource}:*"
        action_wildcard = f"*:{action}"
        
        if (required_permission in permissions or 
            resource_wildcard in permissions or 
            action_wildcard in permissions):
            
            # Check resource limits
            limit_check = self._check_resource_limits(username, resource)
            if not limit_check["allowed"]:
                return limit_check
            
            return {
                "allowed": True,
                "reason": "Permission granted"
            }
        
        return {
            "allowed": False,
            "reason": "Insufficient permissions"
        }
    
    def _check_resource_limits(self, username: str, resource: str) -> Dict[str, Any]:
        """Check resource-specific limits."""
        user_info = self.users.get(username)
        role_info = self.roles.get(user_info.get("role"))
        
        if not role_info:
            return {"allowed": False, "reason": "Invalid role"}
        
        resource_limits = role_info.get("resource_limits", {})
        
        # Check session limits
        max_sessions = resource_limits.get("max_concurrent_sessions", 2)
        if max_sessions > 0:
            active_sessions = sum(1 for session in self.active_sessions.values() 
                                if session["username"] == username)
            if active_sessions >= max_sessions:
                return {
                    "allowed": False,
                    "reason": f"Maximum concurrent sessions ({max_sessions}) exceeded"
                }
        
        # Check resource locks
        if resource in self.resource_locks:
            lock_info = self.resource_locks[resource]
            if lock_info["locked_by"] != username:
                return {
                    "allowed": False,
                    "reason": f"Resource locked by {lock_info['locked_by']}"
                }
        
        return {"allowed": True, "reason": "Resource limits OK"}
    
    def request_resource_access(self, username: str, resource: str, action: str, 
                              session_id: str = None) -> Dict[str, Any]:
        """Request access to a specific resource."""
        # Check permission
        permission_check = self.check_permission(username, resource, action)
        
        if not permission_check["allowed"]:
            self._log_access_event("access_denied", 
                                 f"Access denied for {username} to {resource}:{action} - {permission_check['reason']}")
            return permission_check
        
        # Grant access
        access_id = f"{username}_{resource}_{action}_{int(time.time())}"
        
        access_record = {
            "access_id": access_id,
            "username": username,
            "resource": resource,
            "action": action,
            "granted_at": datetime.now().isoformat(),
            "session_id": session_id,
            "status": "active"
        }
        
        # Store access record
        self.access_history.append(access_record)
        
        # Log access
        self._log_access_event("access_granted", 
                             f"Access granted to {username} for {resource}:{action}")
        
        return {
            "allowed": True,
            "access_id": access_id,
            "expires_at": self._calculate_access_expiry(username),
            "reason": "Access granted"
        }
    
    def _calculate_access_expiry(self, username: str) -> str:
        """Calculate when access expires."""
        user_info = self.users.get(username)
        role_info = self.roles.get(user_info.get("role"))
        
        session_duration = role_info.get("resource_limits", {}).get("session_duration_minutes", 120)
        
        if session_duration <= 0:
            return "never"
        
        expiry_time = datetime.now() + timedelta(minutes=session_duration)
        return expiry_time.isoformat()
    
    def revoke_access(self, access_id: str, reason: str = "Manual revocation") -> bool:
        """Revoke specific access."""
        for access_record in self.access_history:
            if access_record["access_id"] == access_id and access_record["status"] == "active":
                access_record["status"] = "revoked"
                access_record["revoked_at"] = datetime.now().isoformat()
                access_record["revocation_reason"] = reason
                
                self._log_access_event("access_revoked", 
                                     f"Access revoked: {access_id} - {reason}")
                return True
        
        return False
    
    def lock_resource(self, resource: str, username: str, duration_minutes: int = 60) -> bool:
        """Lock a resource for exclusive access."""
        if resource in self.resource_locks:
            return False  # Already locked
        
        lock_info = {
            "locked_by": username,
            "locked_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(minutes=duration_minutes)).isoformat(),
            "duration_minutes": duration_minutes
        }
        
        self.resource_locks[resource] = lock_info
        
        self._log_access_event("resource_locked", 
                             f"Resource {resource} locked by {username} for {duration_minutes} minutes")
        
        return True
    
    def unlock_resource(self, resource: str, username: str = None) -> bool:
        """Unlock a resource."""
        if resource not in self.resource_locks:
            return False
        
        lock_info = self.resource_locks[resource]
        
        # Check if user can unlock (either the locker or admin)
        if username and lock_info["locked_by"] != username:
            user_info = self.users.get(username)
            if not user_info or user_info.get("role") != "admin":
                return False
        
        del self.resource_locks[resource]
        
        self._log_access_event("resource_unlocked", 
                             f"Resource {resource} unlocked by {username or 'system'}")
        
        return True
    
    def create_session(self, username: str, client_info: Dict[str, Any] = None) -> Optional[str]:
        """Create user session."""
        user_info = self.users.get(username)
        if not user_info or not user_info.get("active", False):
            return None
        
        # Check session limits
        role_info = self.roles.get(user_info.get("role"))
        max_sessions = role_info.get("resource_limits", {}).get("max_concurrent_sessions", 2)
        
        if max_sessions > 0:
            active_user_sessions = sum(1 for session in self.active_sessions.values() 
                                     if session["username"] == username)
            if active_user_sessions >= max_sessions:
                return None
        
        # Create session
        session_id = f"session_{username}_{int(time.time())}"
        
        session_info = {
            "session_id": session_id,
            "username": username,
            "role": user_info.get("role"),
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "client_info": client_info or {},
            "status": "active"
        }
        
        self.active_sessions[session_id] = session_info
        
        # Update user last login
        self.users[username]["last_login"] = datetime.now().isoformat()
        self._save_configuration()
        
        self._log_access_event("session_created", f"Session created for {username}: {session_id}")
        
        return session_id
    
    def validate_session(self, session_id: str) -> Dict[str, Any]:
        """Validate active session."""
        session_info = self.active_sessions.get(session_id)
        
        if not session_info:
            return {"valid": False, "reason": "Session not found"}
        
        # Check if session is still active
        if session_info.get("status") != "active":
            return {"valid": False, "reason": "Session inactive"}
        
        # Check session timeout
        username = session_info["username"]
        user_info = self.users.get(username)
        role_info = self.roles.get(user_info.get("role"))
        
        session_duration = role_info.get("resource_limits", {}).get("session_duration_minutes", 120)
        
        if session_duration > 0:
            created_at = datetime.fromisoformat(session_info["created_at"])
            if datetime.now() > created_at + timedelta(minutes=session_duration):
                # Session expired
                session_info["status"] = "expired"
                return {"valid": False, "reason": "Session expired"}
        
        # Update last activity
        session_info["last_activity"] = datetime.now().isoformat()
        
        return {
            "valid": True,
            "username": username,
            "role": session_info["role"],
            "session_info": session_info
        }
    
    def end_session(self, session_id: str) -> bool:
        """End user session."""
        if session_id not in self.active_sessions:
            return False
        
        session_info = self.active_sessions[session_id]
        session_info["status"] = "ended"
        session_info["ended_at"] = datetime.now().isoformat()
        
        self._log_access_event("session_ended", 
                             f"Session ended: {session_id} for user {session_info['username']}")
        
        return True
    
    def _log_access_event(self, event_type: str, details: str):
        """Log access control event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details
        }
        
        # Keep only recent events (last 1000)
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-1000:]
    
    def get_user_permissions(self, username: str) -> Dict[str, Any]:
        """Get all permissions for a user."""
        user_info = self.users.get(username)
        if not user_info:
            return {"error": "User not found"}
        
        role_name = user_info.get("role")
        role_info = self.roles.get(role_name)
        
        if not role_info:
            return {"error": "Invalid role"}
        
        return {
            "username": username,
            "role": role_name,
            "permissions": role_info.get("permissions", []),
            "resource_limits": role_info.get("resource_limits", {}),
            "active": user_info.get("active", False)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get access control system status."""
        return {
            "total_users": len(self.users),
            "total_roles": len(self.roles),
            "active_sessions": len([s for s in self.active_sessions.values() if s["status"] == "active"]),
            "locked_resources": len(self.resource_locks),
            "recent_access_events": len(self.access_history),
            "resource_locks": list(self.resource_locks.keys()),
            "available_resources": [rt.value for rt in ResourceType],
            "permission_levels": [pl.name for pl in PermissionLevel]
        }
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions and resource locks."""
        current_time = datetime.now()
        
        # Clean up expired sessions
        expired_sessions = []
        for session_id, session_info in self.active_sessions.items():
            if session_info.get("status") == "active":
                username = session_info["username"]
                user_info = self.users.get(username)
                role_info = self.roles.get(user_info.get("role"))
                
                session_duration = role_info.get("resource_limits", {}).get("session_duration_minutes", 120)
                
                if session_duration > 0:
                    created_at = datetime.fromisoformat(session_info["created_at"])
                    if current_time > created_at + timedelta(minutes=session_duration):
                        expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.end_session(session_id)
        
        # Clean up expired resource locks
        expired_locks = []
        for resource, lock_info in self.resource_locks.items():
            expires_at = datetime.fromisoformat(lock_info["expires_at"])
            if current_time > expires_at:
                expired_locks.append(resource)
        
        for resource in expired_locks:
            self.unlock_resource(resource)


# Example usage and testing
if __name__ == "__main__":
    # Initialize access control manager
    acm = AccessControlManager()
    
    # Create a test user
    acm.create_user("test_user", "user", "Test user account")
    
    # Create a session
    session_id = acm.create_session("test_user", {"client_type": "desktop", "ip": "192.168.1.100"})
    print(f"Created session: {session_id}")
    
    # Test permission checking
    permission_result = acm.check_permission("test_user", "camera", "read")
    print(f"Camera read permission: {permission_result}")
    
    # Request resource access
    access_result = acm.request_resource_access("test_user", "camera", "read", session_id)
    print(f"Camera access request: {access_result}")
    
    # Test resource locking
    lock_result = acm.lock_resource("camera", "test_user", 30)
    print(f"Camera lock result: {lock_result}")
    
    # Get user permissions
    user_perms = acm.get_user_permissions("test_user")
    print(f"User permissions: {json.dumps(user_perms, indent=2)}")
    
    # Get system status
    status = acm.get_system_status()
    print(f"System status: {json.dumps(status, indent=2)}")
    
    # Validate session
    session_validation = acm.validate_session(session_id)
    print(f"Session validation: {session_validation}")
    
    # Clean up
    acm.cleanup_expired_sessions()