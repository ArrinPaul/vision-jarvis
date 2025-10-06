"""
Advanced Security Features for JARVIS

Implements comprehensive security measures including access control,
encrypted communications, and security monitoring.
"""

import os
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Optional dependencies with fallbacks
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

class SecurityManager:
    """Comprehensive security management system."""
    
    def __init__(self, config_path: str = "security_config.json"):
        self.config_path = config_path
        self.security_logs = []
        self.active_sessions = {}
        self.failed_attempts = defaultdict(int)
        self.blocked_ips = set()
        self.access_tokens = {}
        
        # Load or create security configuration
        self.config = self._load_security_config()
        
        # Initialize encryption
        self.encryption_key = self._get_or_create_encryption_key()
        
        # Setup logging
        self._setup_security_logging()
        
    def _load_security_config(self) -> Dict[str, Any]:
        """Load security configuration."""
        default_config = {
            "max_failed_attempts": 5,
            "lockout_duration_minutes": 30,
            "session_timeout_minutes": 60,
            "password_min_length": 8,
            "require_biometric": False,
            "enable_two_factor": False,
            "allowed_ip_ranges": [],
            "security_level": "medium"
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for any missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                self._log_security_event("config_error", f"Failed to load config: {e}")
                return default_config
        else:
            # Save default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _get_or_create_encryption_key(self) -> Optional[bytes]:
        """Get or create encryption key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return None
            
        key_file = "jarvis_encryption.key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def _setup_security_logging(self):
        """Setup security event logging."""
        logging.basicConfig(
            filename='security.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('SecurityManager')
    
    def _log_security_event(self, event_type: str, details: str, severity: str = "INFO"):
        """Log security event."""
        timestamp = datetime.now().isoformat()
        event = {
            "timestamp": timestamp,
            "type": event_type,
            "details": details,
            "severity": severity
        }
        
        self.security_logs.append(event)
        
        # Log to file
        log_message = f"{event_type}: {details}"
        if severity == "CRITICAL":
            self.logger.critical(log_message)
        elif severity == "WARNING":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def authenticate_user(self, username: str, password: str = None, 
                         biometric_data: bytes = None, ip_address: str = None) -> Dict[str, Any]:
        """Authenticate user with multiple factors."""
        
        # Check if IP is blocked
        if ip_address and ip_address in self.blocked_ips:
            self._log_security_event("blocked_access", f"Blocked IP attempted access: {ip_address}", "WARNING")
            return {"success": False, "error": "Access denied"}
        
        # Check failed attempts
        if self.failed_attempts[username] >= self.config["max_failed_attempts"]:
            self._log_security_event("max_attempts", f"Max attempts reached for user: {username}", "WARNING")
            return {"success": False, "error": "Account temporarily locked"}
        
        # Validate credentials
        if not self._validate_credentials(username, password):
            self.failed_attempts[username] += 1
            self._log_security_event("failed_auth", f"Failed authentication for user: {username}", "WARNING")
            
            # Block IP if too many failures
            if ip_address and self.failed_attempts[username] >= self.config["max_failed_attempts"]:
                self.blocked_ips.add(ip_address)
                
            return {"success": False, "error": "Invalid credentials"}
        
        # Biometric verification if required
        if self.config["require_biometric"] and not self._verify_biometric(username, biometric_data):
            self._log_security_event("biometric_fail", f"Biometric verification failed for user: {username}", "WARNING")
            return {"success": False, "error": "Biometric verification failed"}
        
        # Create session
        session_id = self._create_session(username, ip_address)
        
        # Reset failed attempts on successful login
        if username in self.failed_attempts:
            del self.failed_attempts[username]
        
        self._log_security_event("successful_auth", f"User authenticated: {username}")
        
        return {
            "success": True,
            "session_id": session_id,
            "expires_at": (datetime.now() + timedelta(minutes=self.config["session_timeout_minutes"])).isoformat()
        }
    
    def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials."""
        # In a real implementation, this would check against a secure database
        # For demo purposes, using a simple hash comparison
        
        users_file = "users.json"
        if not os.path.exists(users_file):
            return False
        
        try:
            with open(users_file, 'r') as f:
                users = json.load(f)
            
            if username not in users:
                return False
            
            stored_hash = users[username].get("password_hash")
            if not stored_hash:
                return False
            
            # Hash the provided password and compare
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            return password_hash == stored_hash
            
        except Exception as e:
            self._log_security_event("credential_error", f"Error validating credentials: {e}", "ERROR")
            return False
    
    def _verify_biometric(self, username: str, biometric_data: bytes) -> bool:
        """Verify biometric data."""
        if not biometric_data:
            return False
        
        # In a real implementation, this would use actual biometric comparison
        # For demo purposes, return True if biometric data is provided
        return True
    
    def _create_session(self, username: str, ip_address: str = None) -> str:
        """Create authenticated session."""
        session_id = hashlib.sha256(f"{username}{time.time()}{ip_address}".encode()).hexdigest()
        
        session_data = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(minutes=self.config["session_timeout_minutes"])).isoformat(),
            "ip_address": ip_address,
            "last_activity": datetime.now().isoformat()
        }
        
        self.active_sessions[session_id] = session_data
        return session_id
    
    def validate_session(self, session_id: str) -> Dict[str, Any]:
        """Validate active session."""
        if session_id not in self.active_sessions:
            return {"valid": False, "error": "Invalid session"}
        
        session = self.active_sessions[session_id]
        expires_at = datetime.fromisoformat(session["expires_at"])
        
        if datetime.now() > expires_at:
            del self.active_sessions[session_id]
            return {"valid": False, "error": "Session expired"}
        
        # Update last activity
        session["last_activity"] = datetime.now().isoformat()
        
        return {"valid": True, "username": session["username"]}
    
    def encrypt_data(self, data: str) -> Optional[str]:
        """Encrypt sensitive data."""
        if not CRYPTOGRAPHY_AVAILABLE or not self.encryption_key:
            # Fallback: simple base64 encoding (NOT secure, for demo only)
            return base64.b64encode(data.encode()).decode()
        
        try:
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            self._log_security_event("encryption_error", f"Failed to encrypt data: {e}", "ERROR")
            return None
    
    def decrypt_data(self, encrypted_data: str) -> Optional[str]:
        """Decrypt sensitive data."""
        if not CRYPTOGRAPHY_AVAILABLE or not self.encryption_key:
            # Fallback: simple base64 decoding (NOT secure, for demo only)
            try:
                return base64.b64decode(encrypted_data.encode()).decode()
            except:
                return None
        
        try:
            fernet = Fernet(self.encryption_key)
            decrypted_data = fernet.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            self._log_security_event("decryption_error", f"Failed to decrypt data: {e}", "ERROR")
            return None
    
    def generate_access_token(self, username: str, permissions: List[str] = None) -> Optional[str]:
        """Generate JWT access token."""
        if not JWT_AVAILABLE:
            # Fallback: simple token generation
            token_data = f"{username}:{time.time()}:{':'.join(permissions or [])}"
            return base64.b64encode(token_data.encode()).decode()
        
        try:
            payload = {
                "username": username,
                "permissions": permissions or [],
                "issued_at": time.time(),
                "expires_at": time.time() + 3600  # 1 hour
            }
            
            token = jwt.encode(payload, "secret_key", algorithm="HS256")
            self.access_tokens[token] = payload
            return token
            
        except Exception as e:
            self._log_security_event("token_error", f"Failed to generate token: {e}", "ERROR")
            return None
    
    def validate_access_token(self, token: str) -> Dict[str, Any]:
        """Validate access token."""
        if not JWT_AVAILABLE:
            # Fallback validation
            try:
                decoded = base64.b64decode(token.encode()).decode()
                parts = decoded.split(':')
                if len(parts) >= 2:
                    return {"valid": True, "username": parts[0]}
            except:
                pass
            return {"valid": False}
        
        try:
            payload = jwt.decode(token, "secret_key", algorithms=["HS256"])
            
            if time.time() > payload.get("expires_at", 0):
                return {"valid": False, "error": "Token expired"}
            
            return {"valid": True, "username": payload["username"], "permissions": payload.get("permissions", [])}
            
        except Exception as e:
            self._log_security_event("token_validation_error", f"Token validation failed: {e}", "WARNING")
            return {"valid": False, "error": "Invalid token"}
    
    def monitor_system_access(self) -> Dict[str, Any]:
        """Monitor system access patterns."""
        current_time = datetime.now()
        recent_events = [
            event for event in self.security_logs
            if datetime.fromisoformat(event["timestamp"]) > current_time - timedelta(hours=24)
        ]
        
        # Count event types
        event_counts = defaultdict(int)
        for event in recent_events:
            event_counts[event["type"]] += 1
        
        # Detect suspicious patterns
        suspicious_activities = []
        
        if event_counts["failed_auth"] > 10:
            suspicious_activities.append("High number of failed authentication attempts")
        
        if len(self.blocked_ips) > 5:
            suspicious_activities.append("Multiple IP addresses blocked")
        
        if event_counts.get("max_attempts", 0) > 3:
            suspicious_activities.append("Multiple accounts locked due to failed attempts")
        
        return {
            "active_sessions": len(self.active_sessions),
            "blocked_ips": len(self.blocked_ips),
            "failed_attempts_24h": event_counts["failed_auth"],
            "suspicious_activities": suspicious_activities,
            "security_level": self._calculate_security_level(),
            "last_update": current_time.isoformat()
        }
    
    def _calculate_security_level(self) -> str:
        """Calculate current security threat level."""
        threat_score = 0
        
        # Factor in recent failed attempts
        recent_failures = sum(1 for event in self.security_logs[-100:] if event["type"] == "failed_auth")
        threat_score += min(recent_failures * 2, 20)
        
        # Factor in blocked IPs
        threat_score += len(self.blocked_ips) * 5
        
        # Factor in active sessions
        if len(self.active_sessions) > 10:
            threat_score += 10
        
        if threat_score < 10:
            return "LOW"
        elif threat_score < 30:
            return "MEDIUM"
        elif threat_score < 50:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def create_user(self, username: str, password: str, permissions: List[str] = None) -> bool:
        """Create new user account."""
        if len(password) < self.config["password_min_length"]:
            return False
        
        users_file = "users.json"
        users = {}
        
        if os.path.exists(users_file):
            with open(users_file, 'r') as f:
                users = json.load(f)
        
        if username in users:
            return False  # User already exists
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        users[username] = {
            "password_hash": password_hash,
            "permissions": permissions or ["basic_access"],
            "created_at": datetime.now().isoformat(),
            "last_login": None
        }
        
        with open(users_file, 'w') as f:
            json.dump(users, f, indent=2)
        
        self._log_security_event("user_created", f"New user created: {username}")
        return True
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        return {
            "system_status": self.monitor_system_access(),
            "recent_events": self.security_logs[-10:],  # Last 10 events
            "configuration": {
                "security_level": self.config["security_level"],
                "biometric_required": self.config["require_biometric"],
                "two_factor_enabled": self.config["enable_two_factor"],
                "session_timeout": self.config["session_timeout_minutes"]
            },
            "statistics": {
                "total_security_events": len(self.security_logs),
                "active_sessions": len(self.active_sessions),
                "blocked_ips": len(self.blocked_ips),
                "failed_attempts": len(self.failed_attempts)
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize security manager
    security = SecurityManager()
    
    # Create a test user
    security.create_user("admin", "secure_password123", ["admin", "system_control"])
    
    # Test authentication
    auth_result = security.authenticate_user("admin", "secure_password123", ip_address="192.168.1.100")
    print("Authentication result:", auth_result)
    
    if auth_result["success"]:
        session_id = auth_result["session_id"]
        
        # Test session validation
        session_check = security.validate_session(session_id)
        print("Session validation:", session_check)
        
        # Test encryption
        sensitive_data = "This is confidential information"
        encrypted = security.encrypt_data(sensitive_data)
        decrypted = security.decrypt_data(encrypted) if encrypted else None
        print(f"Encryption test: {decrypted == sensitive_data}")
        
        # Generate access token
        token = security.generate_access_token("admin", ["read", "write", "admin"])
        if token:
            token_validation = security.validate_access_token(token)
            print("Token validation:", token_validation)
    
    # Generate security report
    report = security.get_security_report()
    print("Security Report:")
    print(json.dumps(report, indent=2))