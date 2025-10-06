"""
Security Integration Module for JARVIS

Integrates all security components into a unified security system
providing centralized security management and coordination.
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Import security components
from .security_manager import SecurityManager
from .secure_communications import SecureCommunicator
from .access_control import AccessControlManager
from .security_monitoring import SecurityMonitor, ThreatLevel, SecurityEvent

class JarvisSecuritySystem:
    """Integrated security system for JARVIS."""
    
    def __init__(self, config_file: str = "jarvis_security.json"):
        self.config_file = config_file
        self.running = False
        
        # Initialize security components
        self.security_manager = SecurityManager()
        self.communicator = SecureCommunicator()
        self.access_control = AccessControlManager()
        self.monitor = SecurityMonitor()
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Security state
        self.security_state = {
            "system_locked": False,
            "emergency_mode": False,
            "last_security_check": None,
            "active_incidents": [],
            "security_level": "NORMAL"
        }
        
        # Integration setup
        self._setup_integration()
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load security system configuration."""
        default_config = {
            "auto_start_monitoring": True,
            "auto_start_communications": True,
            "integration_settings": {
                "sync_user_sessions": True,
                "cross_component_alerts": True,
                "centralized_logging": True,
                "auto_incident_response": True
            },
            "security_policies": {
                "max_failed_attempts": 3,
                "session_timeout_minutes": 60,
                "require_mfa": False,
                "auto_lock_on_threat": True,
                "emergency_contacts": []
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                print(f"Failed to load security config: {e}")
                return default_config
        else:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _setup_integration(self):
        """Setup integration between security components."""
        # Register security event handlers
        self.monitor.register_event_handler("unauthorized_access", self._handle_unauthorized_access)
        self.monitor.register_event_handler("brute_force", self._handle_brute_force)
        self.monitor.register_event_handler("critical_threat", self._handle_critical_threat)
        
        # Setup secure communication handlers
        self.communicator.register_message_handler("security_alert", self._handle_security_message)
        self.communicator.register_message_handler("access_request", self._handle_access_request)
    
    def initialize_security_system(self) -> Dict[str, Any]:
        """Initialize the complete security system."""
        initialization_results = {
            "security_manager": False,
            "communications": False,
            "access_control": False,
            "monitoring": False,
            "overall_success": False
        }
        
        try:
            # Initialize security manager (always successful)
            initialization_results["security_manager"] = True
            
            # Start secure communications if configured
            if self.config["auto_start_communications"]:
                initialization_results["communications"] = self.communicator.start_secure_server()
            else:
                initialization_results["communications"] = True
            
            # Initialize access control (always successful)
            initialization_results["access_control"] = True
            
            # Start security monitoring if configured
            if self.config["auto_start_monitoring"]:
                initialization_results["monitoring"] = self.monitor.start_monitoring()
            else:
                initialization_results["monitoring"] = True
            
            # Check overall success
            initialization_results["overall_success"] = all([
                initialization_results["security_manager"],
                initialization_results["communications"],
                initialization_results["access_control"],
                initialization_results["monitoring"]
            ])
            
            if initialization_results["overall_success"]:
                self.running = True
                self.security_state["last_security_check"] = datetime.now().isoformat()
                self._log_security_event("system_initialized", "JARVIS security system initialized successfully")
            
            return initialization_results
            
        except Exception as e:
            self._log_security_event("initialization_error", f"Security system initialization failed: {e}")
            return initialization_results
    
    def authenticate_user(self, username: str, password: str = None, 
                         biometric_data: bytes = None, 
                         client_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive user authentication."""
        try:
            # Use security manager for authentication
            auth_result = self.security_manager.authenticate_user(
                username, password, biometric_data, 
                client_info.get("ip_address") if client_info else None
            )
            
            if auth_result["success"]:
                # Create session in access control
                session_id = self.access_control.create_session(username, client_info)
                
                if session_id:
                    auth_result["access_session_id"] = session_id
                    
                    # Log successful authentication
                    self._log_security_event("user_authenticated", 
                                           f"User {username} authenticated successfully")
                    
                    return auth_result
                else:
                    return {
                        "success": False,
                        "error": "Failed to create access session"
                    }
            else:
                # Log failed authentication
                self._log_security_event("authentication_failed", 
                                       f"Authentication failed for user {username}")
                return auth_result
                
        except Exception as e:
            self._log_security_event("authentication_error", f"Authentication error: {e}")
            return {
                "success": False,
                "error": "Authentication system error"
            }
    
    def request_system_access(self, username: str, resource: str, action: str, 
                            session_id: str = None) -> Dict[str, Any]:
        """Request access to system resources."""
        try:
            # Validate session first
            if session_id:
                session_validation = self.access_control.validate_session(session_id)
                if not session_validation["valid"]:
                    return {
                        "allowed": False,
                        "error": "Invalid or expired session"
                    }
            
            # Check permissions
            permission_result = self.access_control.check_permission(username, resource, action)
            
            if permission_result["allowed"]:
                # Request access
                access_result = self.access_control.request_resource_access(
                    username, resource, action, session_id
                )
                
                # Log access request
                self._log_security_event("access_granted", 
                                       f"Access granted to {username} for {resource}:{action}")
                
                return access_result
            else:
                # Log access denial
                self._log_security_event("access_denied", 
                                       f"Access denied to {username} for {resource}:{action}")
                return permission_result
                
        except Exception as e:
            self._log_security_event("access_error", f"Access request error: {e}")
            return {
                "allowed": False,
                "error": "Access control system error"
            }
    
    def send_secure_message(self, message: str, recipient: str = None, 
                          encrypt: bool = True) -> Dict[str, Any]:
        """Send secure message through the communication system."""
        try:
            if encrypt:
                success = self.communicator.send_secure_message(message, recipient)
            else:
                # For non-encrypted messages, use basic sending
                success = self.communicator.send_secure_message(message, recipient)
            
            if success:
                self._log_security_event("message_sent", f"Secure message sent to {recipient or 'all'}")
                return {"success": True, "message": "Message sent successfully"}
            else:
                return {"success": False, "error": "Failed to send message"}
                
        except Exception as e:
            self._log_security_event("message_error", f"Message sending error: {e}")
            return {"success": False, "error": "Communication system error"}
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        try:
            # Get component statuses
            comm_status = self.communicator.get_connection_status()
            monitor_dashboard = self.monitor.get_security_dashboard()
            access_status = self.access_control.get_system_status()
            security_report = self.security_manager.get_security_report()
            
            return {
                "system_running": self.running,
                "security_state": self.security_state,
                "components": {
                    "security_manager": {
                        "active": True,
                        "recent_events": len(security_report.get("recent_events", [])),
                        "blocked_ips": security_report["statistics"]["blocked_ips"]
                    },
                    "communications": {
                        "server_running": comm_status["server_running"],
                        "trusted_clients": comm_status["trusted_clients"],
                        "encryption_available": comm_status["encryption_available"]
                    },
                    "access_control": {
                        "active_sessions": access_status["active_sessions"],
                        "locked_resources": access_status["locked_resources"],
                        "total_users": access_status["total_users"]
                    },
                    "monitoring": {
                        "status": monitor_dashboard["monitoring_status"],
                        "threat_level": monitor_dashboard["current_threat_level"],
                        "recent_alerts": len(monitor_dashboard["recent_alerts"])
                    }
                },
                "overall_threat_level": self._calculate_overall_threat_level(),
                "recommendations": self._generate_security_recommendations()
            }
            
        except Exception as e:
            self._log_security_event("status_error", f"Failed to get security status: {e}")
            return {
                "error": "Failed to retrieve security status",
                "system_running": False
            }
    
    def _calculate_overall_threat_level(self) -> str:
        """Calculate overall system threat level."""
        try:
            # Get threat levels from monitoring
            monitor_dashboard = self.monitor.get_security_dashboard()
            monitor_threat = monitor_dashboard.get("current_threat_level", "LOW")
            
            # Get security metrics
            security_report = self.security_manager.get_security_report()
            
            # Simple threat level calculation
            threat_indicators = 0
            
            if monitor_threat in ["HIGH", "CRITICAL"]:
                threat_indicators += 2
            elif monitor_threat == "MEDIUM":
                threat_indicators += 1
            
            if security_report["statistics"]["blocked_ips"] > 5:
                threat_indicators += 1
            
            if len(self.security_state["active_incidents"]) > 0:
                threat_indicators += 1
            
            if threat_indicators >= 3:
                return "CRITICAL"
            elif threat_indicators >= 2:
                return "HIGH"
            elif threat_indicators >= 1:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception:
            return "UNKNOWN"
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        try:
            overall_threat = self._calculate_overall_threat_level()
            
            if overall_threat in ["HIGH", "CRITICAL"]:
                recommendations.append("Consider enabling emergency mode")
                recommendations.append("Review and strengthen access controls")
                recommendations.append("Increase monitoring frequency")
            
            if not self.config["security_policies"]["require_mfa"]:
                recommendations.append("Consider enabling multi-factor authentication")
            
            # Get monitoring dashboard for specific recommendations
            monitor_dashboard = self.monitor.get_security_dashboard()
            system_metrics = monitor_dashboard.get("system_metrics", {})
            
            if system_metrics.get("cpu_usage", 0) > 80:
                recommendations.append("High CPU usage detected - investigate potential threats")
            
            if system_metrics.get("memory_usage", 0) > 85:
                recommendations.append("High memory usage detected - check for memory leaks or attacks")
            
            if not recommendations:
                recommendations.append("Security system operating normally")
                
        except Exception:
            recommendations.append("Unable to generate recommendations - system check required")
        
        return recommendations
    
    def activate_emergency_mode(self, reason: str = "Manual activation") -> bool:
        """Activate emergency security mode."""
        try:
            self.security_state["emergency_mode"] = True
            self.security_state["security_level"] = "EMERGENCY"
            
            # Lock down system
            self.security_state["system_locked"] = True
            
            # Create security incident
            incident = {
                "id": f"incident_{int(time.time())}",
                "type": "emergency_activation",
                "reason": reason,
                "activated_at": datetime.now().isoformat(),
                "status": "active"
            }
            
            self.security_state["active_incidents"].append(incident)
            
            # Log emergency activation
            self._log_security_event("emergency_activated", f"Emergency mode activated: {reason}")
            
            # Notify through communication system
            self.send_secure_message(f"EMERGENCY MODE ACTIVATED: {reason}", encrypt=True)
            
            return True
            
        except Exception as e:
            self._log_security_event("emergency_error", f"Failed to activate emergency mode: {e}")
            return False
    
    def deactivate_emergency_mode(self, admin_user: str) -> bool:
        """Deactivate emergency security mode."""
        try:
            if not self.security_state["emergency_mode"]:
                return True
            
            self.security_state["emergency_mode"] = False
            self.security_state["system_locked"] = False
            self.security_state["security_level"] = "NORMAL"
            
            # Close active emergency incidents
            for incident in self.security_state["active_incidents"]:
                if incident["type"] == "emergency_activation" and incident["status"] == "active":
                    incident["status"] = "resolved"
                    incident["resolved_at"] = datetime.now().isoformat()
                    incident["resolved_by"] = admin_user
            
            # Log deactivation
            self._log_security_event("emergency_deactivated", f"Emergency mode deactivated by {admin_user}")
            
            return True
            
        except Exception as e:
            self._log_security_event("emergency_error", f"Failed to deactivate emergency mode: {e}")
            return False
    
    def _handle_unauthorized_access(self, alert: Dict[str, Any]):
        """Handle unauthorized access alerts."""
        self._log_security_event("unauthorized_access_detected", 
                                f"Unauthorized access detected: {alert['message']}")
        
        # Auto-lock system if configured
        if self.config["security_policies"]["auto_lock_on_threat"]:
            self.activate_emergency_mode("Unauthorized access detected")
    
    def _handle_brute_force(self, alert: Dict[str, Any]):
        """Handle brute force attack alerts."""
        self._log_security_event("brute_force_detected", 
                                f"Brute force attack detected: {alert['message']}")
        
        # Implement additional security measures
        details = alert.get("details", {})
        ip_address = details.get("ip_address")
        
        if ip_address:
            # Add to blocked IPs in security manager
            self.security_manager.blocked_ips.add(ip_address)
    
    def _handle_critical_threat(self, alert: Dict[str, Any]):
        """Handle critical threat alerts."""
        self._log_security_event("critical_threat_detected", 
                                f"Critical threat detected: {alert['message']}")
        
        # Automatically activate emergency mode for critical threats
        self.activate_emergency_mode(f"Critical threat: {alert['message']}")
    
    def _handle_security_message(self, message: str, client_info: Any):
        """Handle security-related messages."""
        self._log_security_event("security_message_received", 
                                f"Security message received from {client_info}")
    
    def _handle_access_request(self, message: str, client_info: Any):
        """Handle access request messages."""
        self._log_security_event("access_request_received", 
                                f"Access request received from {client_info}")
    
    def _log_security_event(self, event_type: str, message: str):
        """Log security event across all components."""
        timestamp = datetime.now().isoformat()
        
        # Log to security manager
        self.security_manager._log_security_event(event_type, message)
        
        # Create monitoring alert if needed
        if event_type in ["unauthorized_access_detected", "brute_force_detected", "critical_threat_detected"]:
            self.monitor._create_alert(
                ThreatLevel.HIGH,
                SecurityEvent.SUSPICIOUS_ACTIVITY,
                message,
                {"source": "security_integration", "event_type": event_type}
            )
    
    def shutdown_security_system(self):
        """Shutdown the security system gracefully."""
        try:
            self.running = False
            
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            # Stop communications
            self.communicator.stop_server()
            
            # End all sessions
            for session_id in list(self.access_control.active_sessions.keys()):
                self.access_control.end_session(session_id)
            
            self._log_security_event("system_shutdown", "JARVIS security system shutdown")
            
        except Exception as e:
            print(f"Error during security system shutdown: {e}")


# Example usage and integration testing
if __name__ == "__main__":
    # Initialize integrated security system
    security_system = JarvisSecuritySystem()
    
    # Initialize the system
    print("Initializing JARVIS Security System...")
    init_result = security_system.initialize_security_system()
    
    print("Initialization Results:")
    for component, status in init_result.items():
        print(f"  {component}: {'✓' if status else '✗'}")
    
    if init_result["overall_success"]:
        print("\n✓ JARVIS Security System initialized successfully!")
        
        # Test user authentication
        print("\nTesting user authentication...")
        auth_result = security_system.authenticate_user(
            "admin", 
            "secure_password123",
            client_info={"ip_address": "192.168.1.100", "client_type": "desktop"}
        )
        print(f"Authentication result: {auth_result.get('success', False)}")
        
        if auth_result.get("success"):
            # Test resource access
            print("\nTesting resource access...")
            access_result = security_system.request_system_access(
                "admin", 
                "camera", 
                "read",
                auth_result.get("access_session_id")
            )
            print(f"Access result: {access_result.get('allowed', False)}")
        
        # Test secure messaging
        print("\nTesting secure messaging...")
        message_result = security_system.send_secure_message(
            "Test security message", 
            "admin", 
            encrypt=True
        )
        print(f"Message result: {message_result.get('success', False)}")
        
        # Get security status
        print("\nGetting security status...")
        status = security_system.get_security_status()
        print(f"Overall threat level: {status.get('overall_threat_level', 'Unknown')}")
        
        # Show recommendations
        recommendations = status.get("recommendations", [])
        print(f"\nSecurity recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Test emergency mode
        print("\nTesting emergency mode...")
        emergency_result = security_system.activate_emergency_mode("Testing emergency procedures")
        print(f"Emergency mode activated: {emergency_result}")
        
        # Deactivate emergency mode
        deactivate_result = security_system.deactivate_emergency_mode("admin")
        print(f"Emergency mode deactivated: {deactivate_result}")
        
        # Shutdown system
        print("\nShutting down security system...")
        security_system.shutdown_security_system()
        print("Security system shutdown complete.")
        
    else:
        print("\n✗ Failed to initialize JARVIS Security System")
        print("Check component initialization results above.")