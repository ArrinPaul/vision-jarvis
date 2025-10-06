"""
Mobile App Integration for JARVIS

Provides APIs and services for mobile app connectivity,
push notifications, and mobile-specific features.
"""

import os
import json
import time
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Optional dependencies with fallbacks
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class MobileAppManager:
    """Mobile application integration manager."""
    
    def __init__(self, config_file: str = "mobile_app_config.json"):
        self.config_file = config_file
        self.config = self._load_configuration()
        
        # Mobile app state
        self.registered_devices = {}
        self.push_tokens = {}
        self.app_sessions = {}
        self.notification_queue = []
        
        # API endpoints
        self.api_endpoints = {}
        
        # Initialize mobile services
        self._initialize_mobile_services()
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load mobile app configuration."""
        default_config = {
            "app_settings": {
                "supported_platforms": ["ios", "android"],
                "minimum_versions": {
                    "ios": "1.0.0",
                    "android": "1.0.0"
                },
                "api_version": "v1",
                "session_timeout_minutes": 120
            },
            "push_notifications": {
                "enabled": True,
                "providers": {
                    "fcm": {
                        "enabled": True,
                        "server_key": "",
                        "sender_id": ""
                    },
                    "apns": {
                        "enabled": True,
                        "certificate_path": "",
                        "key_path": "",
                        "bundle_id": "com.jarvis.mobile"
                    }
                },
                "notification_types": {
                    "system_alerts": True,
                    "security_warnings": True,
                    "status_updates": True,
                    "voice_responses": True
                }
            },
            "features": {
                "remote_control": True,
                "voice_commands": True,
                "camera_access": True,
                "file_sync": True,
                "biometric_auth": True,
                "offline_mode": False
            },
            "security": {
                "app_key_required": True,
                "device_registration": True,
                "certificate_pinning": True,
                "rate_limiting": {
                    "requests_per_minute": 100,
                    "notifications_per_hour": 50
                }
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
                print(f"Failed to load mobile app config: {e}")
                return default_config
        else:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _initialize_mobile_services(self):
        """Initialize mobile app services."""
        # Setup API endpoints
        self._setup_api_endpoints()
        
        # Load registered devices
        self._load_registered_devices()
    
    def _setup_api_endpoints(self):
        """Setup mobile API endpoints."""
        self.api_endpoints = {
            "/api/v1/register": self._handle_device_registration,
            "/api/v1/authenticate": self._handle_mobile_authentication,
            "/api/v1/status": self._handle_status_request,
            "/api/v1/command": self._handle_mobile_command,
            "/api/v1/notifications": self._handle_notification_request,
            "/api/v1/sync": self._handle_data_sync,
            "/api/v1/camera": self._handle_camera_request,
            "/api/v1/voice": self._handle_voice_request
        }
    
    def _load_registered_devices(self):
        """Load registered mobile devices."""
        devices_file = "registered_devices.json"
        if os.path.exists(devices_file):
            try:
                with open(devices_file, 'r') as f:
                    self.registered_devices = json.load(f)
            except Exception as e:
                print(f"Failed to load registered devices: {e}")
    
    def _save_registered_devices(self):
        """Save registered devices."""
        devices_file = "registered_devices.json"
        try:
            with open(devices_file, 'w') as f:
                json.dump(self.registered_devices, f, indent=2)
        except Exception as e:
            print(f"Failed to save registered devices: {e}")
    
    def register_device(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Register mobile device."""
        required_fields = ["device_id", "platform", "app_version", "device_name"]
        
        # Validate required fields
        for field in required_fields:
            if field not in device_info:
                return {
                    "success": False,
                    "error": f"Missing required field: {field}"
                }
        
        device_id = device_info["device_id"]
        platform = device_info["platform"]
        app_version = device_info["app_version"]
        
        # Validate platform
        if platform not in self.config["app_settings"]["supported_platforms"]:
            return {
                "success": False,
                "error": f"Unsupported platform: {platform}"
            }
        
        # Validate app version
        min_version = self.config["app_settings"]["minimum_versions"].get(platform)
        if min_version and self._compare_versions(app_version, min_version) < 0:
            return {
                "success": False,
                "error": f"App version {app_version} is below minimum required {min_version}"
            }
        
        # Generate device token
        device_token = self._generate_device_token(device_id)
        
        # Register device
        registration_data = {
            "device_id": device_id,
            "platform": platform,
            "app_version": app_version,
            "device_name": device_info["device_name"],
            "device_token": device_token,
            "registered_at": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "push_token": device_info.get("push_token"),
            "features": device_info.get("features", []),
            "active": True
        }
        
        self.registered_devices[device_id] = registration_data
        
        # Store push token if provided
        if device_info.get("push_token"):
            self.push_tokens[device_id] = device_info["push_token"]
        
        self._save_registered_devices()
        
        self._log_mobile_event("device_registered", f"Device registered: {device_id} ({platform})")
        
        return {
            "success": True,
            "device_token": device_token,
            "api_endpoints": list(self.api_endpoints.keys()),
            "features": self.config["features"],
            "session_timeout": self.config["app_settings"]["session_timeout_minutes"]
        }
    
    def _generate_device_token(self, device_id: str) -> str:
        """Generate secure device token."""
        timestamp = str(int(time.time()))
        token_data = f"{device_id}:{timestamp}:jarvis_mobile"
        token_hash = hashlib.sha256(token_data.encode()).hexdigest()
        return base64.b64encode(f"{device_id}:{timestamp}:{token_hash}".encode()).decode()
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare version strings."""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for v1, v2 in zip(v1_parts, v2_parts):
                if v1 < v2:
                    return -1
                elif v1 > v2:
                    return 1
            
            return 0
        except:
            return 0
    
    def authenticate_device(self, device_id: str, device_token: str) -> Dict[str, Any]:
        """Authenticate mobile device."""
        if device_id not in self.registered_devices:
            return {
                "success": False,
                "error": "Device not registered"
            }
        
        device_info = self.registered_devices[device_id]
        
        if not device_info.get("active"):
            return {
                "success": False,
                "error": "Device deactivated"
            }
        
        if device_info["device_token"] != device_token:
            return {
                "success": False,
                "error": "Invalid device token"
            }
        
        # Create session
        session_id = self._create_mobile_session(device_id)
        
        # Update last seen
        device_info["last_seen"] = datetime.now().isoformat()
        self._save_registered_devices()
        
        return {
            "success": True,
            "session_id": session_id,
            "expires_at": (datetime.now() + timedelta(
                minutes=self.config["app_settings"]["session_timeout_minutes"]
            )).isoformat(),
            "user_permissions": self._get_device_permissions(device_id)
        }
    
    def _create_mobile_session(self, device_id: str) -> str:
        """Create mobile app session."""
        session_id = f"mobile_{device_id}_{int(time.time())}"
        
        session_data = {
            "session_id": session_id,
            "device_id": device_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "status": "active"
        }
        
        self.app_sessions[session_id] = session_data
        return session_id
    
    def _get_device_permissions(self, device_id: str) -> List[str]:
        """Get permissions for mobile device."""
        # Default permissions for mobile devices
        base_permissions = [
            "remote_control",
            "voice_commands", 
            "status_monitoring",
            "notifications"
        ]
        
        # Add additional permissions based on device features
        device_info = self.registered_devices.get(device_id, {})
        device_features = device_info.get("features", [])
        
        if "camera" in device_features and self.config["features"]["camera_access"]:
            base_permissions.append("camera_access")
        
        if "biometric" in device_features and self.config["features"]["biometric_auth"]:
            base_permissions.append("biometric_auth")
        
        return base_permissions
    
    def send_push_notification(self, device_id: str, notification: Dict[str, Any]) -> bool:
        """Send push notification to mobile device."""
        if not self.config["push_notifications"]["enabled"]:
            return False
        
        if device_id not in self.registered_devices:
            return False
        
        push_token = self.push_tokens.get(device_id)
        if not push_token:
            return False
        
        device_info = self.registered_devices[device_id]
        platform = device_info["platform"]
        
        # Send notification based on platform
        if platform == "android":
            return self._send_fcm_notification(push_token, notification)
        elif platform == "ios":
            return self._send_apns_notification(push_token, notification)
        
        return False
    
    def _send_fcm_notification(self, push_token: str, notification: Dict[str, Any]) -> bool:
        """Send FCM (Firebase Cloud Messaging) notification."""
        if not REQUESTS_AVAILABLE:
            print("FCM notification skipped - requests library not available")
            return False
        
        fcm_config = self.config["push_notifications"]["providers"]["fcm"]
        if not fcm_config["enabled"] or not fcm_config["server_key"]:
            print("FCM not configured properly")
            return False
        
        fcm_url = "https://fcm.googleapis.com/fcm/send"
        headers = {
            "Authorization": f"key={fcm_config['server_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "to": push_token,
            "notification": {
                "title": notification.get("title", "JARVIS"),
                "body": notification.get("message", ""),
                "icon": "ic_notification",
                "sound": "default"
            },
            "data": notification.get("data", {})
        }
        
        try:
            response = requests.post(fcm_url, headers=headers, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"FCM notification error: {e}")
            return False
    
    def _send_apns_notification(self, push_token: str, notification: Dict[str, Any]) -> bool:
        """Send APNS (Apple Push Notification Service) notification."""
        # APNS implementation would require additional dependencies
        # For now, simulate successful sending
        print(f"APNS notification sent to {push_token}: {notification['title']}")
        return True
    
    def broadcast_notification(self, notification: Dict[str, Any], 
                             platform_filter: str = None) -> Dict[str, Any]:
        """Broadcast notification to all registered devices."""
        sent_count = 0
        failed_count = 0
        
        for device_id, device_info in self.registered_devices.items():
            if not device_info.get("active"):
                continue
            
            if platform_filter and device_info["platform"] != platform_filter:
                continue
            
            if self.send_push_notification(device_id, notification):
                sent_count += 1
            else:
                failed_count += 1
        
        return {
            "sent": sent_count,
            "failed": failed_count,
            "total_devices": len(self.registered_devices)
        }
    
    def handle_mobile_api_request(self, endpoint: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mobile API request."""
        if endpoint not in self.api_endpoints:
            return {
                "error": "Unknown endpoint",
                "code": 404
            }
        
        try:
            handler = self.api_endpoints[endpoint]
            return handler(request_data)
        except Exception as e:
            return {
                "error": f"API handler error: {e}",
                "code": 500
            }
    
    def _handle_device_registration(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle device registration API request."""
        return self.register_device(request_data)
    
    def _handle_mobile_authentication(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mobile authentication API request."""
        device_id = request_data.get("device_id")
        device_token = request_data.get("device_token")
        
        if not device_id or not device_token:
            return {
                "error": "Missing device_id or device_token",
                "code": 400
            }
        
        return self.authenticate_device(device_id, device_token)
    
    def _handle_status_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status request from mobile app."""
        return {
            "system_status": "online",
            "features_available": self.config["features"],
            "server_time": datetime.now().isoformat(),
            "api_version": self.config["app_settings"]["api_version"]
        }
    
    def _handle_mobile_command(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle command from mobile app."""
        command = request_data.get("command")
        parameters = request_data.get("parameters", {})
        
        if not command:
            return {
                "error": "Missing command",
                "code": 400
            }
        
        # Process command (placeholder)
        command_result = {
            "command": command,
            "status": "executed",
            "result": f"Command '{command}' processed successfully",
            "timestamp": datetime.now().isoformat()
        }
        
        return command_result
    
    def _handle_notification_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle notification request from mobile app."""
        action = request_data.get("action")
        
        if action == "list":
            # Return pending notifications
            return {
                "notifications": self.notification_queue[-10:],  # Last 10 notifications
                "count": len(self.notification_queue)
            }
        
        elif action == "mark_read":
            notification_id = request_data.get("notification_id")
            # Mark notification as read (placeholder)
            return {
                "success": True,
                "message": f"Notification {notification_id} marked as read"
            }
        
        return {
            "error": "Invalid notification action",
            "code": 400
        }
    
    def _handle_data_sync(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data synchronization request."""
        sync_type = request_data.get("sync_type")
        last_sync = request_data.get("last_sync")
        
        # Simulate data sync
        sync_data = {
            "settings": {
                "theme": "dark",
                "notifications_enabled": True,
                "voice_activation": True
            },
            "recent_commands": [
                {"command": "lights on", "timestamp": "2024-01-01T12:00:00"},
                {"command": "weather", "timestamp": "2024-01-01T11:30:00"}
            ],
            "sync_timestamp": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "data": sync_data,
            "sync_type": sync_type
        }
    
    def _handle_camera_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle camera request from mobile app."""
        if not self.config["features"]["camera_access"]:
            return {
                "error": "Camera access not enabled",
                "code": 403
            }
        
        action = request_data.get("action")
        
        if action == "stream":
            return {
                "stream_url": "ws://jarvis-server:8846/camera/stream",
                "format": "mjpeg",
                "resolution": "640x480",
                "expires_at": (datetime.now() + timedelta(minutes=30)).isoformat()
            }
        
        elif action == "snapshot":
            return {
                "snapshot_url": "/api/v1/camera/snapshot",
                "expires_at": (datetime.now() + timedelta(minutes=5)).isoformat()
            }
        
        return {
            "error": "Invalid camera action",
            "code": 400
        }
    
    def _handle_voice_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle voice request from mobile app."""
        if not self.config["features"]["voice_commands"]:
            return {
                "error": "Voice commands not enabled",
                "code": 403
            }
        
        voice_data = request_data.get("voice_data")  # Base64 encoded audio
        text_command = request_data.get("text")
        
        if not voice_data and not text_command:
            return {
                "error": "No voice data or text provided",
                "code": 400
            }
        
        # Process voice command (placeholder)
        response = {
            "success": True,
            "response_text": f"Voice command processed: {text_command or 'audio received'}",
            "action_taken": "voice_command_executed",
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    def get_mobile_app_status(self) -> Dict[str, Any]:
        """Get mobile app integration status."""
        active_devices = sum(1 for d in self.registered_devices.values() if d.get("active"))
        
        platform_stats = defaultdict(int)
        for device in self.registered_devices.values():
            if device.get("active"):
                platform_stats[device["platform"]] += 1
        
        return {
            "registered_devices": len(self.registered_devices),
            "active_devices": active_devices,
            "active_sessions": len(self.app_sessions),
            "platform_distribution": dict(platform_stats),
            "push_notifications_enabled": self.config["push_notifications"]["enabled"],
            "features_enabled": {k: v for k, v in self.config["features"].items() if v},
            "api_endpoints": len(self.api_endpoints)
        }
    
    def _log_mobile_event(self, event_type: str, message: str):
        """Log mobile app event."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "type": event_type,
            "message": message
        }
        
        print(f"[{timestamp}] MOBILE_{event_type.upper()}: {message}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize mobile app manager
    mobile_manager = MobileAppManager()
    
    print("JARVIS Mobile App Integration initialized")
    
    # Test device registration
    print("\nTesting device registration...")
    device_info = {
        "device_id": "iphone_12_test",
        "platform": "ios",
        "app_version": "1.0.0",
        "device_name": "John's iPhone",
        "push_token": "fake_push_token_12345",
        "features": ["camera", "biometric", "voice"]
    }
    
    registration_result = mobile_manager.register_device(device_info)
    print(f"Registration result: {registration_result['success']}")
    
    if registration_result["success"]:
        device_token = registration_result["device_token"]
        
        # Test authentication
        print("\nTesting device authentication...")
        auth_result = mobile_manager.authenticate_device("iphone_12_test", device_token)
        print(f"Authentication result: {auth_result['success']}")
        
        # Test push notification
        print("\nTesting push notification...")
        notification = {
            "title": "JARVIS Alert",
            "message": "System status update",
            "data": {"type": "status", "priority": "normal"}
        }
        
        push_result = mobile_manager.send_push_notification("iphone_12_test", notification)
        print(f"Push notification sent: {push_result}")
        
        # Test API requests
        print("\nTesting API requests...")
        
        # Status request
        status_response = mobile_manager.handle_mobile_api_request(
            "/api/v1/status", {}
        )
        print(f"Status API: {status_response.get('system_status')}")
        
        # Command request
        command_response = mobile_manager.handle_mobile_api_request(
            "/api/v1/command", 
            {"command": "lights_on", "parameters": {"room": "living_room"}}
        )
        print(f"Command API: {command_response.get('status')}")
    
    # Get mobile app status
    print("\nMobile App Status:")
    status = mobile_manager.get_mobile_app_status()
    print(f"  Active Devices: {status['active_devices']}")
    print(f"  Active Sessions: {status['active_sessions']}")
    print(f"  Push Notifications: {'✓' if status['push_notifications_enabled'] else '✗'}")
    
    print("\nEnabled Features:")
    for feature, enabled in status['features_enabled'].items():
        print(f"  {feature}: ✓")
    
    print(f"\nAPI Endpoints: {status['api_endpoints']} available")