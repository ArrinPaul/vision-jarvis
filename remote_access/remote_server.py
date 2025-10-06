"""
Remote Access Server for JARVIS

Implements secure remote access capabilities allowing external clients
to connect and interact with the JARVIS system securely.
"""

import os
import json
import time
import socket
import threading
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict

# Optional dependencies with fallbacks
try:
    import ssl
    SSL_AVAILABLE = True
except ImportError:
    SSL_AVAILABLE = False

try:
    import websockets
    import asyncio
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

class RemoteAccessServer:
    """Secure remote access server for JARVIS."""
    
    def __init__(self, config_file: str = "remote_access_config.json"):
        self.config_file = config_file
        self.config = self._load_configuration()
        
        # Server state
        self.running = False
        self.server_socket = None
        self.websocket_server = None
        
        # Client management
        self.connected_clients = {}
        self.client_sessions = {}
        self.pending_connections = {}
        
        # Security integration
        self.security_system = None  # Will be injected
        
        # Command handlers
        self.command_handlers = {}
        self.api_endpoints = {}
        
        # Initialize server components
        self._initialize_server()
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load remote access configuration."""
        default_config = {
            "server_settings": {
                "host": "0.0.0.0",
                "port": 8844,
                "websocket_port": 8845,
                "max_connections": 10,
                "connection_timeout": 300
            },
            "security_settings": {
                "require_authentication": True,
                "require_encryption": True,
                "allowed_origins": ["localhost", "127.0.0.1"],
                "rate_limiting": {
                    "requests_per_minute": 60,
                    "max_message_size": 1048576  # 1MB
                }
            },
            "features": {
                "enable_file_transfer": True,
                "enable_camera_streaming": True,
                "enable_voice_commands": True,
                "enable_gesture_control": False,
                "enable_system_control": False
            },
            "mobile_app": {
                "enable_push_notifications": True,
                "supported_platforms": ["ios", "android"],
                "app_version_requirements": {
                    "minimum": "1.0.0",
                    "recommended": "1.2.0"
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
                print(f"Failed to load remote access config: {e}")
                return default_config
        else:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _initialize_server(self):
        """Initialize server components."""
        # Register default command handlers
        self._register_default_handlers()
        
        # Setup SSL context if available
        self.ssl_context = self._setup_ssl_context() if SSL_AVAILABLE else None
    
    def _register_default_handlers(self):
        """Register default command handlers."""
        self.register_command_handler("ping", self._handle_ping)
        self.register_command_handler("status", self._handle_status)
        self.register_command_handler("authenticate", self._handle_authenticate)
        self.register_command_handler("voice_command", self._handle_voice_command)
        self.register_command_handler("camera_stream", self._handle_camera_stream)
        self.register_command_handler("file_transfer", self._handle_file_transfer)
        self.register_command_handler("system_info", self._handle_system_info)
    
    def _setup_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Setup SSL context for secure connections."""
        if not SSL_AVAILABLE:
            return None
        
        try:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            
            # Use existing certificate or create new one
            cert_file = "remote_access_cert.pem"
            key_file = "remote_access_key.pem"
            
            if not os.path.exists(cert_file) or not os.path.exists(key_file):
                self._generate_self_signed_cert(cert_file, key_file)
            
            context.load_cert_chain(cert_file, key_file)
            return context
            
        except Exception as e:
            print(f"Failed to setup SSL context: {e}")
            return None
    
    def _generate_self_signed_cert(self, cert_file: str, key_file: str):
        """Generate self-signed certificate for remote access."""
        try:
            import subprocess
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_file, "-out", cert_file,
                "-days", "365", "-nodes",
                "-subj", "/C=US/ST=State/L=City/O=JARVIS/CN=jarvis-remote"
            ], check=True, capture_output=True)
        except:
            # Fallback: create dummy files
            with open(cert_file, 'w') as f:
                f.write("-----BEGIN CERTIFICATE-----\nDUMMY CERTIFICATE\n-----END CERTIFICATE-----")
            with open(key_file, 'w') as f:
                f.write("-----BEGIN PRIVATE KEY-----\nDUMMY KEY\n-----END PRIVATE KEY-----")
    
    def start_server(self) -> bool:
        """Start the remote access server."""
        try:
            # Start TCP server
            tcp_result = self._start_tcp_server()
            
            # Start WebSocket server if available
            websocket_result = True
            if WEBSOCKETS_AVAILABLE:
                websocket_result = self._start_websocket_server()
            
            if tcp_result and websocket_result:
                self.running = True
                self._log_event("server_started", "Remote access server started successfully")
                return True
            else:
                return False
                
        except Exception as e:
            self._log_event("server_error", f"Failed to start remote access server: {e}")
            return False
    
    def _start_tcp_server(self) -> bool:
        """Start TCP server for remote connections."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            host = self.config["server_settings"]["host"]
            port = self.config["server_settings"]["port"]
            
            self.server_socket.bind((host, port))
            self.server_socket.listen(self.config["server_settings"]["max_connections"])
            
            # Start server thread
            server_thread = threading.Thread(target=self._tcp_server_loop, daemon=True)
            server_thread.start()
            
            print(f"TCP server started on {host}:{port}")
            return True
            
        except Exception as e:
            print(f"Failed to start TCP server: {e}")
            return False
    
    def _start_websocket_server(self) -> bool:
        """Start WebSocket server for web-based connections."""
        if not WEBSOCKETS_AVAILABLE:
            return True  # Skip if not available
        
        try:
            # This would typically run in an async context
            # For this demo, we'll simulate WebSocket server startup
            websocket_port = self.config["server_settings"]["websocket_port"]
            print(f"WebSocket server would start on port {websocket_port}")
            return True
            
        except Exception as e:
            print(f"Failed to start WebSocket server: {e}")
            return False
    
    def _tcp_server_loop(self):
        """Main TCP server loop."""
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_tcp_client,
                    args=(client_socket, client_address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"TCP server error: {e}")
                break
    
    def _handle_tcp_client(self, client_socket: socket.socket, client_address):
        """Handle individual TCP client connection."""
        client_id = str(uuid.uuid4())
        
        try:
            # Wrap with SSL if available
            if self.ssl_context:
                client_socket = self.ssl_context.wrap_socket(client_socket, server_side=True)
            
            # Register client
            self.connected_clients[client_id] = {
                "socket": client_socket,
                "address": client_address,
                "connected_at": datetime.now().isoformat(),
                "authenticated": False,
                "last_activity": datetime.now().isoformat()
            }
            
            self._log_event("client_connected", f"Client connected from {client_address}")
            
            # Send welcome message
            welcome_msg = {
                "type": "welcome",
                "client_id": client_id,
                "server_version": "1.0.0",
                "features": list(self.config["features"].keys()),
                "authentication_required": self.config["security_settings"]["require_authentication"]
            }
            
            self._send_message(client_socket, welcome_msg)
            
            # Handle client messages
            while self.running:
                data = client_socket.recv(4096)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode())
                    response = self._process_client_message(client_id, message)
                    
                    if response:
                        self._send_message(client_socket, response)
                        
                except json.JSONDecodeError:
                    error_response = {"error": "Invalid message format"}
                    self._send_message(client_socket, error_response)
                
                # Update last activity
                if client_id in self.connected_clients:
                    self.connected_clients[client_id]["last_activity"] = datetime.now().isoformat()
                
        except Exception as e:
            print(f"Client handling error: {e}")
        finally:
            # Clean up client
            if client_id in self.connected_clients:
                del self.connected_clients[client_id]
            client_socket.close()
            self._log_event("client_disconnected", f"Client {client_id} disconnected")
    
    def _send_message(self, client_socket: socket.socket, message: Dict[str, Any]):
        """Send message to client."""
        try:
            message_json = json.dumps(message)
            client_socket.send(message_json.encode())
        except Exception as e:
            print(f"Failed to send message: {e}")
    
    def _process_client_message(self, client_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process message from client."""
        message_type = message.get("type")
        command = message.get("command")
        
        # Check authentication requirement
        client_info = self.connected_clients.get(client_id)
        if not client_info:
            return {"error": "Client not found"}
        
        if (self.config["security_settings"]["require_authentication"] and 
            not client_info["authenticated"] and 
            command != "authenticate"):
            return {"error": "Authentication required"}
        
        # Process command
        if command in self.command_handlers:
            try:
                return self.command_handlers[command](client_id, message)
            except Exception as e:
                return {"error": f"Command processing error: {e}"}
        else:
            return {"error": f"Unknown command: {command}"}
    
    def register_command_handler(self, command: str, handler: Callable):
        """Register command handler."""
        self.command_handlers[command] = handler
    
    def _handle_ping(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping command."""
        return {
            "type": "response",
            "command": "ping",
            "data": {
                "pong": True,
                "timestamp": datetime.now().isoformat(),
                "client_id": client_id
            }
        }
    
    def _handle_status(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status request."""
        return {
            "type": "response",
            "command": "status",
            "data": {
                "server_running": self.running,
                "connected_clients": len(self.connected_clients),
                "uptime": self._calculate_uptime(),
                "features_enabled": {k: v for k, v in self.config["features"].items() if v}
            }
        }
    
    def _handle_authenticate(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle authentication request."""
        credentials = message.get("data", {})
        username = credentials.get("username")
        password = credentials.get("password")
        token = credentials.get("token")
        
        # Use security system for authentication if available
        if self.security_system:
            if token:
                # Token-based authentication
                token_validation = self.security_system.security_manager.validate_access_token(token)
                if token_validation.get("valid"):
                    self.connected_clients[client_id]["authenticated"] = True
                    self.connected_clients[client_id]["username"] = token_validation.get("username")
                    
                    return {
                        "type": "response",
                        "command": "authenticate",
                        "data": {
                            "success": True,
                            "username": token_validation.get("username"),
                            "permissions": token_validation.get("permissions", [])
                        }
                    }
                else:
                    return {
                        "type": "response",
                        "command": "authenticate",
                        "data": {"success": False, "error": "Invalid token"}
                    }
            
            elif username and password:
                # Username/password authentication
                client_info = self.connected_clients[client_id]
                auth_result = self.security_system.authenticate_user(
                    username, password,
                    client_info={"ip_address": client_info["address"][0]}
                )
                
                if auth_result.get("success"):
                    self.connected_clients[client_id]["authenticated"] = True
                    self.connected_clients[client_id]["username"] = username
                    
                    return {
                        "type": "response",
                        "command": "authenticate",
                        "data": {
                            "success": True,
                            "username": username,
                            "session_id": auth_result.get("session_id")
                        }
                    }
                else:
                    return {
                        "type": "response",
                        "command": "authenticate",
                        "data": {"success": False, "error": auth_result.get("error")}
                    }
        
        # Fallback authentication (demo mode)
        if username == "demo" and password == "demo":
            self.connected_clients[client_id]["authenticated"] = True
            self.connected_clients[client_id]["username"] = username
            
            return {
                "type": "response",
                "command": "authenticate",
                "data": {"success": True, "username": username}
            }
        
        return {
            "type": "response",
            "command": "authenticate",
            "data": {"success": False, "error": "Invalid credentials"}
        }
    
    def _handle_voice_command(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle voice command from remote client."""
        if not self.config["features"]["enable_voice_commands"]:
            return {"error": "Voice commands not enabled"}
        
        command_data = message.get("data", {})
        voice_text = command_data.get("text")
        audio_data = command_data.get("audio")  # Base64 encoded audio
        
        if not voice_text and not audio_data:
            return {"error": "No voice data provided"}
        
        # Process voice command (placeholder)
        response_text = f"Processed voice command: {voice_text or 'audio data received'}"
        
        return {
            "type": "response",
            "command": "voice_command",
            "data": {
                "success": True,
                "response": response_text,
                "action_taken": "voice_command_processed"
            }
        }
    
    def _handle_camera_stream(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle camera streaming request."""
        if not self.config["features"]["enable_camera_streaming"]:
            return {"error": "Camera streaming not enabled"}
        
        stream_request = message.get("data", {})
        action = stream_request.get("action")  # start, stop, status
        
        if action == "start":
            # Start camera stream (placeholder)
            return {
                "type": "response",
                "command": "camera_stream",
                "data": {
                    "success": True,
                    "stream_url": f"ws://localhost:8846/stream/{client_id}",
                    "format": "mjpeg",
                    "resolution": "640x480"
                }
            }
        
        elif action == "stop":
            # Stop camera stream
            return {
                "type": "response",
                "command": "camera_stream",
                "data": {"success": True, "message": "Stream stopped"}
            }
        
        elif action == "status":
            # Get stream status
            return {
                "type": "response",
                "command": "camera_stream",
                "data": {
                    "active": False,  # Placeholder
                    "clients": 0,
                    "format": "mjpeg"
                }
            }
        
        return {"error": "Invalid camera stream action"}
    
    def _handle_file_transfer(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file transfer request."""
        if not self.config["features"]["enable_file_transfer"]:
            return {"error": "File transfer not enabled"}
        
        transfer_data = message.get("data", {})
        action = transfer_data.get("action")  # upload, download, list
        
        if action == "list":
            # List available files (placeholder)
            return {
                "type": "response",
                "command": "file_transfer",
                "data": {
                    "success": True,
                    "files": [
                        {"name": "system_log.txt", "size": 1024, "modified": "2024-01-01T12:00:00"},
                        {"name": "config.json", "size": 512, "modified": "2024-01-01T11:00:00"}
                    ]
                }
            }
        
        elif action == "download":
            filename = transfer_data.get("filename")
            if not filename:
                return {"error": "Filename required for download"}
            
            # Handle file download (placeholder)
            return {
                "type": "response",
                "command": "file_transfer",
                "data": {
                    "success": True,
                    "download_url": f"/api/download/{filename}",
                    "expires_at": (datetime.now() + timedelta(minutes=10)).isoformat()
                }
            }
        
        elif action == "upload":
            filename = transfer_data.get("filename")
            file_data = transfer_data.get("data")  # Base64 encoded file data
            
            if not filename or not file_data:
                return {"error": "Filename and data required for upload"}
            
            # Handle file upload (placeholder)
            return {
                "type": "response",
                "command": "file_transfer",
                "data": {
                    "success": True,
                    "message": f"File {filename} uploaded successfully"
                }
            }
        
        return {"error": "Invalid file transfer action"}
    
    def _handle_system_info(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system info request."""
        return {
            "type": "response",
            "command": "system_info",
            "data": {
                "system": "JARVIS Remote Access",
                "version": "1.0.0",
                "uptime": self._calculate_uptime(),
                "features": self.config["features"],
                "connected_clients": len(self.connected_clients),
                "security_enabled": self.config["security_settings"]["require_authentication"]
            }
        }
    
    def _calculate_uptime(self) -> str:
        """Calculate server uptime."""
        # Placeholder - would track actual start time
        return "1h 23m 45s"
    
    def set_security_system(self, security_system):
        """Set security system integration."""
        self.security_system = security_system
    
    def broadcast_message(self, message: Dict[str, Any], authenticated_only: bool = True):
        """Broadcast message to all connected clients."""
        for client_id, client_info in self.connected_clients.items():
            if authenticated_only and not client_info.get("authenticated"):
                continue
            
            try:
                self._send_message(client_info["socket"], message)
            except Exception as e:
                print(f"Failed to broadcast to client {client_id}: {e}")
    
    def send_notification(self, client_id: str, notification: Dict[str, Any]) -> bool:
        """Send notification to specific client."""
        client_info = self.connected_clients.get(client_id)
        if not client_info:
            return False
        
        notification_message = {
            "type": "notification",
            "data": notification
        }
        
        try:
            self._send_message(client_info["socket"], notification_message)
            return True
        except Exception as e:
            print(f"Failed to send notification to {client_id}: {e}")
            return False
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status."""
        return {
            "running": self.running,
            "connections": {
                "total": len(self.connected_clients),
                "authenticated": sum(1 for c in self.connected_clients.values() if c.get("authenticated")),
                "max_allowed": self.config["server_settings"]["max_connections"]
            },
            "security": {
                "ssl_enabled": self.ssl_context is not None,
                "authentication_required": self.config["security_settings"]["require_authentication"],
                "encryption_required": self.config["security_settings"]["require_encryption"]
            },
            "features": self.config["features"],
            "uptime": self._calculate_uptime()
        }
    
    def _log_event(self, event_type: str, message: str):
        """Log server event."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "type": event_type,
            "message": message
        }
        
        # In a real implementation, this would log to file or database
        print(f"[{timestamp}] {event_type.upper()}: {message}")
    
    def stop_server(self):
        """Stop the remote access server."""
        self.running = False
        
        # Close all client connections
        for client_id, client_info in list(self.connected_clients.items()):
            try:
                client_info["socket"].close()
            except:
                pass
        
        self.connected_clients.clear()
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
        
        self._log_event("server_stopped", "Remote access server stopped")


# Example usage and testing
if __name__ == "__main__":
    # Initialize remote access server
    server = RemoteAccessServer()
    
    # Start server
    print("Starting JARVIS Remote Access Server...")
    if server.start_server():
        print("✓ Remote access server started successfully!")
        
        # Show server status
        status = server.get_server_status()
        print("\nServer Status:")
        print(f"  Running: {status['running']}")
        print(f"  Max Connections: {status['connections']['max_allowed']}")
        print(f"  SSL Enabled: {status['security']['ssl_enabled']}")
        print(f"  Authentication Required: {status['security']['authentication_required']}")
        
        print("\nEnabled Features:")
        for feature, enabled in status['features'].items():
            print(f"  {feature}: {'✓' if enabled else '✗'}")
        
        print(f"\nServer listening on port {server.config['server_settings']['port']}")
        print("Connect using: telnet localhost 8844")
        print("Press Ctrl+C to stop server...")
        
        try:
            # Keep server running
            while server.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.stop_server()
            print("Server stopped.")
            
    else:
        print("✗ Failed to start remote access server")