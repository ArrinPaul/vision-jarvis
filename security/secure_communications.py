"""
Encrypted Communications Module for JARVIS

Implements secure communication channels, message encryption,
and secure data transmission protocols.
"""

import os
import json
import time
import socket
import threading
import hashlib
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from collections import deque

# Optional dependencies with fallbacks
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import ssl
    SSL_AVAILABLE = True
except ImportError:
    SSL_AVAILABLE = False

class SecureCommunicator:
    """Secure communication system for JARVIS."""
    
    def __init__(self, port: int = 8443):
        self.port = port
        self.encryption_key = None
        self.private_key = None
        self.public_key = None
        self.trusted_clients = {}
        self.message_queue = deque(maxlen=1000)
        self.message_handlers = {}
        self.running = False
        self.server_socket = None
        
        # Initialize encryption
        self._initialize_encryption()
        
        # Setup SSL context if available
        self.ssl_context = self._setup_ssl_context() if SSL_AVAILABLE else None
    
    def _initialize_encryption(self):
        """Initialize encryption keys."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return
        
        # Generate or load symmetric key
        key_file = "communication_key.key"
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                self.encryption_key = f.read()
        else:
            self.encryption_key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(self.encryption_key)
        
        # Generate or load RSA key pair
        private_key_file = "private_key.pem"
        public_key_file = "public_key.pem"
        
        if os.path.exists(private_key_file) and os.path.exists(public_key_file):
            # Load existing keys
            with open(private_key_file, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(f.read(), password=None)
            
            with open(public_key_file, 'rb') as f:
                self.public_key = serialization.load_pem_public_key(f.read())
        else:
            # Generate new key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
            
            # Save keys
            with open(private_key_file, 'wb') as f:
                f.write(self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            with open(public_key_file, 'wb') as f:
                f.write(self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
    
    def _setup_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Setup SSL context for secure connections."""
        if not SSL_AVAILABLE:
            return None
        
        try:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            
            # Generate self-signed certificate if not exists
            cert_file = "jarvis_cert.pem"
            key_file = "jarvis_key.pem"
            
            if not os.path.exists(cert_file) or not os.path.exists(key_file):
                self._generate_self_signed_cert(cert_file, key_file)
            
            context.load_cert_chain(cert_file, key_file)
            return context
            
        except Exception as e:
            print(f"Failed to setup SSL context: {e}")
            return None
    
    def _generate_self_signed_cert(self, cert_file: str, key_file: str):
        """Generate self-signed certificate."""
        # This is a simplified version - in production, use proper certificate generation
        try:
            import subprocess
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_file, "-out", cert_file,
                "-days", "365", "-nodes",
                "-subj", "/C=US/ST=State/L=City/O=JARVIS/CN=localhost"
            ], check=True, capture_output=True)
        except:
            # Fallback: create dummy files
            with open(cert_file, 'w') as f:
                f.write("DUMMY CERTIFICATE")
            with open(key_file, 'w') as f:
                f.write("DUMMY KEY")
    
    def encrypt_message(self, message: str, recipient_id: str = None) -> Dict[str, Any]:
        """Encrypt message for secure transmission."""
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback: simple base64 encoding (NOT secure)
            encoded = base64.b64encode(message.encode()).decode()
            return {
                "encrypted_data": encoded,
                "method": "base64",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Use symmetric encryption for efficiency
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(message.encode())
            
            # Create message envelope
            envelope = {
                "encrypted_data": encrypted_data.decode(),
                "method": "fernet",
                "timestamp": datetime.now().isoformat(),
                "sender": "jarvis_system",
                "recipient": recipient_id,
                "message_id": hashlib.sha256(f"{message}{time.time()}".encode()).hexdigest()[:16]
            }
            
            return envelope
            
        except Exception as e:
            print(f"Encryption failed: {e}")
            return None
    
    def decrypt_message(self, encrypted_envelope: Dict[str, Any]) -> Optional[str]:
        """Decrypt received message."""
        if not encrypted_envelope:
            return None
        
        method = encrypted_envelope.get("method", "unknown")
        encrypted_data = encrypted_envelope.get("encrypted_data")
        
        if not encrypted_data:
            return None
        
        if method == "base64":
            # Fallback decryption
            try:
                return base64.b64decode(encrypted_data.encode()).decode()
            except:
                return None
        
        elif method == "fernet" and CRYPTOGRAPHY_AVAILABLE:
            try:
                fernet = Fernet(self.encryption_key)
                decrypted_data = fernet.decrypt(encrypted_data.encode())
                return decrypted_data.decode()
            except Exception as e:
                print(f"Decryption failed: {e}")
                return None
        
        return None
    
    def sign_message(self, message: str) -> Optional[str]:
        """Create digital signature for message."""
        if not CRYPTOGRAPHY_AVAILABLE or not self.private_key:
            # Fallback: simple hash
            return hashlib.sha256(message.encode()).hexdigest()
        
        try:
            signature = self.private_key.sign(
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode()
        except Exception as e:
            print(f"Message signing failed: {e}")
            return None
    
    def verify_signature(self, message: str, signature: str, public_key=None) -> bool:
        """Verify message signature."""
        if not signature:
            return False
        
        if not CRYPTOGRAPHY_AVAILABLE or not public_key:
            # Fallback: compare hashes
            expected_hash = hashlib.sha256(message.encode()).hexdigest()
            return signature == expected_hash
        
        try:
            signature_bytes = base64.b64decode(signature.encode())
            public_key.verify(
                signature_bytes,
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            print(f"Signature verification failed: {e}")
            return False
    
    def start_secure_server(self):
        """Start secure communication server."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('localhost', self.port))
            self.server_socket.listen(5)
            
            self.running = True
            
            print(f"Secure server started on port {self.port}")
            
            # Start server thread
            server_thread = threading.Thread(target=self._server_loop, daemon=True)
            server_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Failed to start secure server: {e}")
            return False
    
    def _server_loop(self):
        """Main server loop."""
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"Server error: {e}")
                break
    
    def _handle_client(self, client_socket: socket.socket, client_address):
        """Handle individual client connection."""
        try:
            # Wrap socket with SSL if available
            if self.ssl_context:
                client_socket = self.ssl_context.wrap_socket(client_socket, server_side=True)
            
            while True:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    break
                
                try:
                    # Parse message
                    message_data = json.loads(data.decode())
                    response = self._process_message(message_data, client_address)
                    
                    # Send response
                    if response:
                        response_json = json.dumps(response)
                        client_socket.send(response_json.encode())
                
                except json.JSONDecodeError:
                    error_response = {"error": "Invalid message format"}
                    client_socket.send(json.dumps(error_response).encode())
                
        except Exception as e:
            print(f"Client handling error: {e}")
        finally:
            client_socket.close()
    
    def _process_message(self, message_data: Dict[str, Any], client_address) -> Dict[str, Any]:
        """Process received message."""
        message_type = message_data.get("type")
        
        if message_type == "handshake":
            return self._handle_handshake(message_data, client_address)
        elif message_type == "encrypted_message":
            return self._handle_encrypted_message(message_data, client_address)
        elif message_type == "ping":
            return {"type": "pong", "timestamp": datetime.now().isoformat()}
        else:
            return {"error": "Unknown message type"}
    
    def _handle_handshake(self, message_data: Dict[str, Any], client_address) -> Dict[str, Any]:
        """Handle client handshake."""
        client_id = message_data.get("client_id")
        client_public_key = message_data.get("public_key")
        
        if not client_id:
            return {"error": "Client ID required"}
        
        # Store client information
        self.trusted_clients[client_id] = {
            "address": client_address,
            "public_key": client_public_key,
            "connected_at": datetime.now().isoformat()
        }
        
        # Send our public key
        response = {
            "type": "handshake_response",
            "server_id": "jarvis_system",
            "status": "connected"
        }
        
        if CRYPTOGRAPHY_AVAILABLE and self.public_key:
            public_key_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
            response["public_key"] = public_key_pem
        
        return response
    
    def _handle_encrypted_message(self, message_data: Dict[str, Any], client_address) -> Dict[str, Any]:
        """Handle encrypted message."""
        decrypted_message = self.decrypt_message(message_data)
        
        if not decrypted_message:
            return {"error": "Failed to decrypt message"}
        
        # Add to message queue
        self.message_queue.append({
            "message": decrypted_message,
            "from": client_address,
            "timestamp": datetime.now().isoformat()
        })
        
        # Process message with handlers
        for handler_name, handler_func in self.message_handlers.items():
            try:
                handler_func(decrypted_message, client_address)
            except Exception as e:
                print(f"Handler {handler_name} error: {e}")
        
        return {"status": "message_received", "message_id": message_data.get("message_id")}
    
    def send_secure_message(self, message: str, client_id: str = None) -> bool:
        """Send encrypted message to client."""
        if client_id and client_id not in self.trusted_clients:
            print(f"Client {client_id} not found in trusted clients")
            return False
        
        # Encrypt message
        encrypted_envelope = self.encrypt_message(message, client_id)
        if not encrypted_envelope:
            return False
        
        # Add signature
        signature = self.sign_message(message)
        if signature:
            encrypted_envelope["signature"] = signature
        
        encrypted_envelope["type"] = "encrypted_message"
        
        # Send to specific client or broadcast
        if client_id:
            return self._send_to_client(encrypted_envelope, client_id)
        else:
            # Broadcast to all clients
            success_count = 0
            for cid in self.trusted_clients.keys():
                if self._send_to_client(encrypted_envelope, cid):
                    success_count += 1
            return success_count > 0
    
    def _send_to_client(self, message_data: Dict[str, Any], client_id: str) -> bool:
        """Send message to specific client."""
        # In a real implementation, this would maintain persistent connections
        # For this demo, we'll simulate successful sending
        try:
            client_info = self.trusted_clients.get(client_id)
            if not client_info:
                return False
            
            # Add to message queue for demonstration
            self.message_queue.append({
                "message": f"Sent to {client_id}: {message_data}",
                "to": client_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            print(f"Failed to send message to {client_id}: {e}")
            return False
    
    def register_message_handler(self, name: str, handler: Callable[[str, Any], None]):
        """Register message handler."""
        self.message_handlers[name] = handler
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status."""
        return {
            "server_running": self.running,
            "port": self.port,
            "trusted_clients": len(self.trusted_clients),
            "messages_processed": len(self.message_queue),
            "encryption_available": CRYPTOGRAPHY_AVAILABLE,
            "ssl_available": SSL_AVAILABLE and self.ssl_context is not None
        }
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages."""
        return list(self.message_queue)[-count:]
    
    def stop_server(self):
        """Stop the secure server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()


# Example usage and testing
if __name__ == "__main__":
    # Initialize secure communicator
    comm = SecureCommunicator(port=8443)
    
    # Test message encryption/decryption
    test_message = "This is a secure test message"
    encrypted = comm.encrypt_message(test_message, "test_client")
    
    if encrypted:
        print("Encrypted message:", encrypted)
        
        decrypted = comm.decrypt_message(encrypted)
        print("Decrypted message:", decrypted)
        print("Encryption/Decryption successful:", decrypted == test_message)
    
    # Test message signing
    signature = comm.sign_message(test_message)
    if signature:
        print("Message signature:", signature[:50] + "...")
        verified = comm.verify_signature(test_message, signature, comm.public_key)
        print("Signature verification:", verified)
    
    # Register a test message handler
    def test_handler(message: str, client_address):
        print(f"Received message from {client_address}: {message}")
    
    comm.register_message_handler("test_handler", test_handler)
    
    # Start secure server
    if comm.start_secure_server():
        print("Secure communication system initialized successfully")
        
        # Get status
        status = comm.get_connection_status()
        print("Connection Status:")
        print(json.dumps(status, indent=2))
        
        # Simulate sending a message
        comm.send_secure_message("Hello from JARVIS secure system", "test_client")
        
        # Show recent messages
        recent = comm.get_recent_messages(5)
        print("Recent Messages:")
        for msg in recent:
            print(f"  {msg['timestamp']}: {msg['message'][:50]}")
    
    else:
        print("Failed to initialize secure communication system")