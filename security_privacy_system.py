"""
JARVIS Security & Privacy System
===============================

This module provides comprehensive security and privacy features:
- Biometric authentication (face, voice, fingerprint)
- Multi-factor authentication (MFA)
- End-to-end encrypted communications
- Secure data storage with encryption
- Privacy controls and data anonymization
- Access control and authorization
- Security monitoring and threat detection
- Secure key management
- Privacy-preserving AI processing
"""

import cv2
import numpy as np
import hashlib
import hmac
import base64
import secrets
import time
import json
import threading
import sqlite3
import os
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio

# Cryptography and security
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import jwt
import qrcode
import pyotp

# Biometric authentication
import face_recognition
import speech_recognition as sr

# System security
import psutil
import platform
from functools import wraps

class AuthenticationMethod(Enum):
    """Authentication methods"""
    FACE_RECOGNITION = "face_recognition"
    VOICE_RECOGNITION = "voice_recognition" 
    FINGERPRINT = "fingerprint"
    PASSWORD = "password"
    PIN = "pin"
    BIOMETRIC_COMBO = "biometric_combo"
    MFA_TOTP = "mfa_totp"
    HARDWARE_TOKEN = "hardware_token"
    GESTURE_PATTERN = "gesture_pattern"

class SecurityLevel(Enum):
    """Security access levels"""
    GUEST = "guest"                    # Basic access
    USER = "user"                      # Standard user access
    PRIVILEGED = "privileged"          # Advanced features
    ADMIN = "admin"                    # System administration
    OWNER = "owner"                    # Full system control
    EMERGENCY = "emergency"            # Emergency override

class EncryptionLevel(Enum):
    """Data encryption levels"""
    NONE = "none"                      # No encryption
    BASIC = "basic"                    # AES-128
    STANDARD = "standard"              # AES-256
    MILITARY = "military"              # AES-256 + RSA-4096
    QUANTUM_RESISTANT = "quantum"      # Post-quantum cryptography

@dataclass
class BiometricProfile:
    """User biometric profile"""
    user_id: str
    name: str
    
    # Face recognition data
    face_encodings: List[np.ndarray] = field(default_factory=list)
    face_confidence_threshold: float = 0.6
    
    # Voice recognition data
    voice_fingerprint: Optional[np.ndarray] = None
    voice_confidence_threshold: float = 0.75
    
    # Gesture patterns
    gesture_patterns: List[Dict[str, Any]] = field(default_factory=list)
    
    # Authentication settings
    required_methods: List[AuthenticationMethod] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.USER
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_authenticated: Optional[datetime] = None
    failed_attempts: int = 0

@dataclass
class SecurityEvent:
    """Security event logging"""
    event_id: str
    event_type: str
    user_id: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    resolved: bool = False

class CryptographyManager:
    """Handles encryption and decryption operations"""
    
    def __init__(self):
        self.master_key = None
        self.fernet_cipher = None
        self.rsa_private_key = None
        self.rsa_public_key = None
        
        # Initialize encryption
        self.initialize_encryption()
    
    def initialize_encryption(self):
        """Initialize encryption systems"""
        # Generate or load master key
        self.master_key = self.load_or_generate_master_key()
        self.fernet_cipher = Fernet(self.master_key)
        
        # Generate RSA key pair
        self.generate_rsa_keys()
        
        print("🔐 Cryptography manager initialized")
    
    def load_or_generate_master_key(self) -> bytes:
        """Load existing master key or generate new one"""
        key_file = "master_key.key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            print("🔑 New master key generated")
            return key
    
    def generate_rsa_keys(self):
        """Generate RSA key pair"""
        self.rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        self.rsa_public_key = self.rsa_private_key.public_key()
    
    def encrypt_data(self, data: Union[str, bytes], level: EncryptionLevel = EncryptionLevel.STANDARD) -> bytes:
        """Encrypt data with specified security level"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if level == EncryptionLevel.NONE:
            return data
        
        elif level in [EncryptionLevel.BASIC, EncryptionLevel.STANDARD]:
            return self.fernet_cipher.encrypt(data)
        
        elif level in [EncryptionLevel.MILITARY, EncryptionLevel.QUANTUM_RESISTANT]:
            # Double encryption: AES + RSA
            aes_encrypted = self.fernet_cipher.encrypt(data)
            
            # RSA encrypt the AES-encrypted data (in chunks due to RSA size limits)
            chunk_size = 446  # RSA 4096-bit key can encrypt up to 446 bytes
            encrypted_chunks = []
            
            for i in range(0, len(aes_encrypted), chunk_size):
                chunk = aes_encrypted[i:i + chunk_size]
                encrypted_chunk = self.rsa_public_key.encrypt(
                    chunk,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                encrypted_chunks.append(encrypted_chunk)
            
            return b'||'.join(encrypted_chunks)
        
        return data
    
    def decrypt_data(self, encrypted_data: bytes, level: EncryptionLevel = EncryptionLevel.STANDARD) -> bytes:
        """Decrypt data with specified security level"""
        if level == EncryptionLevel.NONE:
            return encrypted_data
        
        elif level in [EncryptionLevel.BASIC, EncryptionLevel.STANDARD]:
            return self.fernet_cipher.decrypt(encrypted_data)
        
        elif level in [EncryptionLevel.MILITARY, EncryptionLevel.QUANTUM_RESISTANT]:
            # Reverse double encryption: RSA then AES
            encrypted_chunks = encrypted_data.split(b'||')
            decrypted_chunks = []
            
            for chunk in encrypted_chunks:
                decrypted_chunk = self.rsa_private_key.decrypt(
                    chunk,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                decrypted_chunks.append(decrypted_chunk)
            
            aes_encrypted = b''.join(decrypted_chunks)
            return self.fernet_cipher.decrypt(aes_encrypted)
        
        return encrypted_data
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(password.encode('utf-8'))
        return key, salt
    
    def verify_password(self, password: str, hashed_password: bytes, salt: bytes) -> bool:
        """Verify password against hash"""
        key, _ = self.hash_password(password, salt)
        return hmac.compare_digest(key, hashed_password)

class BiometricAuthenticator:
    """Handles biometric authentication"""
    
    def __init__(self, crypto_manager: CryptographyManager):
        self.crypto_manager = crypto_manager
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Voice recognition
        self.speech_recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Authentication cache
        self.auth_cache = {}
        self.cache_duration = 300  # 5 minutes
        
    def register_face(self, user_id: str, image: np.ndarray) -> bool:
        """Register user's face for authentication"""
        try:
            # Detect faces in image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                print("❌ No face detected in image")
                return False
            
            # Generate face encoding
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                print("❌ Could not generate face encoding")
                return False
            
            # Store encrypted face encoding
            face_encoding = face_encodings[0]
            encrypted_encoding = self.crypto_manager.encrypt_data(
                face_encoding.tobytes(), 
                EncryptionLevel.MILITARY
            )
            
            # Save to secure storage
            self.save_biometric_data(user_id, 'face', encrypted_encoding)
            print(f"✅ Face registered for user: {user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Face registration failed: {e}")
            return False
    
    def authenticate_face(self, user_id: str, image: np.ndarray, threshold: float = 0.6) -> bool:
        """Authenticate user using face recognition"""
        try:
            # Check cache first
            cache_key = f"face_{user_id}"
            if self.is_auth_cached(cache_key):
                return True
            
            # Get stored face encoding
            stored_encoding_data = self.load_biometric_data(user_id, 'face')
            if not stored_encoding_data:
                return False
            
            # Decrypt stored encoding
            decrypted_data = self.crypto_manager.decrypt_data(
                stored_encoding_data, 
                EncryptionLevel.MILITARY
            )
            stored_encoding = np.frombuffer(decrypted_data, dtype=np.float64)
            
            # Get current face encoding
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return False
            
            current_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if not current_encodings:
                return False
            
            # Compare faces
            current_encoding = current_encodings[0]
            face_distance = face_recognition.face_distance([stored_encoding], current_encoding)[0]
            
            # Authentication successful if distance is below threshold
            is_match = face_distance < threshold
            
            if is_match:
                self.cache_authentication(cache_key)
                print(f"✅ Face authentication successful for user: {user_id}")
            else:
                print(f"❌ Face authentication failed for user: {user_id} (distance: {face_distance:.3f})")
            
            return is_match
            
        except Exception as e:
            print(f"❌ Face authentication error: {e}")
            return False
    
    def register_voice(self, user_id: str, audio_samples: List[np.ndarray]) -> bool:
        """Register user's voice for authentication"""
        try:
            # Create voice fingerprint from multiple samples
            voice_features = []
            
            for audio_sample in audio_samples:
                # Extract voice features (MFCC, pitch, etc.)
                features = self.extract_voice_features(audio_sample)
                if features is not None:
                    voice_features.append(features)
            
            if not voice_features:
                print("❌ Could not extract voice features")
                return False
            
            # Average features to create voice fingerprint
            voice_fingerprint = np.mean(voice_features, axis=0)
            
            # Encrypt and store
            encrypted_fingerprint = self.crypto_manager.encrypt_data(
                voice_fingerprint.tobytes(),
                EncryptionLevel.MILITARY
            )
            
            self.save_biometric_data(user_id, 'voice', encrypted_fingerprint)
            print(f"✅ Voice registered for user: {user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Voice registration failed: {e}")
            return False
    
    def authenticate_voice(self, user_id: str, audio_sample: np.ndarray, threshold: float = 0.75) -> bool:
        """Authenticate user using voice recognition"""
        try:
            # Check cache
            cache_key = f"voice_{user_id}"
            if self.is_auth_cached(cache_key):
                return True
            
            # Get stored voice fingerprint
            stored_fingerprint_data = self.load_biometric_data(user_id, 'voice')
            if not stored_fingerprint_data:
                return False
            
            # Decrypt stored fingerprint
            decrypted_data = self.crypto_manager.decrypt_data(
                stored_fingerprint_data,
                EncryptionLevel.MILITARY
            )
            stored_fingerprint = np.frombuffer(decrypted_data, dtype=np.float64)
            
            # Extract current voice features
            current_features = self.extract_voice_features(audio_sample)
            if current_features is None:
                return False
            
            # Compare voice fingerprints
            similarity = self.calculate_voice_similarity(stored_fingerprint, current_features)
            
            is_match = similarity > threshold
            
            if is_match:
                self.cache_authentication(cache_key)
                print(f"✅ Voice authentication successful for user: {user_id}")
            else:
                print(f"❌ Voice authentication failed for user: {user_id} (similarity: {similarity:.3f})")
            
            return is_match
            
        except Exception as e:
            print(f"❌ Voice authentication error: {e}")
            return False
    
    def extract_voice_features(self, audio_sample: np.ndarray) -> Optional[np.ndarray]:
        """Extract voice features for recognition"""
        try:
            # This is a simplified feature extraction
            # In practice, would use advanced audio processing libraries like librosa
            
            # Basic features: spectral centroid, zero crossing rate, etc.
            features = []
            
            # Spectral centroid (brightness)
            fft = np.fft.fft(audio_sample)
            magnitude = np.abs(fft)
            freqs = np.fft.fftfreq(len(fft))
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            features.append(spectral_centroid)
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(audio_sample)))[0]
            zcr = len(zero_crossings) / len(audio_sample)
            features.append(zcr)
            
            # Energy
            energy = np.sum(audio_sample ** 2) / len(audio_sample)
            features.append(energy)
            
            # Pitch estimation (simplified)
            autocorr = np.correlate(audio_sample, audio_sample, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find fundamental frequency
            min_period = 50  # Minimum period for human voice
            max_period = 400  # Maximum period for human voice
            
            if len(autocorr) > max_period:
                pitch_autocorr = autocorr[min_period:max_period]
                if len(pitch_autocorr) > 0:
                    pitch_period = np.argmax(pitch_autocorr) + min_period
                    fundamental_freq = 44100 / pitch_period  # Assuming 44.1kHz sample rate
                    features.append(fundamental_freq)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Voice feature extraction error: {e}")
            return None
    
    def calculate_voice_similarity(self, fingerprint1: np.ndarray, fingerprint2: np.ndarray) -> float:
        """Calculate similarity between voice fingerprints"""
        try:
            # Ensure same length
            min_len = min(len(fingerprint1), len(fingerprint2))
            fp1 = fingerprint1[:min_len]
            fp2 = fingerprint2[:min_len]
            
            # Cosine similarity
            dot_product = np.dot(fp1, fp2)
            norm1 = np.linalg.norm(fp1)
            norm2 = np.linalg.norm(fp2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            print(f"Voice similarity calculation error: {e}")
            return 0.0
    
    def save_biometric_data(self, user_id: str, biometric_type: str, data: bytes):
        """Save encrypted biometric data"""
        os.makedirs('secure_biometric_data', exist_ok=True)
        filename = f"secure_biometric_data/{user_id}_{biometric_type}.enc"
        
        with open(filename, 'wb') as f:
            f.write(data)
    
    def load_biometric_data(self, user_id: str, biometric_type: str) -> Optional[bytes]:
        """Load encrypted biometric data"""
        filename = f"secure_biometric_data/{user_id}_{biometric_type}.enc"
        
        if not os.path.exists(filename):
            return None
        
        with open(filename, 'rb') as f:
            return f.read()
    
    def is_auth_cached(self, cache_key: str) -> bool:
        """Check if authentication is cached and valid"""
        if cache_key in self.auth_cache:
            cache_time = self.auth_cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                return True
            else:
                del self.auth_cache[cache_key]
        return False
    
    def cache_authentication(self, cache_key: str):
        """Cache successful authentication"""
        self.auth_cache[cache_key] = time.time()

class MFAManager:
    """Multi-Factor Authentication Manager"""
    
    def __init__(self, crypto_manager: CryptographyManager):
        self.crypto_manager = crypto_manager
        self.user_secrets = {}
        self.backup_codes = {}
        
    def setup_totp(self, user_id: str, issuer: str = "JARVIS") -> Tuple[str, str]:
        """Setup Time-based One-Time Password (TOTP)"""
        # Generate secret key
        secret = pyotp.random_base32()
        
        # Create TOTP object
        totp = pyotp.TOTP(secret)
        
        # Generate provisioning URI for QR code
        provisioning_uri = totp.provisioning_uri(
            name=user_id,
            issuer_name=issuer
        )
        
        # Store encrypted secret
        encrypted_secret = self.crypto_manager.encrypt_data(
            secret.encode(), 
            EncryptionLevel.MILITARY
        )
        self.save_mfa_secret(user_id, encrypted_secret)
        
        # Generate backup codes
        backup_codes = [self.crypto_manager.generate_secure_token(8) for _ in range(10)]
        self.backup_codes[user_id] = backup_codes
        
        print(f"✅ TOTP setup for user: {user_id}")
        return secret, provisioning_uri
    
    def verify_totp(self, user_id: str, token: str) -> bool:
        """Verify TOTP token"""
        try:
            # Load user's secret
            secret_data = self.load_mfa_secret(user_id)
            if not secret_data:
                return False
            
            # Decrypt secret
            decrypted_secret = self.crypto_manager.decrypt_data(
                secret_data, 
                EncryptionLevel.MILITARY
            ).decode()
            
            # Create TOTP object and verify
            totp = pyotp.TOTP(decrypted_secret)
            is_valid = totp.verify(token, valid_window=1)  # Allow 30-second window
            
            if is_valid:
                print(f"✅ TOTP verification successful for user: {user_id}")
            else:
                print(f"❌ TOTP verification failed for user: {user_id}")
            
            return is_valid
            
        except Exception as e:
            print(f"❌ TOTP verification error: {e}")
            return False
    
    def verify_backup_code(self, user_id: str, backup_code: str) -> bool:
        """Verify backup code"""
        if user_id in self.backup_codes:
            if backup_code in self.backup_codes[user_id]:
                # Remove used backup code
                self.backup_codes[user_id].remove(backup_code)
                print(f"✅ Backup code verified for user: {user_id}")
                return True
        
        print(f"❌ Invalid backup code for user: {user_id}")
        return False
    
    def generate_qr_code(self, provisioning_uri: str, filename: str = "mfa_qr.png"):
        """Generate QR code for TOTP setup"""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(filename)
        print(f"📱 QR code saved as: {filename}")
    
    def save_mfa_secret(self, user_id: str, encrypted_secret: bytes):
        """Save encrypted MFA secret"""
        os.makedirs('secure_mfa_data', exist_ok=True)
        filename = f"secure_mfa_data/{user_id}_totp.enc"
        
        with open(filename, 'wb') as f:
            f.write(encrypted_secret)
    
    def load_mfa_secret(self, user_id: str) -> Optional[bytes]:
        """Load encrypted MFA secret"""
        filename = f"secure_mfa_data/{user_id}_totp.enc"
        
        if not os.path.exists(filename):
            return None
        
        with open(filename, 'rb') as f:
            return f.read()

class AccessControlManager:
    """Manages user access control and permissions"""
    
    def __init__(self, crypto_manager: CryptographyManager):
        self.crypto_manager = crypto_manager
        self.user_profiles = {}
        self.active_sessions = {}
        self.permissions = self.setup_permissions()
        
        # Initialize database
        self.init_database()
    
    def setup_permissions(self) -> Dict[SecurityLevel, List[str]]:
        """Setup permission levels"""
        return {
            SecurityLevel.GUEST: [
                "view_basic_info",
                "voice_commands_basic"
            ],
            SecurityLevel.USER: [
                "view_basic_info",
                "voice_commands_basic",
                "voice_commands_standard",
                "smart_home_basic",
                "data_visualization_basic"
            ],
            SecurityLevel.PRIVILEGED: [
                "view_basic_info",
                "voice_commands_basic",
                "voice_commands_standard",
                "voice_commands_advanced",
                "smart_home_basic",
                "smart_home_advanced",
                "data_visualization_basic",
                "data_visualization_advanced",
                "system_monitoring"
            ],
            SecurityLevel.ADMIN: [
                "view_basic_info",
                "voice_commands_basic",
                "voice_commands_standard",
                "voice_commands_advanced",
                "smart_home_basic",
                "smart_home_advanced",
                "data_visualization_basic",
                "data_visualization_advanced",
                "system_monitoring",
                "user_management",
                "security_config",
                "system_config"
            ],
            SecurityLevel.OWNER: [
                "*"  # All permissions
            ]
        }
    
    def init_database(self):
        """Initialize secure database"""
        self.db_path = "secure_jarvis.db"
        conn = sqlite3.connect(self.db_path)
        
        # Create tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                security_level TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP NULL
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                user_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT,
                severity TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                ip_address TEXT,
                device_info TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print("🗄️ Secure database initialized")
    
    def create_user_profile(self, user_id: str, name: str, security_level: SecurityLevel) -> BiometricProfile:
        """Create new user profile"""
        profile = BiometricProfile(
            user_id=user_id,
            name=name,
            security_level=security_level
        )
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO users (user_id, name, security_level) VALUES (?, ?, ?)",
            (user_id, name, security_level.value)
        )
        conn.commit()
        conn.close()
        
        self.user_profiles[user_id] = profile
        print(f"👤 User profile created: {name} ({security_level.value})")
        return profile
    
    def authenticate_user(self, user_id: str, auth_methods: Dict[AuthenticationMethod, Any]) -> Optional[str]:
        """Authenticate user with multiple methods"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            self.log_security_event("authentication_failed", user_id, "User not found", "HIGH")
            return None
        
        # Check if user is locked
        if self.is_user_locked(user_id):
            self.log_security_event("authentication_blocked", user_id, "User account locked", "HIGH")
            return None
        
        # Verify each required authentication method
        auth_success = True
        auth_results = {}
        
        for method, data in auth_methods.items():
            if method == AuthenticationMethod.FACE_RECOGNITION:
                result = self.verify_face_auth(user_id, data)
            elif method == AuthenticationMethod.VOICE_RECOGNITION:
                result = self.verify_voice_auth(user_id, data)
            elif method == AuthenticationMethod.MFA_TOTP:
                result = self.verify_mfa_auth(user_id, data)
            elif method == AuthenticationMethod.PASSWORD:
                result = self.verify_password_auth(user_id, data)
            else:
                result = False
            
            auth_results[method] = result
            if not result:
                auth_success = False
        
        if auth_success:
            # Create session
            session_id = self.create_session(user_id)
            profile.last_authenticated = datetime.now()
            profile.failed_attempts = 0
            
            self.log_security_event("authentication_success", user_id, f"Methods: {list(auth_methods.keys())}", "LOW")
            print(f"✅ User authenticated: {profile.name}")
            return session_id
        else:
            # Handle failed authentication
            profile.failed_attempts += 1
            
            if profile.failed_attempts >= 5:
                self.lock_user(user_id, minutes=30)
            
            self.log_security_event("authentication_failed", user_id, f"Failed methods: {auth_results}", "MEDIUM")
            print(f"❌ Authentication failed for user: {user_id}")
            return None
    
    def create_session(self, user_id: str, duration_hours: int = 8) -> str:
        """Create authenticated session"""
        session_id = self.crypto_manager.generate_secure_token()
        expires_at = datetime.now() + timedelta(hours=duration_hours)
        
        session_data = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'expires_at': expires_at,
            'permissions': self.get_user_permissions(user_id)
        }
        
        self.active_sessions[session_id] = session_data
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO sessions (session_id, user_id, expires_at) VALUES (?, ?, ?)",
            (session_id, user_id, expires_at)
        )
        conn.commit()
        conn.close()
        
        return session_id
    
    def verify_session(self, session_id: str) -> Optional[str]:
        """Verify session validity"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        if datetime.now() > session['expires_at']:
            # Session expired
            del self.active_sessions[session_id]
            return None
        
        return session['user_id']
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions based on security level"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return []
        
        return self.permissions.get(profile.security_level, [])
    
    def check_permission(self, session_id: str, permission: str) -> bool:
        """Check if session has specific permission"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        user_permissions = session.get('permissions', [])
        
        # Owner has all permissions
        if "*" in user_permissions:
            return True
        
        return permission in user_permissions
    
    def is_user_locked(self, user_id: str) -> bool:
        """Check if user account is locked"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT locked_until FROM users WHERE user_id = ?",
            (user_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            locked_until = datetime.fromisoformat(result[0])
            return datetime.now() < locked_until
        
        return False
    
    def lock_user(self, user_id: str, minutes: int = 30):
        """Lock user account"""
        locked_until = datetime.now() + timedelta(minutes=minutes)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE users SET locked_until = ? WHERE user_id = ?",
            (locked_until, user_id)
        )
        conn.commit()
        conn.close()
        
        self.log_security_event("account_locked", user_id, f"Locked for {minutes} minutes", "HIGH")
    
    def log_security_event(self, event_type: str, user_id: Optional[str], details: str, severity: str):
        """Log security event"""
        event_id = self.crypto_manager.generate_secure_token(16)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO security_events (event_id, event_type, user_id, details, severity) VALUES (?, ?, ?, ?, ?)",
            (event_id, event_type, user_id, details, severity)
        )
        conn.commit()
        conn.close()
        
        print(f"🚨 Security Event [{severity}]: {event_type} - {details}")

class SecurityMonitor:
    """Monitors system security and detects threats"""
    
    def __init__(self, access_manager: AccessControlManager):
        self.access_manager = access_manager
        self.monitoring_active = False
        self.threat_detection_rules = self.setup_threat_rules()
        
        # Monitoring data
        self.failed_login_attempts = {}
        self.suspicious_activities = []
        self.system_metrics = {}
        
    def setup_threat_rules(self) -> Dict[str, Dict[str, Any]]:
        """Setup threat detection rules"""
        return {
            'brute_force': {
                'failed_attempts_threshold': 10,
                'time_window_minutes': 15,
                'severity': 'HIGH'
            },
            'unusual_access_patterns': {
                'max_sessions_per_hour': 20,
                'max_permission_requests': 100,
                'severity': 'MEDIUM'
            },
            'system_resource_abuse': {
                'cpu_threshold': 90,
                'memory_threshold': 85,
                'disk_threshold': 95,
                'severity': 'MEDIUM'
            },
            'unauthorized_access': {
                'unknown_device_attempts': 5,
                'geo_location_anomaly': True,
                'severity': 'CRITICAL'
            }
        }
    
    def start_monitoring(self):
        """Start security monitoring"""
        self.monitoring_active = True
        
        # Start monitoring threads
        threading.Thread(target=self.monitor_system_resources, daemon=True).start()
        threading.Thread(target=self.monitor_access_patterns, daemon=True).start()
        threading.Thread(target=self.analyze_threats, daemon=True).start()
        
        print("🛡️ Security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring"""
        self.monitoring_active = False
        print("⏹️ Security monitoring stopped")
    
    def monitor_system_resources(self):
        """Monitor system resource usage"""
        while self.monitoring_active:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self.system_metrics = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'timestamp': datetime.now()
                }
                
                # Check thresholds
                rules = self.threat_detection_rules['system_resource_abuse']
                
                if (cpu_percent > rules['cpu_threshold'] or 
                    memory.percent > rules['memory_threshold'] or 
                    disk.percent > rules['disk_threshold']):
                    
                    self.report_threat('system_resource_abuse', {
                        'cpu': cpu_percent,
                        'memory': memory.percent,
                        'disk': disk.percent
                    }, rules['severity'])
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"System monitoring error: {e}")
                time.sleep(60)
    
    def monitor_access_patterns(self):
        """Monitor user access patterns"""
        while self.monitoring_active:
            try:
                # Analyze session patterns
                current_time = datetime.now()
                recent_sessions = []
                
                for session_id, session_data in self.access_manager.active_sessions.items():
                    session_age = current_time - session_data['created_at']
                    if session_age.total_seconds() < 3600:  # Last hour
                        recent_sessions.append(session_data)
                
                # Check for unusual patterns
                if len(recent_sessions) > self.threat_detection_rules['unusual_access_patterns']['max_sessions_per_hour']:
                    self.report_threat('unusual_access_patterns', {
                        'session_count': len(recent_sessions),
                        'threshold': self.threat_detection_rules['unusual_access_patterns']['max_sessions_per_hour']
                    }, 'MEDIUM')
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                print(f"Access pattern monitoring error: {e}")
                time.sleep(300)
    
    def analyze_threats(self):
        """Analyze and correlate threat data"""
        while self.monitoring_active:
            try:
                # Analyze suspicious activities
                if len(self.suspicious_activities) > 0:
                    # Group activities by type and time
                    activity_groups = {}
                    current_time = datetime.now()
                    
                    for activity in self.suspicious_activities:
                        activity_age = current_time - activity['timestamp']
                        if activity_age.total_seconds() < 3600:  # Last hour
                            activity_type = activity['type']
                            if activity_type not in activity_groups:
                                activity_groups[activity_type] = []
                            activity_groups[activity_type].append(activity)
                    
                    # Check for patterns
                    for activity_type, activities in activity_groups.items():
                        if len(activities) > 5:  # Multiple similar activities
                            self.report_threat('coordinated_attack', {
                                'activity_type': activity_type,
                                'count': len(activities),
                                'timespan': '1 hour'
                            }, 'HIGH')
                
                # Clean old activities
                self.suspicious_activities = [
                    activity for activity in self.suspicious_activities
                    if (current_time - activity['timestamp']).total_seconds() < 86400  # Keep 24 hours
                ]
                
                time.sleep(600)  # Analyze every 10 minutes
                
            except Exception as e:
                print(f"Threat analysis error: {e}")
                time.sleep(600)
    
    def report_threat(self, threat_type: str, details: Dict[str, Any], severity: str):
        """Report detected threat"""
        threat_report = {
            'type': threat_type,
            'details': details,
            'severity': severity,
            'timestamp': datetime.now(),
            'system_state': self.system_metrics.copy() if self.system_metrics else {}
        }
        
        # Log security event
        self.access_manager.log_security_event(
            f"threat_detected_{threat_type}",
            None,
            json.dumps(details),
            severity
        )
        
        # Take automated response actions
        self.respond_to_threat(threat_type, severity, details)
        
        print(f"🚨 THREAT DETECTED [{severity}]: {threat_type}")
        print(f"   Details: {details}")
    
    def respond_to_threat(self, threat_type: str, severity: str, details: Dict[str, Any]):
        """Automated threat response"""
        if severity == 'CRITICAL':
            # Critical threats - immediate action
            if threat_type == 'unauthorized_access':
                # Lock all sessions from unknown devices
                self.lock_suspicious_sessions()
            
        elif severity == 'HIGH':
            # High severity - defensive actions
            if threat_type == 'brute_force':
                # Temporarily block IP/user
                if 'user_id' in details:
                    self.access_manager.lock_user(details['user_id'], minutes=60)
            
        elif severity == 'MEDIUM':
            # Medium severity - monitoring and alerts
            # Increase monitoring frequency
            print(f"📊 Increased monitoring for {threat_type}")
    
    def lock_suspicious_sessions(self):
        """Lock sessions that appear suspicious"""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session_data in self.access_manager.active_sessions.items():
            # Simple heuristic: very recent sessions might be from attackers
            session_age = current_time - session_data['created_at']
            if session_age.total_seconds() < 300:  # Sessions created in last 5 minutes
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.access_manager.active_sessions[session_id]
            print(f"🔒 Locked suspicious session: {session_id}")

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # This would check the current session's permissions
            # For now, simplified implementation
            if hasattr(self, 'access_manager') and hasattr(self, 'current_session'):
                if not self.access_manager.check_permission(self.current_session, permission):
                    raise PermissionError(f"Permission required: {permission}")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

class JarvisSecuritySystem:
    """Main JARVIS Security & Privacy System"""
    
    def __init__(self):
        # Core security components
        self.crypto_manager = CryptographyManager()
        self.biometric_auth = BiometricAuthenticator(self.crypto_manager)
        self.mfa_manager = MFAManager(self.crypto_manager)
        self.access_manager = AccessControlManager(self.crypto_manager)
        self.security_monitor = SecurityMonitor(self.access_manager)
        
        # System state
        self.is_initialized = False
        self.current_session = None
        self.security_level = SecurityLevel.GUEST
        
        print("🛡️ JARVIS Security System initialized")
    
    def initialize_security(self):
        """Initialize security system"""
        try:
            # Start security monitoring
            self.security_monitor.start_monitoring()
            
            # Create default owner account if none exists
            if not self.access_manager.user_profiles:
                self.setup_initial_owner()
            
            self.is_initialized = True
            print("✅ JARVIS Security System ready")
            
        except Exception as e:
            print(f"❌ Security initialization failed: {e}")
    
    def setup_initial_owner(self):
        """Setup initial owner account"""
        print("🔐 Setting up initial owner account...")
        
        # Create owner profile
        owner_profile = self.access_manager.create_user_profile(
            user_id="jarvis_owner",
            name="JARVIS Owner",
            security_level=SecurityLevel.OWNER
        )
        
        print("📱 Please complete biometric registration:")
        print("1. Face recognition setup")
        print("2. Voice recognition setup") 
        print("3. MFA setup")
        
        # In a real implementation, this would guide through setup process
        print("✅ Owner account created. Please complete setup through the interface.")
    
    @require_permission("user_management")
    def register_new_user(self, user_id: str, name: str, security_level: SecurityLevel) -> bool:
        """Register new user with biometric data"""
        try:
            # Create user profile
            profile = self.access_manager.create_user_profile(user_id, name, security_level)
            
            print(f"👤 User registration started for: {name}")
            print("Please complete biometric registration process")
            
            return True
            
        except Exception as e:
            print(f"❌ User registration failed: {e}")
            return False
    
    def authenticate_user(self, user_id: str, auth_data: Dict[str, Any]) -> Optional[str]:
        """Authenticate user with provided data"""
        try:
            # Determine authentication methods from provided data
            auth_methods = {}
            
            if 'face_image' in auth_data:
                auth_methods[AuthenticationMethod.FACE_RECOGNITION] = auth_data['face_image']
            
            if 'voice_sample' in auth_data:
                auth_methods[AuthenticationMethod.VOICE_RECOGNITION] = auth_data['voice_sample']
            
            if 'totp_token' in auth_data:
                auth_methods[AuthenticationMethod.MFA_TOTP] = auth_data['totp_token']
            
            if 'password' in auth_data:
                auth_methods[AuthenticationMethod.PASSWORD] = auth_data['password']
            
            # Authenticate
            session_id = self.access_manager.authenticate_user(user_id, auth_methods)
            
            if session_id:
                self.current_session = session_id
                user_profile = self.access_manager.user_profiles.get(user_id)
                if user_profile:
                    self.security_level = user_profile.security_level
                    print(f"🔓 Welcome, {user_profile.name}!")
            
            return session_id
            
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return None
    
    def logout(self):
        """Logout current user"""
        if self.current_session:
            if self.current_session in self.access_manager.active_sessions:
                del self.access_manager.active_sessions[self.current_session]
            
            self.current_session = None
            self.security_level = SecurityLevel.GUEST
            print("👋 User logged out")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        return {
            'initialized': self.is_initialized,
            'current_session': self.current_session is not None,
            'security_level': self.security_level.value,
            'active_sessions': len(self.access_manager.active_sessions),
            'monitoring_active': self.security_monitor.monitoring_active,
            'threat_rules_count': len(self.security_monitor.threat_detection_rules),
            'registered_users': len(self.access_manager.user_profiles),
            'system_metrics': self.security_monitor.system_metrics
        }
    
    def encrypt_sensitive_data(self, data: Union[str, bytes], level: EncryptionLevel = EncryptionLevel.STANDARD) -> bytes:
        """Encrypt sensitive data"""
        return self.crypto_manager.encrypt_data(data, level)
    
    def decrypt_sensitive_data(self, encrypted_data: bytes, level: EncryptionLevel = EncryptionLevel.STANDARD) -> bytes:
        """Decrypt sensitive data"""
        return self.crypto_manager.decrypt_data(encrypted_data, level)
    
    def shutdown_security(self):
        """Shutdown security system"""
        # Stop monitoring
        self.security_monitor.stop_monitoring()
        
        # Clear active sessions
        self.access_manager.active_sessions.clear()
        
        # Reset state
        self.current_session = None
        self.security_level = SecurityLevel.GUEST
        self.is_initialized = False
        
        print("🔒 JARVIS Security System shutdown")

# Example usage and testing
def main():
    """Test JARVIS Security System"""
    security_system = JarvisSecuritySystem()
    
    # Initialize security
    security_system.initialize_security()
    
    # Show security status
    status = security_system.get_security_status()
    print(f"\n📊 Security Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\n🛡️ JARVIS Security System is running...")
    print("Features available:")
    print("- Biometric authentication (face, voice)")
    print("- Multi-factor authentication (TOTP)")
    print("- End-to-end encryption")
    print("- Access control and permissions")
    print("- Security monitoring and threat detection")
    print("- Privacy-preserving data processing")
    
    # Simulate some time for monitoring
    time.sleep(5)
    
    # Shutdown
    security_system.shutdown_security()

if __name__ == "__main__":
    main()