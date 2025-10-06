# JARVIS User Profiles & Personalization Module

## Overview

The User Profiles module is the personalization core of the JARVIS AI assistant, providing comprehensive user management, behavioral learning, biometric authentication, and adaptive personalization. It creates intelligent user experiences that evolve and adapt based on individual preferences, behaviors, and interactions.

## Features

### 🧠 Advanced User Profiling
- **Comprehensive Profile Management**: Complete user profiles with personal info, preferences, goals, and behavioral patterns
- **AI-Powered Learning**: Machine learning algorithms that adapt to user behavior over time
- **Behavioral Analysis**: Deep analysis of interaction patterns, usage habits, and preferences
- **Contextual Adaptation**: Intelligent responses based on user context and situation

### 🔐 Multi-Modal Biometric Authentication
- **Face Recognition**: Advanced face enrollment and identification
- **Voice Authentication**: Voice print analysis and verification
- **Behavioral Biometrics**: Typing patterns, mouse movement, and interaction behavior analysis
- **Security Features**: Failed attempt tracking, user lockout, and encryption

### 🎨 Comprehensive Preference Management
- **Visual Customization**: Themes, colors, fonts, animations, and UI preferences
- **Interaction Preferences**: Communication style, response length, confirmation levels
- **Accessibility Support**: High contrast, large text, reduced motion, screen reader compatibility
- **Performance Tuning**: Resource usage, optimization settings, and hardware preferences

### 🔄 Adaptive Intelligence
- **Response Personalization**: AI-powered response adaptation based on user patterns
- **Feature Recommendations**: Intelligent suggestions for new features and optimizations
- **Usage Analytics**: Detailed insights into user behavior and engagement
- **Privacy Controls**: Granular privacy settings and data protection options

## Architecture

```
UserProfileManager (Core)
├── User Profile Storage
│   ├── Personal Information
│   ├── Preferences & Settings
│   ├── Behavioral Patterns
│   └── Usage Statistics
├── Learning Systems
│   ├── BehaviorAnalyzer
│   ├── PreferenceLearner
│   ├── ActivityTracker
│   └── RecommendationEngine
├── BiometricAuthenticator
│   ├── Face Recognition
│   ├── Voice Authentication
│   └── Behavioral Patterns
├── PreferenceManager
│   ├── Theme Management
│   ├── Setting Validation
│   └── Customization Tools
└── PrivacyManager
    ├── Data Encryption
    ├── Anonymization
    └── Access Controls
```

## Installation

### Prerequisites
```bash
pip install scikit-learn numpy opencv-python face-recognition cryptography
```

### Optional Dependencies
```bash
# For advanced voice processing
pip install librosa soundfile

# For enhanced ML capabilities
pip install tensorflow keras

# For biometric security
pip install cryptography pyotp
```

## Quick Start

### Basic User Profile Creation
```python
from user_profiles import UserProfileManager

# Initialize manager
profile_manager = UserProfileManager()

# Create comprehensive user profile
user_info = {
    "id": "user123",
    "name": "John Doe",
    "age": 30,
    "language": "en",
    "communication_style": "professional",
    "interests": ["technology", "productivity", "music"],
    "goals": ["automation", "efficiency"],
    "learning_enabled": True
}

profile = profile_manager.create_profile(user_info)
print(f"Created profile for {profile['personal_info']['name']}")
```

### Session Management and Learning
```python
# Start interaction session
session_id = profile_manager.start_session("user123", {
    "device": "desktop",
    "location": "office",
    "time_of_day": "morning"
})

# Record user interactions
interaction_data = {
    "type": "voice_command",
    "command": "turn on office lights",
    "success": True,
    "response_time": 0.8,
    "satisfaction": 0.9,
    "context": {"room": "office", "lighting_level": "dim"}
}

profile_manager.record_interaction(session_id, interaction_data)

# End session (triggers learning updates)
profile_manager.end_session(session_id)
```

### Personalized Responses
```python
# Get personalized response
base_response = "I've turned on the lights for you."
personalized_response = profile_manager.get_personalized_response(
    "user123", 
    base_response,
    context={"time": "morning", "mood": "productive"}
)

print(personalized_response)
# Output might be: "Good morning! I've turned on the office lights to help you start your productive day."
```

### Biometric Authentication
```python
from biometric_auth import BiometricAuthenticator
import cv2

# Initialize authenticator
authenticator = BiometricAuthenticator()

# Enroll user biometrics
face_image = cv2.imread("user_photo.jpg")
voice_sample = load_audio("user_voice.wav")

enrollment_result = authenticator.enroll_user_biometrics(
    "user123",
    face_image=face_image,
    voice_sample=voice_sample
)

# Authenticate user
auth_result = authenticator.authenticate_user(
    "user123",
    face_image=current_face_image,
    voice_sample=current_voice_sample
)

if auth_result["authenticated"]:
    print(f"User authenticated with {auth_result['overall_confidence']:.2f} confidence")
else:
    print(f"Authentication failed: {auth_result['reason']}")
```

### Preference Management
```python
from preference_manager import PreferenceManager

# Initialize preference manager
pref_manager = PreferenceManager()

# Set user preferences
preferences = {
    "appearance.theme": "dark",
    "appearance.color_scheme": "blue",
    "voice.speech_rate": 1.2,
    "voice.volume": 0.8,
    "privacy.data_collection_level": "balanced",
    "accessibility.large_text": True
}

pref_manager.bulk_update_preferences("user123", preferences)

# Apply theme
pref_manager.apply_theme("user123", "iron_man")

# Get specific preference
theme = pref_manager.get_preference_value("user123", "appearance.theme")
print(f"Current theme: {theme}")
```

## Advanced Usage

### Custom Learning Models
```python
# Access user's personalization model
if "user123" in profile_manager.user_models:
    user_model = profile_manager.user_models["user123"]
    
    # Get learning confidence
    confidence = user_model.get_learning_confidence()
    print(f"Learning confidence: {confidence:.2f}")
    
    # Personalize response using trained model
    context = {"task": "productivity", "time": "work_hours"}
    personalized = user_model.personalize_response(base_response, context)
```

### Behavioral Analysis
```python
# Analyze user behavior patterns
behavior_analysis = profile_manager.analyze_user_behavior("user123")

print("Behavior Analysis:")
print(f"Peak hours: {behavior_analysis.get('peak_hours', {})}")
print(f"Session patterns: {behavior_analysis.get('session_patterns', {})}")
print(f"Interaction trends: {behavior_analysis.get('interaction_trends', {})}")

# Get comprehensive insights
insights = profile_manager.get_user_insights("user123")
print(f"Profile completeness: {insights['profile_completeness']:.2f}")
print(f"Engagement level: {insights['engagement_level']:.2f}")
print(f"Recommendations: {len(insights['recommendations'])}")
```

### Privacy and Security
```python
from biometric_auth import PrivacyManager

privacy_manager = PrivacyManager()

# Apply privacy settings
user_data = profile_manager.get_profile("user123")
protected_data = privacy_manager.apply_privacy_settings(user_data, "minimal")

# Encrypt sensitive data
sensitive_data = {"biometric_templates": "...", "personal_notes": "..."}
encrypted = privacy_manager.encrypt_sensitive_data(sensitive_data)

# Store encrypted data safely
# ...

# Decrypt when needed
decrypted = privacy_manager.decrypt_sensitive_data(encrypted)
```

### Custom Themes and Personalization
```python
# Create custom theme
custom_theme = {
    "appearance.theme": "dark",
    "appearance.color_scheme": "purple",
    "appearance.particle_effects": True,
    "appearance.holographic_elements": True,
    "ar_interface.overlay_opacity": 0.9,
    "voice.voice_effects": True
}

success = pref_manager.create_custom_theme("my_theme", custom_theme)
if success:
    pref_manager.apply_theme("user123", "my_theme")

# Add custom shortcuts
from preference_manager import CustomizationManager

customization = CustomizationManager()

# Add keyboard shortcut
customization.add_shortcut("user123", "Ctrl+J", "open_jarvis_interface")

# Add macro
macro_steps = [
    {"action": "open_app", "app": "spotify"},
    {"action": "play_playlist", "playlist": "Focus Music"},
    {"action": "adjust_volume", "level": 0.3}
]
customization.add_macro("user123", "focus_mode", macro_steps)

# Execute customizations
action = customization.execute_shortcut("user123", "Ctrl+J")
steps = customization.execute_macro("user123", "focus_mode")
```

## API Reference

### UserProfileManager

#### Core Methods
```python
# Profile management
create_profile(user_info)                    # Create new user profile
get_profile(user_id)                         # Retrieve user profile
update_profile(user_id, updates)             # Update profile data

# Session management
start_session(user_id, context=None)         # Start interaction session
end_session(session_id)                      # End session and process learning
record_interaction(session_id, interaction)  # Record interaction data

# Personalization
get_personalized_response(user_id, response, context=None)  # Get personalized response
get_recommendations(user_id, category=None)  # Get feature recommendations
get_user_insights(user_id)                   # Get comprehensive user insights

# Analysis
analyze_user_behavior(user_id)               # Analyze behavior patterns
```

#### Profile Structure
```python
{
    "id": "user_id",
    "personal_info": {
        "name": "User Name",
        "age": 30,
        "occupation": "Job Title",
        "location": "City, Country",
        "timezone": "UTC",
        "language": "en",
        "created_at": "2024-01-01T00:00:00",
        "last_updated": "2024-01-01T12:00:00"
    },
    "preferences": {
        "communication_style": "professional",
        "response_length": "medium",
        "interaction_mode": "conversational",
        "privacy_level": "medium",
        "learning_enabled": True,
        "voice_settings": {...},
        "ui_preferences": {...}
    },
    "interests": ["technology", "music"],
    "goals": ["productivity", "automation"],
    "accessibility": {...},
    "behavioral_patterns": {...},
    "security": {...},
    "customization": {...},
    "stats": {...}
}
```

### BiometricAuthenticator

#### Authentication Methods
```python
# Enrollment
enroll_user_biometrics(user_id, face_image=None, voice_sample=None, behavior_data=None)

# Authentication
authenticate_user(user_id=None, face_image=None, voice_sample=None, behavior_data=None)

# Management
get_user_biometric_status(user_id)           # Get enrollment status
remove_user_biometrics(user_id)              # Remove biometric data
```

#### Authentication Result Format
```python
{
    "authenticated": True,
    "user_id": "user123",
    "confidence_scores": {
        "face": 0.92,
        "voice": 0.87,
        "behavior": 0.78
    },
    "method_results": {
        "face": {"success": True, "confidence": 0.92},
        "voice": {"success": True, "confidence": 0.87}
    },
    "overall_confidence": 0.89,
    "timestamp": "2024-01-01T12:00:00"
}
```

### PreferenceManager

#### Preference Methods
```python
# Basic operations
get_user_preferences(user_id)                # Get all preferences
update_preference(user_id, path, value)      # Update single preference
bulk_update_preferences(user_id, updates)    # Update multiple preferences
reset_preferences(user_id, category=None)    # Reset to defaults

# Theme management
get_theme_options()                          # Get available themes
apply_theme(user_id, theme_name)             # Apply pre-defined theme
create_custom_theme(user_id, name, settings) # Create custom theme

# Import/Export
export_preferences(user_id)                 # Export preferences
import_preferences(user_id, data)           # Import preferences
```

#### Preference Categories
- **appearance**: Theme, colors, fonts, animations, transparency
- **interaction**: Communication style, response settings, confirmations
- **voice**: Voice ID, speech rate, pitch, volume, effects
- **gesture**: Sensitivity, timeout, gesture types, feedback
- **ar_interface**: Overlay settings, hologram quality, tracking
- **privacy**: Data collection, sharing, retention, anonymization
- **accessibility**: High contrast, large text, reduced motion, voice descriptions
- **automation**: Smart suggestions, learning, context awareness
- **notifications**: Types, position, duration, sounds
- **performance**: Resource usage, optimization, hardware acceleration
- **security**: Authentication, timeouts, encryption

## Configuration

### Default Preferences File (preferences_config.json)
```json
{
  "appearance": {
    "theme": "dark",
    "font_family": "Segoe UI",
    "font_size": "medium",
    "color_scheme": "blue",
    "animations_enabled": true,
    "transparency": 0.9
  },
  "voice": {
    "voice_id": "default",
    "speech_rate": 1.0,
    "pitch": 1.0,
    "volume": 0.8
  },
  "privacy": {
    "data_collection_level": "balanced",
    "personalization_enabled": true,
    "data_retention_days": 90
  }
}
```

### Biometric Security Settings
```json
{
  "face_recognition": {
    "threshold": 0.6,
    "model": "hog",
    "max_distance": 0.6
  },
  "voice_authentication": {
    "threshold": 0.8,
    "feature_extraction": "mfcc"
  },
  "security": {
    "max_failed_attempts": 3,
    "lockout_duration": 300,
    "encryption_enabled": true
  }
}
```

## Integration Examples

### With Voice Assistant
```python
def handle_voice_command(user_id, command, context):
    # Start session
    session_id = profile_manager.start_session(user_id, context)
    
    # Process command with personalization
    base_response = voice_assistant.process_command(command)
    personalized_response = profile_manager.get_personalized_response(
        user_id, base_response, context
    )
    
    # Record interaction
    interaction_data = {
        "type": "voice_command",
        "command": command,
        "success": True,
        "response": personalized_response,
        "context": context
    }
    profile_manager.record_interaction(session_id, interaction_data)
    
    # End session
    profile_manager.end_session(session_id)
    
    return personalized_response
```

### With AR Interface
```python
def update_ar_interface(user_id):
    # Get user preferences
    prefs = pref_manager.get_user_preferences(user_id)
    
    # Configure AR based on preferences
    ar_config = {
        "overlay_opacity": prefs["ar_interface"]["overlay_opacity"],
        "hologram_quality": prefs["ar_interface"]["hologram_quality"],
        "theme": prefs["appearance"]["theme"],
        "color_scheme": prefs["appearance"]["color_scheme"]
    }
    
    # Apply accessibility settings
    if prefs["accessibility"]["high_contrast"]:
        ar_config["contrast_multiplier"] = 2.0
    if prefs["accessibility"]["large_text"]:
        ar_config["text_scale"] = 1.5
        
    ar_interface.update_configuration(ar_config)
```

### With Smart Home Integration
```python
def personalized_smart_home_control(user_id, command):
    # Get user profile and preferences
    profile = profile_manager.get_profile(user_id)
    prefs = pref_manager.get_user_preferences(user_id)
    
    # Analyze user patterns for context
    behavior = profile_manager.analyze_user_behavior(user_id)
    
    # Determine optimal settings based on user patterns
    if "lighting" in command:
        # Use learned preferences for lighting
        preferred_brightness = behavior.get("preferred_brightness", 0.8)
        preferred_color_temp = behavior.get("preferred_color_temp", "warm")
        
        smart_home.set_lighting(
            brightness=preferred_brightness,
            color_temperature=preferred_color_temp
        )
    
    # Record interaction for learning
    session_id = profile_manager.start_session(user_id, {"device": "smart_home"})
    # ... record and end session
```

## Privacy and Security

### Data Protection
- **Encryption**: All sensitive data encrypted at rest and in transit
- **Anonymization**: Personal data can be anonymized for analytics
- **Access Controls**: Granular permissions for data access
- **Audit Logging**: Complete audit trail of data access and modifications

### Privacy Levels
- **Minimal**: Basic profile info only, limited learning
- **Balanced**: Standard personalization with anonymized analytics
- **Full**: Complete personalization with comprehensive learning

### Biometric Security
- **Multi-Modal**: Combines face, voice, and behavioral authentication
- **Liveness Detection**: Prevents spoofing attacks
- **Template Protection**: Biometric templates encrypted and secured
- **Fallback Methods**: Multiple authentication options available

## Performance Optimization

### Memory Management
```python
# Configure learning buffer sizes
profile_manager.preference_learner.max_history = 1000
profile_manager.activity_tracker.max_activities = 500

# Set model update frequency
profile_manager.adaptation_threshold = 10  # Min interactions before adaptation
```

### Background Processing
```python
# Control background learning
profile_manager.background_learning_enabled = True
profile_manager.learning_interval = 300  # 5 minutes

# Optimize for performance
prefs["performance"]["performance_mode"] = "high_performance"
prefs["performance"]["resource_usage_limit"] = 0.5
```

## Testing

Run the comprehensive test suite:
```bash
cd user_profiles/
python test_user_profiles.py
```

### Test Coverage
- ✅ Profile creation and management
- ✅ Session tracking and learning
- ✅ Biometric authentication (all modes)
- ✅ Preference management and validation
- ✅ Theme and customization systems
- ✅ Privacy and security features
- ✅ Integration workflows
- ✅ Performance and optimization
- ✅ Error handling and edge cases
- ✅ Data persistence and recovery

## Troubleshooting

### Common Issues

1. **Face Recognition Not Working**
   ```bash
   pip install --upgrade face-recognition
   # On Windows, may need Visual Studio C++ Build Tools
   ```

2. **Learning Models Not Updating**
   ```python
   # Check if learning is enabled
   profile = profile_manager.get_profile(user_id)
   if not profile["preferences"]["learning_enabled"]:
       profile_manager.update_profile(user_id, {
           "preferences": {"learning_enabled": True}
       })
   ```

3. **Biometric Authentication Failing**
   ```python
   # Check enrollment status
   status = authenticator.get_user_biometric_status(user_id)
   if not status["face_enrolled"]:
       # Re-enroll biometrics
       authenticator.enroll_user_biometrics(user_id, face_image=image)
   ```

4. **Preferences Not Persisting**
   ```python
   # Check file permissions and disk space
   import os
   pref_dir = pref_manager.preferences_dir
   if not os.access(pref_dir, os.W_OK):
       print("No write permission to preferences directory")
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose logging
profile_manager.debug_mode = True
```

## File Structure

```
user_profiles/
├── __init__.py                 # Module initialization
├── user_profiles.py            # Main UserProfileManager
├── biometric_auth.py          # Biometric authentication system
├── preference_manager.py      # Preference and theme management
├── test_user_profiles.py      # Comprehensive test suite
├── README.md                  # This documentation
├── config/
│   ├── default_preferences.json
│   ├── theme_definitions.json
│   └── biometric_config.json
└── data/                      # User data storage (created at runtime)
    ├── profiles/
    ├── preferences/
    ├── biometrics/
    └── behaviors/
```

## Contributing

1. Follow existing code patterns and architecture
2. Add comprehensive tests for new features
3. Update configuration schemas as needed
4. Maintain backward compatibility
5. Document all public APIs thoroughly
6. Consider privacy implications of new features

## License

This module is part of the JARVIS AI Assistant project and follows the same licensing terms.

---

*The User Profiles module forms the intelligence core of JARVIS, enabling truly personalized AI assistance that learns and adapts to each user's unique needs and preferences.*