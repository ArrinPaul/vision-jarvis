"""
Advanced User preferences and settings management for JARVIS.
Provides comprehensive preference management, AI-powered themes, adaptive customization options,
machine learning insights, and intelligent personalization with behavioral analysis.
"""

import json
import os
import time
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

# Optional imports for enhanced ML features
try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    np = None
    KMeans = StandardScaler = IsolationForest = None

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

@dataclass
class PreferenceChange:
    """Structured preference change event."""
    timestamp: str
    user_id: str
    preference_path: str
    old_value: Any
    new_value: Any
    context: Dict[str, Any]
    satisfaction_impact: Optional[float] = None
    performance_impact: Optional[float] = None

@dataclass
class AdaptiveRecommendation:
    """AI-powered preference recommendation."""
    preference_path: str
    recommended_value: Any
    confidence: float
    reasoning: str
    expected_benefit: str
    category: str
    priority: int = 5  # 1-10 scale

class PreferenceManager:
    """
    Comprehensive preference management system for JARVIS users.
    Handles UI preferences, behavior settings, accessibility options, and customizations.
    """
    
    def __init__(self, preferences_dir="user_preferences"):
        self.preferences_dir = preferences_dir
        self.user_preferences = {}
        self.default_preferences = self._get_default_preferences()
        self.preference_schema = self._get_preference_schema()
        
        # Theme management
        self.theme_manager = ThemeManager()
        self.customization_manager = CustomizationManager()
        
        # Advanced AI-powered features
        self.adaptive_engine = AdaptivePreferenceEngine()
        self.ml_analyzer = PreferenceMLAnalyzer()
        self.behavioral_tracker = PreferenceBehavioralTracker()
        self.recommendation_engine = IntelligentRecommendationEngine()
        
        # Real-time processing
        self.preference_changes = deque(maxlen=1000)  # Recent changes
        self.user_sessions = defaultdict(dict)
        self.contextual_patterns = defaultdict(dict)
        
        # Machine learning models
        self.preference_models = {}
        self.satisfaction_predictors = {}
        self.usage_optimizers = {}
        
        # Background processing
        self.processing_thread = threading.Thread(target=self._background_processing, daemon=True)
        self.processing_enabled = True
        
        # Create directories
        os.makedirs(preferences_dir, exist_ok=True)
        os.makedirs(f"{preferences_dir}/models", exist_ok=True)
        os.makedirs(f"{preferences_dir}/analytics", exist_ok=True)
        
        # Load existing preferences
        self._load_all_preferences()
        self._load_ml_models()
        
        # Start background processing
        self.processing_thread.start()
        
        logging.info("Advanced Preference Manager initialized with AI capabilities")
        
    def _get_default_preferences(self):
        """Get default preference values."""
        return {
            "appearance": {
                "theme": "dark",
                "font_family": "Segoe UI",
                "font_size": "medium",
                "color_scheme": "blue",
                "animations_enabled": True,
                "transparency": 0.9,
                "blur_effects": True,
                "particle_effects": True,
                "holographic_elements": True
            },
            "interaction": {
                "communication_style": "professional",
                "response_speed": "normal",
                "response_length": "medium",
                "confirmation_level": "medium",
                "auto_suggestions": True,
                "predictive_text": True,
                "voice_feedback": True,
                "haptic_feedback": False
            },
            "voice": {
                "voice_id": "default",
                "speech_rate": 1.0,
                "pitch": 1.0,
                "volume": 0.8,
                "voice_effects": False,
                "background_listening": True,
                "wake_word_sensitivity": 0.7,
                "noise_cancellation": True
            },
            "gesture": {
                "gesture_sensitivity": 0.8,
                "gesture_timeout": 3.0,
                "air_tap_enabled": True,
                "swipe_gestures": True,
                "pinch_zoom": True,
                "custom_gestures_enabled": True,
                "gesture_feedback": "visual"
            },
            "ar_interface": {
                "overlay_opacity": 0.8,
                "hologram_quality": "high",
                "spatial_anchoring": True,
                "hand_tracking_enabled": True,
                "eye_gaze_tracking": False,
                "depth_perception": True,
                "occlusion_handling": True
            },
            "privacy": {
                "data_collection_level": "balanced",
                "personalization_enabled": True,
                "usage_analytics": True,
                "crash_reporting": True,
                "location_sharing": False,
                "biometric_auth_required": False,
                "session_recording": False,
                "data_retention_days": 90
            },
            "accessibility": {
                "high_contrast": False,
                "large_text": False,
                "screen_reader_compatible": False,
                "keyboard_navigation": True,
                "reduced_motion": False,
                "color_blind_friendly": False,
                "voice_descriptions": False,
                "subtitle_enabled": False
            },
            "automation": {
                "smart_suggestions": True,
                "auto_complete": True,
                "predictive_actions": True,
                "learning_enabled": True,
                "routine_detection": True,
                "context_awareness": True,
                "proactive_assistance": True
            },
            "notifications": {
                "system_notifications": True,
                "achievement_notifications": True,
                "reminder_notifications": True,
                "security_alerts": True,
                "notification_sound": True,
                "notification_position": "top_right",
                "notification_duration": 5000,
                "do_not_disturb_hours": []
            },
            "performance": {
                "performance_mode": "balanced",
                "resource_usage_limit": 0.7,
                "background_processing": True,
                "cache_enabled": True,
                "preload_content": True,
                "optimize_battery": False,
                "gpu_acceleration": True
            },
            "security": {
                "auto_lock_timeout": 900,  # 15 minutes
                "require_auth_for_sensitive": True,
                "biometric_primary": True,
                "password_complexity": "medium",
                "session_timeout": 3600,  # 1 hour
                "encryption_enabled": True,
                "secure_deletion": True
            }
        }
        
    def _get_preference_schema(self):
        """Get preference validation schema."""
        return {
            "appearance.theme": {"type": "select", "options": ["light", "dark", "auto", "custom"]},
            "appearance.font_size": {"type": "select", "options": ["small", "medium", "large", "extra_large"]},
            "appearance.color_scheme": {"type": "select", "options": ["blue", "green", "red", "purple", "orange", "custom"]},
            "appearance.transparency": {"type": "range", "min": 0.0, "max": 1.0},
            "voice.speech_rate": {"type": "range", "min": 0.5, "max": 2.0},
            "voice.pitch": {"type": "range", "min": 0.5, "max": 2.0},
            "voice.volume": {"type": "range", "min": 0.0, "max": 1.0},
            "gesture.gesture_sensitivity": {"type": "range", "min": 0.1, "max": 1.0},
            "gesture.gesture_timeout": {"type": "range", "min": 1.0, "max": 10.0},
            "privacy.data_retention_days": {"type": "range", "min": 1, "max": 3650},
            "performance.resource_usage_limit": {"type": "range", "min": 0.1, "max": 1.0}
        }
        
    def get_user_preferences(self, user_id):
        """Get comprehensive preferences for a user."""
        if user_id not in self.user_preferences:
            # Initialize with defaults
            self.user_preferences[user_id] = self.default_preferences.copy()
            self._save_user_preferences(user_id)
            
        return self.user_preferences[user_id].copy()
        
    def update_preference(self, user_id, preference_path, value):
        """Update a specific preference."""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = self.default_preferences.copy()
            
        # Validate preference
        if not self._validate_preference(preference_path, value):
            raise ValueError(f"Invalid value for preference {preference_path}: {value}")
            
        # Navigate to the preference and update it
        prefs = self.user_preferences[user_id]
        keys = preference_path.split('.')
        
        for key in keys[:-1]:
            if key not in prefs:
                prefs[key] = {}
            prefs = prefs[key]
            
        prefs[keys[-1]] = value
        
        # Save changes
        self._save_user_preferences(user_id)
        
        # Apply immediate effects
        self._apply_preference_change(user_id, preference_path, value)
        
        logging.info(f"Updated preference {preference_path} = {value} for user {user_id}")
        
    def bulk_update_preferences(self, user_id, preference_updates):
        """Update multiple preferences at once."""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = self.default_preferences.copy()
            
        # Validate all preferences first
        for path, value in preference_updates.items():
            if not self._validate_preference(path, value):
                raise ValueError(f"Invalid value for preference {path}: {value}")
                
        # Apply all updates
        for path, value in preference_updates.items():
            self.update_preference(user_id, path, value)
            
        logging.info(f"Bulk updated {len(preference_updates)} preferences for user {user_id}")
        
    def reset_preferences(self, user_id, category=None):
        """Reset preferences to defaults."""
        if category:
            # Reset specific category
            if user_id in self.user_preferences:
                if category in self.default_preferences:
                    self.user_preferences[user_id][category] = self.default_preferences[category].copy()
                    self._save_user_preferences(user_id)
                    logging.info(f"Reset {category} preferences for user {user_id}")
        else:
            # Reset all preferences
            self.user_preferences[user_id] = self.default_preferences.copy()
            self._save_user_preferences(user_id)
            logging.info(f"Reset all preferences for user {user_id}")
            
    def get_preference_value(self, user_id, preference_path, default=None):
        """Get a specific preference value."""
        prefs = self.get_user_preferences(user_id)
        
        keys = preference_path.split('.')
        current = prefs
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
                
        return current
        
    def export_preferences(self, user_id):
        """Export user preferences to a portable format."""
        if user_id not in self.user_preferences:
            return {}
            
        export_data = {
            "user_id": user_id,
            "export_timestamp": datetime.now().isoformat(),
            "preferences": self.user_preferences[user_id],
            "version": "1.0"
        }
        
        return export_data
        
    def import_preferences(self, user_id, import_data):
        """Import preferences from exported data."""
        if "preferences" not in import_data:
            raise ValueError("Invalid import data format")
            
        imported_prefs = import_data["preferences"]
        
        # Validate imported preferences
        valid_prefs = {}
        for category, settings in imported_prefs.items():
            if category in self.default_preferences:
                valid_prefs[category] = {}
                for setting, value in settings.items():
                    if setting in self.default_preferences[category]:
                        preference_path = f"{category}.{setting}"
                        if self._validate_preference(preference_path, value):
                            valid_prefs[category][setting] = value
                            
        # Apply validated preferences
        self.user_preferences[user_id] = self._merge_preferences(
            self.default_preferences, valid_prefs
        )
        
        self._save_user_preferences(user_id)
        
        logging.info(f"Imported preferences for user {user_id}")
        
    def get_theme_options(self):
        """Get available theme options."""
        return self.theme_manager.get_available_themes()
        
    def apply_theme(self, user_id, theme_name):
        """Apply a theme to user preferences."""
        theme_prefs = self.theme_manager.get_theme_preferences(theme_name)
        
        if theme_prefs:
            self.bulk_update_preferences(user_id, theme_prefs)
            logging.info(f"Applied theme {theme_name} for user {user_id}")
            return True
            
        return False
        
    def create_custom_theme(self, user_id, theme_name, theme_settings):
        """Create and apply a custom theme."""
        if self.theme_manager.create_custom_theme(theme_name, theme_settings):
            self.apply_theme(user_id, theme_name)
            return True
            
        return False
        
    def get_accessibility_recommendations(self, user_id):
        """Get accessibility recommendations based on user needs."""
        current_prefs = self.get_user_preferences(user_id)
        recommendations = []
        
        accessibility = current_prefs.get("accessibility", {})
        
        # Check for potential accessibility needs
        appearance = current_prefs.get("appearance", {})
        
        if appearance.get("font_size") == "small":
            recommendations.append({
                "category": "accessibility",
                "setting": "large_text",
                "recommendation": True,
                "reason": "Larger text may be easier to read"
            })
            
        if not accessibility.get("high_contrast"):
            recommendations.append({
                "category": "accessibility",
                "setting": "high_contrast",
                "recommendation": True,
                "reason": "High contrast improves readability"
            })
            
        if appearance.get("animations_enabled") and not accessibility.get("reduced_motion"):
            recommendations.append({
                "category": "accessibility",
                "setting": "reduced_motion",
                "recommendation": True,
                "reason": "Reduced motion can help with motion sensitivity"
            })
            
        return recommendations
        
    def _validate_preference(self, preference_path, value):
        """Validate a preference value against schema."""
        if preference_path not in self.preference_schema:
            return True  # Allow unknown preferences
            
        schema = self.preference_schema[preference_path]
        
        if schema["type"] == "select":
            return value in schema["options"]
        elif schema["type"] == "range":
            return schema["min"] <= value <= schema["max"]
        elif schema["type"] == "boolean":
            return isinstance(value, bool)
        elif schema["type"] == "string":
            return isinstance(value, str)
        elif schema["type"] == "number":
            return isinstance(value, (int, float))
            
        return True
        
    def _apply_preference_change(self, user_id, preference_path, value):
        """Apply immediate effects of preference changes."""
        # Theme changes
        if preference_path.startswith("appearance."):
            self._apply_appearance_change(user_id, preference_path, value)
            
        # Voice changes
        elif preference_path.startswith("voice."):
            self._apply_voice_change(user_id, preference_path, value)
            
        # Performance changes
        elif preference_path.startswith("performance."):
            self._apply_performance_change(user_id, preference_path, value)
            
    def _apply_appearance_change(self, user_id, preference_path, value):
        """Apply appearance preference changes."""
        # This would trigger UI updates in the actual application
        logging.info(f"Applying appearance change: {preference_path} = {value}")
        
    def _apply_voice_change(self, user_id, preference_path, value):
        """Apply voice preference changes."""
        # This would update voice synthesis settings
        logging.info(f"Applying voice change: {preference_path} = {value}")
        
    def _apply_performance_change(self, user_id, preference_path, value):
        """Apply performance preference changes."""
        # This would adjust system performance settings
        logging.info(f"Applying performance change: {preference_path} = {value}")
        
    def _merge_preferences(self, base_prefs, update_prefs):
        """Deep merge preference dictionaries."""
        result = base_prefs.copy()
        
        for key, value in update_prefs.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_preferences(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def _load_all_preferences(self):
        """Load all user preferences from storage."""
        try:
            prefs_file = os.path.join(self.preferences_dir, "all_preferences.json")
            if os.path.exists(prefs_file):
                with open(prefs_file, 'r') as f:
                    self.user_preferences = json.load(f)
                    
            logging.info("User preferences loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading preferences: {e}")
            
    def _save_user_preferences(self, user_id):
        """Save preferences for a specific user."""
        try:
            # Save to individual file
            user_file = os.path.join(self.preferences_dir, f"{user_id}_preferences.json")
            with open(user_file, 'w') as f:
                json.dump(self.user_preferences[user_id], f, indent=2)
                
            # Save to master file
            all_prefs_file = os.path.join(self.preferences_dir, "all_preferences.json")
            with open(all_prefs_file, 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving preferences: {e}")
    
    def get_ai_recommendations(self, user_id: str, context: Dict[str, Any] = None) -> List[AdaptiveRecommendation]:
        """Get AI-powered preference recommendations for a user."""
        if context is None:
            context = {}
        
        # Get preference history
        preference_history = self._get_preference_history(user_id)
        
        # Generate comprehensive recommendations
        recommendations = self.recommendation_engine.generate_comprehensive_recommendations(
            user_id, context, preference_history
        )
        
        return recommendations
    
    def apply_adaptive_optimizations(self, user_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply AI-powered adaptive optimizations to user preferences."""
        if context is None:
            context = {}
        
        # Get adaptive recommendations
        adaptations = self.adaptive_engine.adapt_preferences_automatically(user_id, context)
        
        results = {
            "applied_adaptations": [],
            "failed_adaptations": [],
            "total_adaptations": len(adaptations)
        }
        
        # Apply each adaptation
        for adaptation in adaptations:
            try:
                self.update_preference(user_id, adaptation.preference_path, adaptation.recommended_value)
                results["applied_adaptations"].append({
                    "preference": adaptation.preference_path,
                    "value": adaptation.recommended_value,
                    "reasoning": adaptation.reasoning,
                    "confidence": adaptation.confidence
                })
            except Exception as e:
                results["failed_adaptations"].append({
                    "preference": adaptation.preference_path,
                    "error": str(e)
                })
        
        return results
    
    def analyze_preference_patterns(self, user_id: str) -> Dict[str, Any]:
        """Perform comprehensive ML analysis of user's preference patterns."""
        preference_history = self._get_preference_history(user_id)
        
        if not preference_history:
            return {"message": "No preference history available for analysis"}
        
        # Perform ML analysis
        ml_analysis = self.ml_analyzer.analyze_preference_patterns(user_id, preference_history)
        
        # Perform behavioral analysis
        behavioral_analysis = self.behavioral_tracker.analyze_behavioral_trends(user_id)
        
        # Combine analyses
        comprehensive_analysis = {
            "user_id": user_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "ml_analysis": ml_analysis,
            "behavioral_analysis": behavioral_analysis,
            "summary": self._generate_analysis_summary(ml_analysis, behavioral_analysis)
        }
        
        return comprehensive_analysis
    
    def track_preference_satisfaction(self, user_id: str, preference_path: str, 
                                    satisfaction_score: float, context: Dict[str, Any] = None):
        """Track user satisfaction with preference changes."""
        if context is None:
            context = {}
        
        # Track with behavioral tracker
        self.behavioral_tracker.track_preference_usage(user_id, preference_path, context, satisfaction_score)
        
        # Record satisfaction change
        change = PreferenceChange(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            preference_path=preference_path,
            old_value=None,  # We don't have old value in this context
            new_value=None,  # We don't have new value in this context
            context=context,
            satisfaction_impact=satisfaction_score
        )
        
        self.preference_changes.append(change)
        
        logging.info(f"Tracked satisfaction for {user_id}: {preference_path} = {satisfaction_score}")
    
    def get_contextual_preferences(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimized preferences for a specific context."""
        # Get behavioral recommendations for this context
        behavioral_recs = self.behavioral_tracker.get_contextual_recommendations(user_id, context)
        
        # Get base preferences
        base_preferences = self.get_user_preferences(user_id)
        
        # Apply contextual optimizations
        optimized_preferences = base_preferences.copy()
        
        for rec in behavioral_recs[:5]:  # Apply top 5 recommendations
            if rec.confidence > 0.7:  # High confidence threshold
                self._set_nested_preference(optimized_preferences, rec.preference_path, rec.recommended_value)
        
        return {
            "base_preferences": base_preferences,
            "optimized_preferences": optimized_preferences,
            "applied_recommendations": [asdict(rec) for rec in behavioral_recs[:5]],
            "context": context
        }
    
    def _background_processing(self):
        """Background processing for continuous preference optimization."""
        while self.processing_enabled:
            try:
                # Process recent preference changes
                if self.preference_changes:
                    recent_changes = list(self.preference_changes)[-50:]  # Last 50 changes
                    self._process_preference_batch(recent_changes)
                
                # Update ML models periodically
                self._update_ml_models()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(300)  # Process every 5 minutes
                
            except Exception as e:
                logging.error(f"Background processing error: {e}")
                time.sleep(300)
    
    def _process_preference_batch(self, changes: List[PreferenceChange]):
        """Process a batch of preference changes for pattern detection."""
        # Group changes by user
        user_changes = defaultdict(list)
        for change in changes:
            user_changes[change.user_id].append(change)
        
        # Analyze patterns for each user
        for user_id, user_change_list in user_changes.items():
            if len(user_change_list) >= 3:  # Minimum changes for analysis
                self._update_user_patterns(user_id, user_change_list)
    
    def _update_user_patterns(self, user_id: str, changes: List[PreferenceChange]):
        """Update user patterns based on recent changes."""
        # Analyze for common preference paths
        preference_frequency = defaultdict(int)
        satisfaction_by_preference = defaultdict(list)
        
        for change in changes:
            preference_frequency[change.preference_path] += 1
            if change.satisfaction_impact is not None:
                satisfaction_by_preference[change.preference_path].append(change.satisfaction_impact)
        
        # Update user profile with insights
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}
        
        session = self.user_sessions[user_id]
        session["most_changed_preferences"] = dict(preference_frequency)
        session["preference_satisfaction"] = {
            pref: np.mean(scores) if np and scores else 0.5 
            for pref, scores in satisfaction_by_preference.items()
        }
        session["last_analysis"] = datetime.now().isoformat()
    
    def _get_preference_history(self, user_id: str) -> List[PreferenceChange]:
        """Get preference change history for a user."""
        return [change for change in self.preference_changes if change.user_id == user_id]
    
    def _generate_analysis_summary(self, ml_analysis: Dict[str, Any], behavioral_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of comprehensive preference analysis."""
        summary = {
            "overall_health": "good",
            "key_insights": [],
            "improvement_opportunities": [],
            "satisfaction_trend": "stable"
        }
        
        # Analyze ML results
        if "patterns" in ml_analysis and ml_analysis["patterns"]:
            summary["key_insights"].append("ML analysis detected preference patterns")
        
        # Analyze behavioral results
        if "overall_metrics" in behavioral_analysis:
            metrics = behavioral_analysis["overall_metrics"]
            success_rate = metrics.get("success_rate", 0.5)
            
            if success_rate > 0.8:
                summary["overall_health"] = "excellent"
                summary["satisfaction_trend"] = "improving"
            elif success_rate < 0.4:
                summary["overall_health"] = "needs_attention"
                summary["improvement_opportunities"].append("Low preference adaptation success rate")
        
        return summary
    
    def _set_nested_preference(self, preferences: Dict[str, Any], preference_path: str, value: Any):
        """Set a nested preference value using dot notation path."""
        keys = preference_path.split('.')
        current = preferences
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _load_ml_models(self):
        """Load saved ML models and analytics data."""
        try:
            models_dir = f"{self.preferences_dir}/models"
            if os.path.exists(models_dir):
                # Load user patterns
                patterns_file = os.path.join(models_dir, "user_patterns.json")
                if os.path.exists(patterns_file):
                    with open(patterns_file, 'r') as f:
                        patterns_data = json.load(f)
                        for user_id, patterns in patterns_data.items():
                            self.user_sessions[user_id] = patterns
                
                logging.info("ML models and patterns loaded successfully")
        except Exception as e:
            logging.error(f"Error loading ML models: {e}")
    
    def _update_ml_models(self):
        """Update and save ML models periodically."""
        try:
            # Save user patterns
            models_dir = f"{self.preferences_dir}/models"
            patterns_file = os.path.join(models_dir, "user_patterns.json")
            
            with open(patterns_file, 'w') as f:
                json.dump(dict(self.user_sessions), f, indent=2, default=str)
            
            logging.debug("ML models updated and saved")
        except Exception as e:
            logging.error(f"Error updating ML models: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory bloat."""
        # Keep only last 30 days of preference changes
        cutoff_time = datetime.now() - timedelta(days=30)
        
        filtered_changes = deque(maxlen=1000)
        for change in self.preference_changes:
            change_time = datetime.fromisoformat(change.timestamp)
            if change_time >= cutoff_time:
                filtered_changes.append(change)
        
        self.preference_changes = filtered_changes
        
        logging.debug("Cleaned up old preference data")

class ThemeManager:
    """
    Advanced theme management with adaptive preferences, smart defaults, and context-aware theming.
    Features: AI-powered theme suggestions, dynamic theme switching, accessibility integration.
    """
    
    def __init__(self):
        self.built_in_themes = self._get_built_in_themes()
        self.custom_themes = {}
        self.adaptive_themes = {}
        self.theme_usage_analytics = defaultdict(dict)
        self.contextual_themes = defaultdict(dict)
        
        # Advanced theming features
        self.auto_theme_switching = {}
        self.accessibility_adaptations = {}
        self.performance_optimizations = {}
        self.user_theme_preferences = defaultdict(dict)
        
        # AI-powered features
        self.theme_recommendation_engine = ThemeRecommendationEngine()
        self.color_harmony_analyzer = ColorHarmonyAnalyzer()
        self.accessibility_checker = AccessibilityChecker()
        
        # Load existing data
        self._load_custom_themes()
        self._load_theme_analytics()
        
        logging.info("Advanced Theme Manager initialized")
        
    def _get_built_in_themes(self):
        """Get comprehensive built-in theme definitions with advanced features."""
        return {
            # Classic themes
            "light": {
                "name": "Light Theme",
                "description": "Clean, bright interface optimized for daylight use",
                "category": "classic",
                "appearance.theme": "light",
                "appearance.color_scheme": "blue",
                "appearance.transparency": 0.95,
                "appearance.blur_effects": False,
                "appearance.particle_effects": False,
                "accessibility.high_contrast": False,
                "performance.gpu_acceleration": False,
                "colors": {
                    "primary": "#007ACC",
                    "secondary": "#E3F2FD",
                    "accent": "#FF6B6B",
                    "background": "#FFFFFF",
                    "surface": "#F5F5F5",
                    "text": "#212121",
                    "text_secondary": "#757575"
                },
                "optimal_conditions": ["daylight", "bright_environment", "outdoor"],
                "accessibility_features": ["color_blind_friendly", "high_readability"]
            },
            "dark": {
                "name": "Dark Theme",
                "description": "Elegant dark interface reducing eye strain in low light",
                "category": "classic",
                "appearance.theme": "dark",
                "appearance.color_scheme": "blue",
                "appearance.transparency": 0.9,
                "appearance.blur_effects": True,
                "appearance.particle_effects": True,
                "accessibility.high_contrast": False,
                "performance.gpu_acceleration": True,
                "colors": {
                    "primary": "#64B5F6",
                    "secondary": "#1E1E1E",
                    "accent": "#FF8A65",
                    "background": "#121212",
                    "surface": "#1E1E1E",
                    "text": "#FFFFFF",
                    "text_secondary": "#B0B0B0"
                },
                "optimal_conditions": ["evening", "low_light", "indoor"],
                "accessibility_features": ["reduced_blue_light", "eye_strain_reduction"]
            },
            
            # Accessibility themes
            "high_contrast": {
                "name": "High Contrast",
                "description": "Maximum contrast for visual accessibility",
                "category": "accessibility",
                "appearance.theme": "dark",
                "appearance.color_scheme": "custom",
                "accessibility.high_contrast": True,
                "accessibility.large_text": True,
                "appearance.animations_enabled": False,
                "appearance.blur_effects": False,
                "colors": {
                    "primary": "#FFFF00",
                    "secondary": "#000000",
                    "accent": "#00FF00",
                    "background": "#000000",
                    "surface": "#1A1A1A",
                    "text": "#FFFFFF",
                    "text_secondary": "#FFFF00"
                },
                "optimal_conditions": ["visual_impairment", "bright_conditions"],
                "accessibility_features": ["maximum_contrast", "large_text", "clear_focus"]
            },
            
            # Specialized themes
            "cyberpunk": {
                "name": "Cyberpunk",
                "description": "Futuristic neon-lit interface for tech enthusiasts",
                "category": "entertainment",
                "appearance.theme": "dark",
                "appearance.color_scheme": "green",
                "appearance.particle_effects": True,
                "appearance.holographic_elements": True,
                "appearance.transparency": 0.8,
                "performance.gpu_acceleration": True,
                "colors": {
                    "primary": "#00FF41",
                    "secondary": "#0D1B2A",
                    "accent": "#FF073A",
                    "background": "#000000",
                    "surface": "#0F3460",
                    "text": "#00FF41",
                    "text_secondary": "#39A0ED"
                },
                "optimal_conditions": ["gaming", "entertainment", "night"],
                "special_effects": ["neon_glow", "matrix_rain", "glitch_effects"]
            },
            "iron_man": {
                "name": "Iron Man",
                "description": "Arc Reactor inspired interface with holographic elements",
                "category": "entertainment",
                "appearance.theme": "dark",
                "appearance.color_scheme": "red",
                "appearance.holographic_elements": True,
                "appearance.particle_effects": True,
                "ar_interface.hologram_quality": "high",
                "ar_interface.overlay_opacity": 0.9,
                "performance.gpu_acceleration": True,
                "colors": {
                    "primary": "#FFD700",
                    "secondary": "#1A0000",
                    "accent": "#FF073A",
                    "background": "#0A0A0A",
                    "surface": "#2D1B1B",
                    "text": "#FFD700",
                    "text_secondary": "#FF6B6B"
                },
                "optimal_conditions": ["presentation", "demo", "tech_showcase"],
                "special_effects": ["arc_reactor_glow", "holographic_overlays", "energy_patterns"]
            },
            
            # Productivity themes
            "focus": {
                "name": "Focus Mode",
                "description": "Minimal distraction interface for maximum concentration",
                "category": "productivity",
                "appearance.theme": "light",
                "appearance.animations_enabled": False,
                "appearance.particle_effects": False,
                "appearance.blur_effects": False,
                "appearance.holographic_elements": False,
                "notification.notification_duration": 2000,
                "colors": {
                    "primary": "#2E7D32",
                    "secondary": "#F8F9FA",
                    "accent": "#FF9800",
                    "background": "#FAFAFA",
                    "surface": "#FFFFFF",
                    "text": "#212121",
                    "text_secondary": "#616161"
                },
                "optimal_conditions": ["work", "study", "concentration"],
                "productivity_features": ["minimal_distractions", "calm_colors", "clean_layout"]
            },
            
            # Environmental adaptive themes
            "nature": {
                "name": "Nature",
                "description": "Earth-toned interface inspired by natural environments",
                "category": "environmental",
                "appearance.theme": "light",
                "appearance.color_scheme": "green",
                "appearance.particle_effects": True,
                "colors": {
                    "primary": "#4CAF50",
                    "secondary": "#E8F5E8",
                    "accent": "#FF9800",
                    "background": "#F1F8E9",
                    "surface": "#FFFFFF",
                    "text": "#2E7D32",
                    "text_secondary": "#689F38"
                },
                "optimal_conditions": ["outdoor", "relaxation", "wellness"],
                "special_effects": ["leaf_particles", "nature_sounds", "breathing_animations"]
            },
            
            # Dynamic themes
            "adaptive_auto": {
                "name": "Smart Adaptive",
                "description": "AI-powered theme that adapts to your environment and usage patterns",
                "category": "dynamic",
                "is_adaptive": True,
                "base_theme": "dark",
                "adaptation_rules": {
                    "time_based": True,
                    "activity_based": True,
                    "environment_based": True,
                    "performance_based": True
                },
                "learning_enabled": True,
                "optimal_conditions": ["any"],
                "ai_features": ["usage_learning", "context_adaptation", "preference_prediction"]
            }
        }
    
    def get_theme_recommendations(self, user_id, context=None):
        """Get AI-powered theme recommendations based on user behavior and context."""
        return self.theme_recommendation_engine.get_recommendations(
            user_id, self.theme_usage_analytics[user_id], context
        )
    
    def apply_adaptive_theme(self, user_id, context):
        """Apply adaptive theme based on current context."""
        if user_id not in self.adaptive_themes:
            self.adaptive_themes[user_id] = {
                "enabled": False,
                "base_theme": "dark",
                "adaptation_history": [],
                "learned_patterns": {}
            }
        
        adaptive_config = self.adaptive_themes[user_id]
        if not adaptive_config["enabled"]:
            return None
            
        # Analyze context for theme adaptation
        optimal_theme = self._determine_optimal_theme(user_id, context)
        
        if optimal_theme:
            # Apply theme adaptations
            adapted_theme = self._create_adapted_theme(optimal_theme, context)
            
            # Track adaptation
            self._track_theme_adaptation(user_id, optimal_theme, context)
            
            return adapted_theme
        
        return None
    
    def create_contextual_theme_rule(self, user_id, context_conditions, theme_name):
        """Create a rule for automatic theme switching based on context."""
        if user_id not in self.contextual_themes:
            self.contextual_themes[user_id] = {}
            
        rule_id = f"rule_{len(self.contextual_themes[user_id]) + 1}"
        self.contextual_themes[user_id][rule_id] = {
            "conditions": context_conditions,
            "theme": theme_name,
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
            "effectiveness_score": 0.5
        }
        
        return rule_id
    
    def optimize_theme_for_accessibility(self, theme_name, accessibility_requirements):
        """Optimize a theme for specific accessibility needs."""
        base_theme = self.get_theme_preferences(theme_name)
        if not base_theme:
            return None
            
        optimized_theme = base_theme.copy()
        
        # Apply accessibility optimizations
        if "visual_impairment" in accessibility_requirements:
            optimized_theme["accessibility.high_contrast"] = True
            optimized_theme["accessibility.large_text"] = True
            optimized_theme["appearance.animations_enabled"] = False
            
        if "color_blind" in accessibility_requirements:
            optimized_theme = self.color_harmony_analyzer.adapt_for_color_blindness(optimized_theme)
            
        if "motor_impairment" in accessibility_requirements:
            optimized_theme["gesture.gesture_timeout"] = 5.0
            optimized_theme["interaction.confirmation_level"] = "high"
            
        # Validate accessibility
        accessibility_score = self.accessibility_checker.evaluate_theme(optimized_theme)
        optimized_theme["accessibility_score"] = accessibility_score
        
        return optimized_theme
    
    def analyze_theme_performance(self, user_id, theme_name):
        """Analyze performance metrics for a theme."""
        usage_data = self.theme_usage_analytics[user_id].get(theme_name, {})
        
        if not usage_data:
            return {"error": "No usage data available"}
            
        return {
            "total_usage_time": usage_data.get("total_time", 0),
            "usage_sessions": usage_data.get("session_count", 0),
            "average_session_length": usage_data.get("avg_session_length", 0),
            "user_satisfaction": usage_data.get("satisfaction_rating", 0),
            "performance_impact": usage_data.get("performance_score", 0),
            "accessibility_compliance": usage_data.get("accessibility_score", 0),
            "most_used_contexts": usage_data.get("contexts", []),
            "effectiveness_rating": self._calculate_theme_effectiveness(user_id, theme_name)
        }
    
    def export_custom_theme(self, theme_name, export_format="json"):
        """Export a custom theme for sharing or backup."""
        if theme_name not in self.custom_themes:
            return None
            
        theme_data = self.custom_themes[theme_name].copy()
        theme_data["export_metadata"] = {
            "exported_at": datetime.now().isoformat(),
            "theme_version": "1.0",
            "export_format": export_format
        }
        
        if export_format == "json":
            return json.dumps(theme_data, indent=2)
        elif export_format == "dict":
            return theme_data
        else:
            return None
    
    def import_theme(self, theme_data, theme_name=None):
        """Import a theme from external source."""
        try:
            if isinstance(theme_data, str):
                theme_config = json.loads(theme_data)
            else:
                theme_config = theme_data
                
            # Validate theme structure
            if not self._validate_theme_structure(theme_config):
                return {"success": False, "error": "Invalid theme structure"}
                
            # Generate name if not provided
            if not theme_name:
                theme_name = theme_config.get("name", f"imported_theme_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
            # Check for conflicts
            if theme_name in self.built_in_themes:
                return {"success": False, "error": "Cannot override built-in theme"}
                
            # Import theme
            self.custom_themes[theme_name] = theme_config
            self._save_custom_themes()
            
            return {"success": True, "theme_name": theme_name}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_theme_usage_insights(self, user_id):
        """Get comprehensive insights about theme usage patterns."""
        usage_data = self.theme_usage_analytics[user_id]
        
        if not usage_data:
            return {"message": "No usage data available"}
            
        # Calculate insights
        total_time = sum(theme.get("total_time", 0) for theme in usage_data.values())
        most_used_theme = max(usage_data.items(), key=lambda x: x[1].get("total_time", 0))[0] if usage_data else None
        
        theme_diversity = len([theme for theme, data in usage_data.items() if data.get("session_count", 0) > 0])
        
        return {
            "total_theme_usage_time": total_time,
            "most_used_theme": most_used_theme,
            "theme_diversity_score": theme_diversity / len(self.built_in_themes),
            "adaptive_theme_effectiveness": self._calculate_adaptive_effectiveness(user_id),
            "recommended_optimizations": self._generate_theme_optimizations(user_id),
            "usage_patterns": self._analyze_usage_patterns(user_id)
        }
        
    def get_available_themes(self):
        """Get list of all available themes."""
        all_themes = {}
        all_themes.update(self.built_in_themes)
        all_themes.update(self.custom_themes)
        
        return {
            "built_in": list(self.built_in_themes.keys()),
            "custom": list(self.custom_themes.keys()),
            "all": list(all_themes.keys())
        }
        
    def get_theme_preferences(self, theme_name):
        """Get preference settings for a theme."""
        if theme_name in self.built_in_themes:
            return self.built_in_themes[theme_name]
        elif theme_name in self.custom_themes:
            return self.custom_themes[theme_name]
        else:
            return None
            
    def create_custom_theme(self, theme_name, theme_settings):
        """Create a new custom theme."""
        if theme_name in self.built_in_themes:
            return False  # Cannot override built-in themes
            
        self.custom_themes[theme_name] = theme_settings
        self._save_custom_themes()
        
        return True
        
    def _save_custom_themes(self):
        """Save custom themes to storage."""
        try:
            with open("custom_themes.json", 'w') as f:
                json.dump(self.custom_themes, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving custom themes: {e}")
    
    # Helper methods for ThemeManager
    def _load_custom_themes(self):
        """Load custom themes from storage."""
        try:
            if os.path.exists("custom_themes.json"):
                with open("custom_themes.json", 'r') as f:
                    self.custom_themes = json.load(f)
        except Exception as e:
            logging.error(f"Error loading custom themes: {e}")
    
    def _load_theme_analytics(self):
        """Load theme usage analytics."""
        try:
            if os.path.exists("theme_analytics.json"):
                with open("theme_analytics.json", 'r') as f:
                    analytics_data = json.load(f)
                    for user_id, data in analytics_data.items():
                        self.theme_usage_analytics[user_id] = data
        except Exception as e:
            logging.error(f"Error loading theme analytics: {e}")
    
    def _determine_optimal_theme(self, user_id, context):
        """Determine optimal theme based on context."""
        # Simple logic for theme determination
        current_hour = datetime.now().hour
        
        if context.get("activity") == "work":
            return "focus"
        elif 6 <= current_hour < 18:
            return "light"
        else:
            return "dark"
    
    def _create_adapted_theme(self, base_theme, context):
        """Create adapted theme based on context."""
        adapted = self.get_theme_preferences(base_theme).copy() if self.get_theme_preferences(base_theme) else {}
        
        # Apply context-based adaptations
        if context.get("performance_mode") == "low":
            adapted["appearance.animations_enabled"] = False
            adapted["appearance.particle_effects"] = False
        
        return adapted
    
    def _track_theme_adaptation(self, user_id, theme_name, context):
        """Track theme adaptation for learning."""
        if user_id not in self.adaptive_themes:
            self.adaptive_themes[user_id] = {"adaptation_history": []}
        
        self.adaptive_themes[user_id]["adaptation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "theme": theme_name,
            "context": context
        })
    
    def _validate_theme_structure(self, theme_config):
        """Validate theme configuration structure."""
        required_fields = ["name", "description"]
        
        for field in required_fields:
            if field not in theme_config:
                return False
        
        return True
    
    def _calculate_theme_effectiveness(self, user_id, theme_name):
        """Calculate effectiveness rating for a theme."""
        usage_data = self.theme_usage_analytics[user_id].get(theme_name, {})
        
        if not usage_data:
            return 0.5
        
        # Simple effectiveness calculation
        satisfaction = usage_data.get("satisfaction_rating", 0.5)
        usage_frequency = min(1.0, usage_data.get("session_count", 0) / 100.0)
        
        return (satisfaction + usage_frequency) / 2
    
    def _calculate_adaptive_effectiveness(self, user_id):
        """Calculate effectiveness of adaptive theme system."""
        if user_id not in self.adaptive_themes:
            return 0.0
        
        history = self.adaptive_themes[user_id].get("adaptation_history", [])
        if len(history) < 5:
            return 0.5
        
        # Simple effectiveness based on adaptation frequency
        recent_adaptations = len([h for h in history[-20:] if h])
        return min(1.0, recent_adaptations / 20.0)
    
    def _generate_theme_optimizations(self, user_id):
        """Generate theme optimization suggestions."""
        optimizations = []
        
        usage_data = self.theme_usage_analytics[user_id]
        if not usage_data:
            return ["Enable theme analytics for personalized optimizations"]
        
        # Find underused themes
        total_usage = sum(data.get("total_time", 0) for data in usage_data.values())
        if total_usage > 0:
            for theme_name, data in usage_data.items():
                usage_percentage = data.get("total_time", 0) / total_usage
                if usage_percentage < 0.05 and data.get("satisfaction_rating", 0) > 0.7:
                    optimizations.append(f"Consider using '{theme_name}' more - it has high satisfaction but low usage")
        
        return optimizations
    
    def _analyze_usage_patterns(self, user_id):
        """Analyze user's theme usage patterns."""
        usage_data = self.theme_usage_analytics[user_id]
        
        if not usage_data:
            return {"message": "No usage data available"}
        
        patterns = {}
        
        # Find most used theme
        if usage_data:
            most_used = max(usage_data.items(), key=lambda x: x[1].get("total_time", 0))
            patterns["most_used_theme"] = most_used[0]
            patterns["usage_distribution"] = {
                theme: data.get("total_time", 0) 
                for theme, data in usage_data.items()
            }
        
        return patterns

class CustomizationManager:
    """Advanced user customization manager with AI-powered optimizations."""
    
    def __init__(self):
        self.user_customizations = defaultdict(dict)
        self.ui_preferences = {}
        self.adaptive_interfaces = {}
        self.customization_analytics = defaultdict(lambda: {
            "usage_frequency": {},
            "satisfaction": {},
            "performance_impact": {},
            "recent_changes": []
        })
        self.interface_optimizer = InterfaceOptimizer()
        self.accessibility_adapter = AccessibilityAdapter()
        self.layout_engine = AdaptiveLayoutEngine()
        
        # Load data
        self.load_customizations()
        self.load_ui_preferences()
        self.load_adaptive_data()
    
    def set_ui_preference(self, user_id, preference_key, value, context=None):
        """Set a UI preference for a user with context tracking."""
        if user_id not in self.ui_preferences:
            self.ui_preferences[user_id] = {}
        
        old_value = self.ui_preferences[user_id].get(preference_key)
        self.ui_preferences[user_id][preference_key] = value
        
        # Track preference change
        self._track_preference_change(user_id, preference_key, old_value, value, context)
        
        # Update adaptive interface
        self._update_adaptive_interface(user_id, preference_key, value)
        
        self.save_ui_preferences()
    
    def get_ui_preference(self, user_id, preference_key, default=None):
        """Get a UI preference for a user with smart defaults."""
        base_preference = self.ui_preferences.get(user_id, {}).get(preference_key, default)
        
        # Apply adaptive modifications
        if user_id in self.adaptive_interfaces:
            adaptive_mods = self.adaptive_interfaces[user_id].get("preference_adaptations", {})
            if preference_key in adaptive_mods:
                return adaptive_mods[preference_key]
        
        return base_preference
    
    def enable_adaptive_interface(self, user_id, enable=True):
        """Enable or disable adaptive interface for a user."""
        if enable:
            if user_id not in self.adaptive_interfaces:
                self.adaptive_interfaces[user_id] = {
                    "enabled": True,
                    "learning_mode": "active",
                    "adaptation_history": [],
                    "preference_adaptations": {},
                    "performance_metrics": {}
                }
        else:
            if user_id in self.adaptive_interfaces:
                self.adaptive_interfaces[user_id]["enabled"] = False
        
        self.save_adaptive_data()
        
    def add_shortcut(self, user_id, shortcut_key, action, smart_optimize=True):
        """Add a custom keyboard shortcut with smart optimization."""
        if "shortcuts" not in self.user_customizations[user_id]:
            self.user_customizations[user_id]["shortcuts"] = {}
        
        # Check for conflicts
        existing_shortcuts = self.user_customizations[user_id]["shortcuts"]
        if shortcut_key in existing_shortcuts:
            logging.warning(f"Shortcut {shortcut_key} already exists for user {user_id}")
            
        self.user_customizations[user_id]["shortcuts"][shortcut_key] = {
            "action": action,
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
            "effectiveness_score": 0.5,
            "context_tags": [],
            "auto_optimized": smart_optimize
        }
        
        # Apply smart optimizations
        if smart_optimize:
            self._optimize_shortcut(user_id, shortcut_key)
        
        self.save_customizations()
        
    def add_macro(self, user_id, macro_name, macro_steps, auto_optimize=True):
        """Add a custom macro with intelligent optimization."""
        if "macros" not in self.user_customizations[user_id]:
            self.user_customizations[user_id]["macros"] = {}
            
        self.user_customizations[user_id]["macros"][macro_name] = {
            "steps": macro_steps,
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
            "execution_time": 0.0,
            "success_rate": 1.0,
            "optimization_level": "standard",
            "tags": [],
            "dependencies": []
        }
        
        # Auto-optimize macro steps
        if auto_optimize:
            optimized_steps = self._optimize_macro_steps(macro_steps)
            self.user_customizations[user_id]["macros"][macro_name]["steps"] = optimized_steps
            self.user_customizations[user_id]["macros"][macro_name]["optimization_level"] = "optimized"
        
        self.save_customizations()
        
    def add_custom_command(self, user_id, command_name, command_config, enable_ai_enhancement=True):
        """Add a custom voice/text command with AI enhancements."""
        if "custom_commands" not in self.user_customizations[user_id]:
            self.user_customizations[user_id]["custom_commands"] = {}
            
        enhanced_config = command_config.copy()
        
        # AI enhancements
        if enable_ai_enhancement:
            enhanced_config.update({
                "ai_suggestions": True,
                "context_awareness": True,
                "adaptive_parameters": True,
                "learning_enabled": True
            })
            
        self.user_customizations[user_id]["custom_commands"][command_name] = {
            "config": enhanced_config,
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
            "accuracy_score": 0.9,
            "response_time": 0.0,
            "user_satisfaction": 0.8,
            "context_tags": [],
            "alternatives": []
        }
        
        self.save_customizations()
        
    def get_user_customizations(self, user_id):
        """Get all customizations for a user."""
        return self.user_customizations[user_id].copy()
        
    def execute_shortcut(self, user_id, shortcut_key):
        """Execute a custom shortcut."""
        shortcuts = self.user_customizations[user_id].get("shortcuts", {})
        
        if shortcut_key in shortcuts:
            shortcut = shortcuts[shortcut_key]
            shortcut["usage_count"] += 1
            
            # Return action to be executed
            return shortcut["action"]
            
        return None
        
    def execute_macro(self, user_id, macro_name):
        """Execute a custom macro."""
        macros = self.user_customizations[user_id].get("macros", {})
        
        if macro_name in macros:
            macro = macros[macro_name]
            macro["usage_count"] += 1
            
            # Return steps to be executed
            return macro["steps"]
            
        return None
    
    def get_interface_recommendations(self, user_id):
        """Get personalized interface recommendations."""
        recommendations = []
        
        # Analyze usage patterns
        usage_patterns = self._analyze_customization_patterns(user_id)
        
        # Check for accessibility needs
        accessibility_needs = self.accessibility_adapter.assess_needs(user_id)
        
        if accessibility_needs:
            recommendations.extend(accessibility_needs)
        
        # Performance-based recommendations
        performance_recs = self.interface_optimizer.get_performance_recommendations(user_id)
        recommendations.extend(performance_recs)
        
        # Layout optimization recommendations
        layout_recs = self.layout_engine.get_layout_recommendations(user_id)
        recommendations.extend(layout_recs)
        
        return recommendations
    
    def get_customization_analytics(self, user_id):
        """Get comprehensive analytics for user customizations."""
        if user_id not in self.customization_analytics:
            return {"message": "No analytics data available"}
        
        analytics = self.customization_analytics[user_id]
        customizations = self.user_customizations[user_id]
        
        return {
            "total_shortcuts": len(customizations.get("shortcuts", {})),
            "total_macros": len(customizations.get("macros", {})),
            "total_commands": len(customizations.get("custom_commands", {})),
            "most_used_shortcut": self._get_most_used(customizations.get("shortcuts", {})),
            "most_used_macro": self._get_most_used(customizations.get("macros", {})),
            "usage_frequency": analytics.get("usage_frequency", {}),
            "satisfaction_metrics": analytics.get("satisfaction", {}),
            "performance_impact": analytics.get("performance_impact", {}),
            "recent_changes": analytics.get("recent_changes", [])
        }
    
    def optimize_all_customizations(self, user_id):
        """Optimize all customizations for a user."""
        optimizations_applied = []
        
        # Optimize shortcuts
        shortcuts = self.user_customizations[user_id].get("shortcuts", {})
        for shortcut_key in shortcuts:
            if self._optimize_shortcut(user_id, shortcut_key):
                optimizations_applied.append(f"Optimized shortcut: {shortcut_key}")
        
        # Optimize macros
        macros = self.user_customizations[user_id].get("macros", {})
        for macro_name in macros:
            if self._optimize_macro(user_id, macro_name):
                optimizations_applied.append(f"Optimized macro: {macro_name}")
        
        # Optimize commands
        commands = self.user_customizations[user_id].get("custom_commands", {})
        for command_name in commands:
            if self._optimize_command(user_id, command_name):
                optimizations_applied.append(f"Optimized command: {command_name}")
        
        return optimizations_applied
    
    def export_user_customizations(self, user_id):
        """Export complete user customization profile."""
        profile = {
            "user_id": user_id,
            "customizations": self.user_customizations[user_id].copy(),
            "ui_preferences": self.ui_preferences.get(user_id, {}),
            "adaptive_settings": self.adaptive_interfaces.get(user_id, {}),
            "analytics": self.customization_analytics[user_id],
            "export_timestamp": datetime.now().isoformat(),
            "version": "2.0"
        }
        
        return profile
    
    def import_user_customizations(self, profile_data):
        """Import user customization profile."""
        user_id = profile_data.get("user_id")
        if not user_id:
            raise ValueError("Invalid profile data - missing user_id")
        
        # Import customizations
        if "customizations" in profile_data:
            self.user_customizations[user_id] = profile_data["customizations"]
        
        # Import UI preferences
        if "ui_preferences" in profile_data:
            if user_id not in self.ui_preferences:
                self.ui_preferences[user_id] = {}
            self.ui_preferences[user_id].update(profile_data["ui_preferences"])
        
        # Import adaptive settings
        if "adaptive_settings" in profile_data:
            self.adaptive_interfaces[user_id] = profile_data["adaptive_settings"]
        
        # Import analytics
        if "analytics" in profile_data:
            self.customization_analytics[user_id] = profile_data["analytics"]
        
        # Save all data
        self.save_customizations()
        self.save_ui_preferences()
        self.save_adaptive_data()
    
    def save_customizations(self):
        """Save all customization data."""
        try:
            with open("user_customizations.json", 'w') as f:
                json.dump(dict(self.user_customizations), f, indent=2)
        except Exception as e:
            logging.error(f"Error saving customizations: {e}")
    
    def load_customizations(self):
        """Load customization data."""
        try:
            if os.path.exists("user_customizations.json"):
                with open("user_customizations.json", 'r') as f:
                    data = json.load(f)
                    self.user_customizations = defaultdict(dict, data)
        except Exception as e:
            logging.error(f"Error loading customizations: {e}")
    
    def save_ui_preferences(self):
        """Save UI preferences."""
        try:
            with open("ui_preferences.json", 'w') as f:
                json.dump(self.ui_preferences, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving UI preferences: {e}")
    
    def load_ui_preferences(self):
        """Load UI preferences."""
        try:
            if os.path.exists("ui_preferences.json"):
                with open("ui_preferences.json", 'r') as f:
                    self.ui_preferences = json.load(f)
        except Exception as e:
            logging.error(f"Error loading UI preferences: {e}")
    
    def save_adaptive_data(self):
        """Save adaptive interface data."""
        try:
            with open("adaptive_interfaces.json", 'w') as f:
                json.dump(self.adaptive_interfaces, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving adaptive data: {e}")
    
    def load_adaptive_data(self):
        """Load adaptive interface data."""
        try:
            if os.path.exists("adaptive_interfaces.json"):
                with open("adaptive_interfaces.json", 'r') as f:
                    self.adaptive_interfaces = json.load(f)
        except Exception as e:
            logging.error(f"Error loading adaptive data: {e}")
    
    # Helper methods for CustomizationManager
    def _track_preference_change(self, user_id, preference_key, old_value, new_value, context):
        """Track preference changes for analytics."""
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "preference": preference_key,
            "old_value": old_value,
            "new_value": new_value,
            "context": context
        }
        
        self.customization_analytics[user_id]["recent_changes"].append(change_record)
        
        # Keep only last 50 changes
        if len(self.customization_analytics[user_id]["recent_changes"]) > 50:
            self.customization_analytics[user_id]["recent_changes"] = \
                self.customization_analytics[user_id]["recent_changes"][-50:]
    
    def _update_adaptive_interface(self, user_id, preference_key, value):
        """Update adaptive interface based on preference change."""
        if user_id not in self.adaptive_interfaces or not self.adaptive_interfaces[user_id].get("enabled"):
            return
        
        # Simple adaptive logic
        if preference_key == "theme" and value == "dark":
            self.adaptive_interfaces[user_id]["preference_adaptations"]["ui.high_contrast"] = True
    
    def _optimize_shortcut(self, user_id, shortcut_key):
        """Optimize a specific shortcut."""
        shortcuts = self.user_customizations[user_id].get("shortcuts", {})
        if shortcut_key not in shortcuts:
            return False
        
        shortcut = shortcuts[shortcut_key]
        
        # Simple optimization: suggest better key combinations
        if shortcut["usage_count"] > 10 and len(shortcut_key) > 3:
            # Suggest shorter combination
            shortcut["optimization_suggestion"] = f"Consider shorter key: Ctrl+{shortcut_key[-1]}"
            return True
        
        return False
    
    def _optimize_macro_steps(self, macro_steps):
        """Optimize macro steps for efficiency."""
        # Simple optimization: remove duplicate steps
        optimized_steps = []
        seen_steps = set()
        
        for step in macro_steps:
            step_key = str(step)
            if step_key not in seen_steps:
                optimized_steps.append(step)
                seen_steps.add(step_key)
        
        return optimized_steps
    
    def _optimize_macro(self, user_id, macro_name):
        """Optimize a specific macro."""
        macros = self.user_customizations[user_id].get("macros", {})
        if macro_name not in macros:
            return False
        
        macro = macros[macro_name]
        original_steps = len(macro["steps"])
        
        # Re-optimize steps
        optimized_steps = self._optimize_macro_steps(macro["steps"])
        
        if len(optimized_steps) < original_steps:
            macro["steps"] = optimized_steps
            macro["optimization_level"] = "highly_optimized"
            return True
        
        return False
    
    def _optimize_command(self, user_id, command_name):
        """Optimize a specific command."""
        commands = self.user_customizations[user_id].get("custom_commands", {})
        if command_name not in commands:
            return False
        
        command = commands[command_name]
        
        # Simple optimization: enable AI features if not already enabled
        if not command["config"].get("ai_suggestions", False):
            command["config"]["ai_suggestions"] = True
            command["config"]["context_awareness"] = True
            return True
        
        return False
    
    def _get_most_used(self, items_dict):
        """Get the most used item from a dictionary."""
        if not items_dict:
            return None
        
        most_used = max(items_dict.items(), key=lambda x: x[1].get("usage_count", 0))
        return most_used[0] if most_used[1].get("usage_count", 0) > 0 else None
    
    def _analyze_customization_patterns(self, user_id):
        """Analyze user's customization usage patterns."""
        if user_id not in self.customization_analytics:
            return {}
        
        analytics = self.customization_analytics[user_id]
        customizations = self.user_customizations[user_id]
        
        patterns = {
            "total_customizations": (
                len(customizations.get("shortcuts", {})) +
                len(customizations.get("macros", {})) +
                len(customizations.get("custom_commands", {}))
            ),
            "customization_types": {
                "shortcuts": len(customizations.get("shortcuts", {})),
                "macros": len(customizations.get("macros", {})),
                "commands": len(customizations.get("custom_commands", {}))
            },
            "usage_frequency": analytics.get("usage_frequency", {}),
            "recent_activity": len(analytics.get("recent_changes", []))
        }
        
        return patterns


# Helper classes for advanced theme management
class ThemeRecommendationEngine:
    """AI-powered theme recommendation system."""
    
    def __init__(self):
        self.recommendation_models = {}
        self.context_patterns = defaultdict(list)
        self.user_preferences = defaultdict(dict)
        
    def get_recommendations(self, user_id, usage_analytics, context=None):
        """Get personalized theme recommendations."""
        recommendations = []
        
        # Context-based recommendations
        if context:
            contextual_themes = self._get_contextual_recommendations(context)
            recommendations.extend(contextual_themes)
        
        # Usage pattern recommendations
        if usage_analytics:
            pattern_themes = self._get_pattern_based_recommendations(user_id, usage_analytics)
            recommendations.extend(pattern_themes)
        
        # Time-based recommendations
        time_themes = self._get_time_based_recommendations()
        recommendations.extend(time_themes)
        
        # Remove duplicates and rank
        unique_recommendations = []
        seen_themes = set()
        
        for rec in recommendations:
            if rec["theme"] not in seen_themes:
                unique_recommendations.append(rec)
                seen_themes.add(rec["theme"])
        
        # Sort by confidence score
        return sorted(unique_recommendations, key=lambda x: x["confidence"], reverse=True)[:5]
    
    def _get_contextual_recommendations(self, context):
        """Get recommendations based on current context."""
        recommendations = []
        
        # Time-based context
        current_hour = datetime.now().hour
        
        if 6 <= current_hour < 18:  # Daytime
            recommendations.append({
                "theme": "light",
                "reason": "Optimized for daytime use",
                "confidence": 0.8,
                "category": "time_based"
            })
        else:  # Evening/Night
            recommendations.append({
                "theme": "dark",
                "reason": "Reduces eye strain in low light",
                "confidence": 0.9,
                "category": "time_based"
            })
        
        # Activity-based context
        activity = context.get("activity", "")
        if activity == "work" or activity == "study":
            recommendations.append({
                "theme": "focus",
                "reason": "Minimizes distractions for productivity",
                "confidence": 0.85,
                "category": "activity_based"
            })
        elif activity == "gaming" or activity == "entertainment":
            recommendations.append({
                "theme": "cyberpunk",
                "reason": "Enhanced visual experience for gaming",
                "confidence": 0.7,
                "category": "activity_based"
            })
        
        return recommendations
    
    def _get_pattern_based_recommendations(self, user_id, analytics):
        """Get recommendations based on user patterns."""
        recommendations = []
        
        # Find most used themes
        if analytics:
            most_used = max(analytics.items(), key=lambda x: x[1].get("total_time", 0))
            
            if most_used[1].get("satisfaction_rating", 0) < 0.6:
                # User seems unsatisfied with current theme
                recommendations.append({
                    "theme": "adaptive_auto",
                    "reason": "AI-powered adaptation for better experience",
                    "confidence": 0.75,
                    "category": "improvement"
                })
        
        return recommendations
    
    def _get_time_based_recommendations(self):
        """Get recommendations based on time of day."""
        current_hour = datetime.now().hour
        recommendations = []
        
        if current_hour < 6 or current_hour > 22:  # Late night/early morning
            recommendations.append({
                "theme": "dark",
                "reason": "Gentle on eyes during late hours",
                "confidence": 0.9,
                "category": "circadian"
            })
        
        return recommendations


class ColorHarmonyAnalyzer:
    """Analyze and optimize color combinations for themes."""
    
    def __init__(self):
        self.color_blind_adaptations = {
            "protanopia": {"red_adjustment": 0.7, "green_boost": 1.2},
            "deuteranopia": {"green_adjustment": 0.8, "red_boost": 1.1},
            "tritanopia": {"blue_adjustment": 0.6, "yellow_boost": 1.3}
        }
    
    def adapt_for_color_blindness(self, theme):
        """Adapt theme colors for color blindness."""
        adapted_theme = theme.copy()
        
        # Apply general color-blind friendly adaptations
        if "colors" in adapted_theme:
            colors = adapted_theme["colors"]
            
            # Increase contrast
            colors["primary"] = self._increase_contrast(colors.get("primary", "#007ACC"))
            colors["accent"] = self._increase_contrast(colors.get("accent", "#FF6B6B"))
            
            # Use patterns/textures as additional visual cues
            adapted_theme["accessibility.pattern_support"] = True
            adapted_theme["accessibility.texture_cues"] = True
        
        return adapted_theme
    
    def _increase_contrast(self, color):
        """Increase contrast of a color for better accessibility."""
        # Simplified contrast enhancement
        # In a real implementation, this would use proper color space calculations
        if color.startswith("#"):
            # Convert hex to RGB, adjust, and convert back
            # This is a placeholder - real implementation would be more sophisticated
            return color
        return color
    
    def analyze_color_harmony(self, theme):
        """Analyze color harmony and suggest improvements."""
        if "colors" not in theme:
            return {"score": 0, "suggestions": ["Add color palette to theme"]}
        
        colors = theme["colors"]
        
        # Calculate harmony score (simplified)
        harmony_score = 0.8  # Placeholder calculation
        
        suggestions = []
        if harmony_score < 0.6:
            suggestions.append("Consider adjusting color relationships for better harmony")
        
        return {
            "harmony_score": harmony_score,
            "accessibility_score": self._calculate_accessibility_score(colors),
            "suggestions": suggestions
        }
    
    def _calculate_accessibility_score(self, colors):
        """Calculate accessibility score for color combination."""
        # Simplified accessibility scoring
        # Real implementation would check WCAG contrast ratios
        return 0.8


class AccessibilityChecker:
    """Check and validate theme accessibility compliance."""
    
    def __init__(self):
        self.wcag_guidelines = {
            "contrast_ratio_normal": 4.5,
            "contrast_ratio_large": 3.0,
            "motion_threshold": 0.3,
            "flash_threshold": 3.0
        }
    
    def evaluate_theme(self, theme):
        """Evaluate theme for accessibility compliance."""
        score = 1.0
        issues = []
        
        # Check high contrast mode
        if not theme.get("accessibility.high_contrast", False):
            if theme.get("appearance.theme") == "dark":
                score *= 0.9
            else:
                score *= 0.95
        
        # Check animation settings
        if theme.get("appearance.animations_enabled", True):
            if not theme.get("accessibility.reduced_motion", False):
                score *= 0.8
                issues.append("Consider reduced motion option for accessibility")
        
        # Check text size
        if not theme.get("accessibility.large_text", False):
            score *= 0.9
            issues.append("Large text option not available")
        
        return {
            "accessibility_score": score,
            "issues": issues,
            "wcag_compliance": "AA" if score > 0.8 else "A" if score > 0.6 else "Non-compliant"
        }


class InterfaceOptimizer:
    """Optimize interface performance and user experience."""
    
    def __init__(self):
        self.optimization_rules = {
            "performance": {
                "disable_animations_low_performance": True,
                "reduce_transparency_mobile": True,
                "optimize_for_touch": True
            },
            "accessibility": {
                "increase_contrast_vision_impaired": True,
                "reduce_motion_sensitive": True,
                "enhance_focus_indicators": True
            }
        }
    
    def optimize_settings(self, user_id, current_settings):
        """Optimize settings based on user context and performance."""
        optimized = current_settings.copy()
        
        # Simple optimization logic
        if self._detect_low_performance():
            optimized["appearance.animations_enabled"] = False
            optimized["appearance.particle_effects"] = False
            optimized["performance.gpu_acceleration"] = False
        
        return optimized
    
    def get_performance_recommendations(self, user_id):
        """Get performance-based recommendations."""
        recommendations = []
        
        if self._detect_low_performance():
            recommendations.append("Disable animations to improve performance")
            recommendations.append("Reduce visual effects for better responsiveness")
        
        return recommendations
    
    def optimize_for_performance(self, user_id, current_settings):
        """Optimize all settings for maximum performance."""
        performance_settings = {
            "appearance.animations_enabled": False,
            "appearance.particle_effects": False,
            "appearance.blur_effects": False,
            "appearance.transparency": 1.0,
            "performance.gpu_acceleration": False
        }
        
        return performance_settings
    
    def _detect_low_performance(self):
        """Simple performance detection."""
        # In a real implementation, this would check system resources
        return False


class AccessibilityAdapter:
    """Adapt interface for accessibility needs."""
    
    def __init__(self):
        self.accessibility_profiles = {
            "vision_impaired": {
                "high_contrast": True,
                "large_text": True,
                "screen_reader_compatible": True
            },
            "motor_impaired": {
                "large_click_targets": True,
                "reduced_precision_required": True,
                "voice_control_optimized": True
            },
            "cognitive_support": {
                "simplified_interface": True,
                "consistent_navigation": True,
                "clear_feedback": True
            }
        }
    
    def assess_needs(self, user_id):
        """Assess user accessibility needs."""
        # This would typically analyze user behavior or explicit preferences
        recommendations = []
        
        # Placeholder logic
        recommendations.append("Enable high contrast mode for better visibility")
        recommendations.append("Consider larger text size for readability")
        
        return recommendations
    
    def generate_adaptations(self, accessibility_needs):
        """Generate specific adaptations based on needs."""
        adaptations = {}
        
        if "vision_impaired" in accessibility_needs:
            adaptations.update(self.accessibility_profiles["vision_impaired"])
        
        if "motor_impaired" in accessibility_needs:
            adaptations.update(self.accessibility_profiles["motor_impaired"])
        
        if "cognitive_support" in accessibility_needs:
            adaptations.update(self.accessibility_profiles["cognitive_support"])
        
        return adaptations


class AdaptiveLayoutEngine:
    """Manage adaptive and responsive layouts."""
    
    def __init__(self):
        self.layout_templates = {
            "compact": {"sidebar_width": 200, "content_padding": 10},
            "comfortable": {"sidebar_width": 250, "content_padding": 20},
            "spacious": {"sidebar_width": 300, "content_padding": 30}
        }
    
    def get_layout_recommendations(self, user_id):
        """Get layout recommendations based on usage patterns."""
        recommendations = []
        
        # Simple recommendations
        recommendations.append("Consider compact layout for smaller screens")
        recommendations.append("Use comfortable spacing for extended work sessions")
        
        return recommendations
    
    def adapt_layout(self, user_id, screen_size, usage_context):
        """Adapt layout based on context."""
        if screen_size == "small":
            return self.layout_templates["compact"]
        elif usage_context == "extended_work":
            return self.layout_templates["comfortable"]
        else:
            return self.layout_templates["spacious"]


class AdaptivePreferenceEngine:
    """
    AI-powered adaptive preference system that learns from user behavior
    and automatically optimizes settings for improved user experience.
    """
    
    def __init__(self):
        self.adaptation_rules = {}
        self.learning_models = {}
        self.user_patterns = defaultdict(dict)
        self.adaptation_history = defaultdict(list)
        self.performance_metrics = defaultdict(dict)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7
        self.confidence_threshold = 0.8
        
        logging.info("Adaptive Preference Engine initialized")
    
    def analyze_usage_patterns(self, user_id: str, interaction_data: List[Dict]) -> Dict[str, Any]:
        """Analyze user interaction patterns to identify preference optimization opportunities."""
        if not interaction_data:
            return {"patterns": [], "recommendations": []}
        
        patterns = []
        recommendations = []
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(interaction_data)
        patterns.extend(temporal_patterns)
        
        # Analyze contextual patterns
        contextual_patterns = self._analyze_contextual_patterns(interaction_data)
        patterns.extend(contextual_patterns)
        
        # Generate recommendations based on patterns
        for pattern in patterns:
            if pattern["confidence"] > self.confidence_threshold:
                rec = self._pattern_to_recommendation(pattern)
                if rec:
                    recommendations.append(rec)
        
        return {
            "patterns": patterns,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def adapt_preferences_automatically(self, user_id: str, context: Dict[str, Any]) -> List[AdaptiveRecommendation]:
        """Automatically adapt preferences based on learned patterns."""
        adaptations = []
        
        if user_id not in self.user_patterns:
            return adaptations
        
        patterns = self.user_patterns[user_id]
        
        # Check for adaptation opportunities
        for pattern_type, pattern_data in patterns.items():
            if pattern_data.get("confidence", 0) > self.adaptation_threshold:
                adaptation = self._create_adaptation(user_id, pattern_type, pattern_data, context)
                if adaptation:
                    adaptations.append(adaptation)
        
        return adaptations
    
    def learn_from_feedback(self, user_id: str, recommendation: AdaptiveRecommendation, 
                           accepted: bool, satisfaction_change: float):
        """Learn from user feedback to improve future recommendations."""
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "recommendation": asdict(recommendation),
            "accepted": accepted,
            "satisfaction_change": satisfaction_change,
            "context": {}
        }
        
        self.adaptation_history[user_id].append(feedback_data)
        
        # Update learning models
        self._update_learning_models(user_id, feedback_data)
        
        logging.info(f"Learned from feedback for user {user_id}: accepted={accepted}, satisfaction_change={satisfaction_change}")
    
    def _analyze_temporal_patterns(self, interaction_data: List[Dict]) -> List[Dict]:
        """Analyze time-based usage patterns."""
        patterns = []
        
        # Group interactions by hour
        hourly_usage = defaultdict(list)
        for interaction in interaction_data:
            if "timestamp" in interaction:
                hour = datetime.fromisoformat(interaction["timestamp"]).hour
                hourly_usage[hour].append(interaction)
        
        # Find peak usage hours
        if hourly_usage:
            peak_hours = sorted(hourly_usage.items(), key=lambda x: len(x[1]), reverse=True)[:3]
            
            for hour, interactions in peak_hours:
                if len(interactions) > 5:  # Minimum threshold
                    patterns.append({
                        "type": "temporal",
                        "pattern": "peak_usage",
                        "hour": hour,
                        "frequency": len(interactions),
                        "confidence": min(1.0, len(interactions) / 20.0),
                        "description": f"High activity at {hour}:00"
                    })
        
        return patterns
    
    def _analyze_contextual_patterns(self, interaction_data: List[Dict]) -> List[Dict]:
        """Analyze context-based usage patterns."""
        patterns = []
        
        # Group by context
        context_groups = defaultdict(list)
        for interaction in interaction_data:
            context = interaction.get("context", {})
            activity = context.get("activity", "unknown")
            context_groups[activity].append(interaction)
        
        # Analyze each context
        for activity, interactions in context_groups.items():
            if len(interactions) > 3:
                # Analyze performance in this context
                avg_satisfaction = np.mean([i.get("satisfaction", 0.5) for i in interactions]) if np else 0.5
                
                patterns.append({
                    "type": "contextual",
                    "pattern": "activity_preference",
                    "activity": activity,
                    "frequency": len(interactions),
                    "avg_satisfaction": avg_satisfaction,
                    "confidence": min(1.0, len(interactions) / 10.0),
                    "description": f"Usage pattern for {activity}"
                })
        
        return patterns
    
    def _pattern_to_recommendation(self, pattern: Dict) -> Optional[AdaptiveRecommendation]:
        """Convert a usage pattern to a preference recommendation."""
        if pattern["type"] == "temporal":
            if pattern["pattern"] == "peak_usage":
                hour = pattern["hour"]
                if 18 <= hour <= 23:  # Evening hours
                    return AdaptiveRecommendation(
                        preference_path="appearance.theme",
                        recommended_value="dark",
                        confidence=pattern["confidence"],
                        reasoning=f"Dark theme reduces eye strain during evening peak usage at {hour}:00",
                        expected_benefit="Reduced eye strain and better visibility",
                        category="comfort",
                        priority=7
                    )
                elif 6 <= hour <= 12:  # Morning hours
                    return AdaptiveRecommendation(
                        preference_path="appearance.theme",
                        recommended_value="light",
                        confidence=pattern["confidence"],
                        reasoning=f"Light theme is optimal for morning usage at {hour}:00",
                        expected_benefit="Better visibility in daylight conditions",
                        category="comfort",
                        priority=6
                    )
        
        elif pattern["type"] == "contextual":
            if pattern["pattern"] == "activity_preference":
                activity = pattern["activity"]
                if activity == "work" and pattern["avg_satisfaction"] < 0.6:
                    return AdaptiveRecommendation(
                        preference_path="appearance.theme",
                        recommended_value="focus",
                        confidence=pattern["confidence"],
                        reasoning="Focus theme optimizes interface for work activities",
                        expected_benefit="Reduced distractions and improved productivity",
                        category="productivity",
                        priority=8
                    )
        
        return None
    
    def _create_adaptation(self, user_id: str, pattern_type: str, pattern_data: Dict, 
                          context: Dict) -> Optional[AdaptiveRecommendation]:
        """Create an adaptation based on learned patterns."""
        # This would use more sophisticated ML models in a full implementation
        if pattern_type == "theme_switching" and pattern_data.get("confidence", 0) > 0.8:
            return AdaptiveRecommendation(
                preference_path="appearance.theme",
                recommended_value="adaptive_auto",
                confidence=pattern_data["confidence"],
                reasoning="Enable automatic theme switching based on learned patterns",
                expected_benefit="Seamless theme adaptation to usage context",
                category="automation",
                priority=9
            )
        
        return None
    
    def _update_learning_models(self, user_id: str, feedback_data: Dict):
        """Update machine learning models based on feedback."""
        if user_id not in self.learning_models:
            self.learning_models[user_id] = {
                "preference_weights": defaultdict(float),
                "satisfaction_predictors": {},
                "adaptation_success_rate": 0.5
            }
        
        model = self.learning_models[user_id]
        
        # Update adaptation success rate
        accepted = feedback_data["accepted"]
        current_rate = model["adaptation_success_rate"]
        model["adaptation_success_rate"] = current_rate + self.learning_rate * (float(accepted) - current_rate)
        
        # Update preference weights based on satisfaction change
        satisfaction_change = feedback_data["satisfaction_change"]
        recommendation = feedback_data["recommendation"]
        preference_path = recommendation["preference_path"]
        
        model["preference_weights"][preference_path] += self.learning_rate * satisfaction_change
        
        logging.debug(f"Updated learning model for {user_id}: success_rate={model['adaptation_success_rate']}")


class PreferenceMLAnalyzer:
    """
    Machine learning analyzer for preference patterns, user behavior prediction,
    and intelligent optimization recommendations using advanced ML techniques.
    """
    
    def __init__(self):
        self.feature_extractors = {}
        self.prediction_models = {}
        self.clustering_models = {}
        self.anomaly_detectors = {}
        
        # ML pipeline components
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = defaultdict(dict)
        
        logging.info("Preference ML Analyzer initialized")
    
    def analyze_preference_patterns(self, user_id: str, preference_history: List[PreferenceChange]) -> Dict[str, Any]:
        """Perform comprehensive ML analysis of preference patterns."""
        if not preference_history or not SKLEARN_AVAILABLE:
            return {"error": "Insufficient data or ML libraries unavailable"}
        
        analysis_results = {}
        
        try:
            # Extract features for ML analysis
            features = self._extract_preference_features(preference_history)
            
            if features.size == 0:
                return {"error": "Could not extract features from preference history"}
            
            # Clustering analysis
            analysis_results["clustering"] = self._perform_preference_clustering(features, preference_history)
            
            # Pattern recognition
            analysis_results["patterns"] = self._recognize_preference_patterns(features, preference_history)
            
            # Anomaly detection
            analysis_results["anomalies"] = self._detect_preference_anomalies(features, preference_history)
            
            # Prediction models
            analysis_results["predictions"] = self._build_preference_predictions(user_id, features, preference_history)
            
            # Feature importance analysis
            analysis_results["feature_importance"] = self._analyze_feature_importance(features, preference_history)
            
        except Exception as e:
            logging.error(f"ML analysis error for user {user_id}: {e}")
            analysis_results["error"] = str(e)
        
        return analysis_results
    
    def predict_preference_satisfaction(self, user_id: str, proposed_changes: Dict[str, Any]) -> Dict[str, float]:
        """Predict user satisfaction with proposed preference changes."""
        predictions = {}
        
        if user_id not in self.prediction_models:
            # Build initial model if not exists
            self._initialize_prediction_model(user_id)
        
        for preference_path, new_value in proposed_changes.items():
            try:
                # Extract features for this preference change
                change_features = self._extract_change_features(preference_path, new_value)
                
                # Predict satisfaction
                if user_id in self.prediction_models and preference_path in self.prediction_models[user_id]:
                    model = self.prediction_models[user_id][preference_path]
                    satisfaction_score = self._predict_with_model(model, change_features)
                else:
                    # Use heuristic prediction
                    satisfaction_score = self._heuristic_satisfaction_prediction(preference_path, new_value)
                
                predictions[preference_path] = satisfaction_score
                
            except Exception as e:
                logging.error(f"Prediction error for {preference_path}: {e}")
                predictions[preference_path] = 0.5  # Neutral prediction
        
        return predictions
    
    def optimize_preference_set(self, user_id: str, current_preferences: Dict[str, Any], 
                               optimization_goals: List[str]) -> Dict[str, Any]:
        """Use ML to optimize a complete set of preferences for specific goals."""
        optimized_preferences = current_preferences.copy()
        
        if not SKLEARN_AVAILABLE:
            return optimized_preferences
        
        try:
            # Define optimization strategies based on goals
            optimization_strategies = {
                "performance": self._optimize_for_performance,
                "accessibility": self._optimize_for_accessibility,
                "user_satisfaction": self._optimize_for_satisfaction,
                "energy_efficiency": self._optimize_for_energy,
                "productivity": self._optimize_for_productivity
            }
            
            # Apply each optimization strategy
            for goal in optimization_goals:
                if goal in optimization_strategies:
                    strategy = optimization_strategies[goal]
                    optimized_preferences = strategy(user_id, optimized_preferences)
            
            # Validate optimizations
            validation_results = self._validate_optimizations(user_id, current_preferences, optimized_preferences)
            
            return {
                "optimized_preferences": optimized_preferences,
                "validation": validation_results,
                "optimization_goals": optimization_goals,
                "confidence": validation_results.get("overall_confidence", 0.5)
            }
            
        except Exception as e:
            logging.error(f"Optimization error for user {user_id}: {e}")
            return {"error": str(e), "original_preferences": current_preferences}
    
    def _extract_preference_features(self, preference_history: List[PreferenceChange]) -> Any:
        """Extract ML features from preference change history."""
        if not preference_history or not np:
            return np.array([])
        
        features = []
        
        for change in preference_history:
            # Extract temporal features
            timestamp = datetime.fromisoformat(change.timestamp)
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Extract preference features
            preference_type = self._categorize_preference(change.preference_path)
            
            # Extract value change features
            value_change_magnitude = self._calculate_value_change_magnitude(change.old_value, change.new_value)
            
            # Extract context features
            context = change.context or {}
            context_complexity = len(context)
            
            feature_vector = [
                hour / 24.0,  # Normalized hour
                day_of_week / 7.0,  # Normalized day
                hash(preference_type) % 1000,  # Preference type encoding
                value_change_magnitude,
                context_complexity,
                change.satisfaction_impact or 0.0,
                change.performance_impact or 0.0
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _perform_preference_clustering(self, features: Any, history: List[PreferenceChange]) -> Dict[str, Any]:
        """Perform clustering analysis on preference patterns."""
        try:
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # K-means clustering
            n_clusters = min(5, len(features) // 3)  # Adaptive cluster count
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Analyze clusters
            cluster_analysis = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_changes = [history[j] for j, mask in enumerate(cluster_mask) if mask]
                
                cluster_analysis[f"cluster_{i}"] = {
                    "size": int(np.sum(cluster_mask)),
                    "avg_satisfaction_impact": float(np.mean([c.satisfaction_impact or 0 for c in cluster_changes])),
                    "common_preferences": self._get_common_preferences(cluster_changes),
                    "temporal_pattern": self._analyze_cluster_timing(cluster_changes),
                    "characteristics": self._characterize_preference_cluster(cluster_changes)
                }
            
            return {
                "n_clusters": n_clusters,
                "cluster_analysis": cluster_analysis,
                "cluster_labels": cluster_labels.tolist()
            }
            
        except Exception as e:
            logging.error(f"Clustering analysis error: {e}")
            return {"error": str(e)}
    
    def _recognize_preference_patterns(self, features: Any, history: List[PreferenceChange]) -> Dict[str, Any]:
        """Recognize patterns in preference changes using ML techniques."""
        patterns = {}
        
        try:
            # Sequential pattern analysis
            patterns["sequential"] = self._analyze_sequential_patterns(history)
            
            # Temporal pattern analysis
            patterns["temporal"] = self._analyze_temporal_preference_patterns(history)
            
            # Context-based pattern analysis
            patterns["contextual"] = self._analyze_contextual_preference_patterns(history)
            
            # Satisfaction pattern analysis
            patterns["satisfaction"] = self._analyze_satisfaction_patterns(history)
            
        except Exception as e:
            logging.error(f"Pattern recognition error: {e}")
            patterns["error"] = str(e)
        
        return patterns
    
    def _detect_preference_anomalies(self, features: Any, history: List[PreferenceChange]) -> Dict[str, Any]:
        """Detect anomalous preference changes using ML anomaly detection."""
        try:
            # Isolation Forest for anomaly detection
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = isolation_forest.fit_predict(features)
            
            # Identify anomalies
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            anomalies = []
            for idx in anomaly_indices:
                if idx < len(history):
                    change = history[idx]
                    anomalies.append({
                        "change": asdict(change),
                        "anomaly_score": float(isolation_forest.decision_function(features[idx:idx+1])[0]),
                        "potential_reasons": self._analyze_anomaly_reasons(change, history)
                    })
            
            return {
                "total_anomalies": len(anomalies),
                "anomaly_rate": len(anomalies) / len(history) if history else 0,
                "detected_anomalies": anomalies[:10],  # Top 10 anomalies
                "analysis_summary": self._summarize_anomalies(anomalies)
            }
            
        except Exception as e:
            logging.error(f"Anomaly detection error: {e}")
            return {"error": str(e)}
    
    def _build_preference_predictions(self, user_id: str, features: Any, 
                                    history: List[PreferenceChange]) -> Dict[str, Any]:
        """Build predictive models for future preference changes."""
        predictions = {}
        
        try:
            # Predict next likely preference changes
            predictions["next_changes"] = self._predict_next_preference_changes(features, history)
            
            # Predict satisfaction with current settings
            predictions["satisfaction_trends"] = self._predict_satisfaction_trends(history)
            
            # Predict optimal settings for different contexts
            predictions["contextual_optimizations"] = self._predict_contextual_optimizations(history)
            
        except Exception as e:
            logging.error(f"Prediction building error: {e}")
            predictions["error"] = str(e)
        
        return predictions
    
    def _analyze_feature_importance(self, features: Any, history: List[PreferenceChange]) -> Dict[str, Any]:
        """Analyze which features are most important for preference decisions."""
        importance_analysis = {}
        
        try:
            # Calculate feature correlations with satisfaction
            satisfaction_scores = [change.satisfaction_impact or 0.5 for change in history]
            
            if len(satisfaction_scores) == features.shape[0]:
                feature_names = ["hour", "day_of_week", "preference_type", "value_change", 
                               "context_complexity", "satisfaction_impact", "performance_impact"]
                
                correlations = {}
                for i, feature_name in enumerate(feature_names[:features.shape[1]]):
                    if np:
                        correlation = np.corrcoef(features[:, i], satisfaction_scores)[0, 1]
                        correlations[feature_name] = float(correlation) if not np.isnan(correlation) else 0.0
                
                importance_analysis["feature_correlations"] = correlations
                importance_analysis["most_important_features"] = sorted(
                    correlations.items(), key=lambda x: abs(x[1]), reverse=True
                )[:5]
            
        except Exception as e:
            logging.error(f"Feature importance analysis error: {e}")
            importance_analysis["error"] = str(e)
        
        return importance_analysis
    
    # Helper methods for ML analysis
    def _categorize_preference(self, preference_path: str) -> str:
        """Categorize preference by type."""
        if preference_path.startswith("appearance"):
            return "appearance"
        elif preference_path.startswith("voice"):
            return "voice"
        elif preference_path.startswith("gesture"):
            return "gesture"
        elif preference_path.startswith("performance"):
            return "performance"
        elif preference_path.startswith("accessibility"):
            return "accessibility"
        else:
            return "other"
    
    def _calculate_value_change_magnitude(self, old_value: Any, new_value: Any) -> float:
        """Calculate the magnitude of change between preference values."""
        try:
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                return abs(float(new_value) - float(old_value))
            elif isinstance(old_value, bool) and isinstance(new_value, bool):
                return 1.0 if old_value != new_value else 0.0
            elif isinstance(old_value, str) and isinstance(new_value, str):
                return 1.0 if old_value != new_value else 0.0
            else:
                return 0.5  # Unknown change type
        except:
            return 0.5


class PreferenceBehavioralTracker:
    """
    Advanced behavioral tracking for preference usage patterns, user satisfaction,
    and contextual preference optimization with real-time adaptation.
    """
    
    def __init__(self):
        self.behavior_patterns = defaultdict(dict)
        self.satisfaction_tracking = defaultdict(list)
        self.context_correlations = defaultdict(dict)
        self.usage_analytics = defaultdict(lambda: {
            "total_changes": 0,
            "successful_adaptations": 0,
            "satisfaction_scores": [],
            "context_usage": defaultdict(int),
            "preference_effectiveness": defaultdict(float)
        })
        
        # Real-time tracking
        self.active_sessions = defaultdict(dict)
        self.real_time_metrics = defaultdict(dict)
        
        logging.info("Preference Behavioral Tracker initialized")
    
    def track_preference_usage(self, user_id: str, preference_path: str, context: Dict[str, Any], 
                              satisfaction_score: float = None):
        """Track how preferences are used in different contexts."""
        usage_data = {
            "timestamp": datetime.now().isoformat(),
            "preference_path": preference_path,
            "context": context,
            "satisfaction_score": satisfaction_score,
            "session_id": context.get("session_id", "unknown")
        }
        
        # Update usage analytics
        analytics = self.usage_analytics[user_id]
        analytics["total_changes"] += 1
        
        if satisfaction_score is not None:
            analytics["satisfaction_scores"].append(satisfaction_score)
            
            # Track preference effectiveness
            current_effectiveness = analytics["preference_effectiveness"][preference_path]
            new_effectiveness = (current_effectiveness + satisfaction_score) / 2
            analytics["preference_effectiveness"][preference_path] = new_effectiveness
            
            if satisfaction_score > 0.7:
                analytics["successful_adaptations"] += 1
        
        # Update context usage
        activity = context.get("activity", "unknown")
        analytics["context_usage"][activity] += 1
        
        # Update behavior patterns
        self._update_behavior_patterns(user_id, usage_data)
        
        # Update real-time metrics
        self._update_real_time_metrics(user_id, usage_data)
        
        logging.debug(f"Tracked preference usage for {user_id}: {preference_path}")
    
    def analyze_behavioral_trends(self, user_id: str, time_window_days: int = 30) -> Dict[str, Any]:
        """Analyze behavioral trends in preference usage over time."""
        analytics = self.usage_analytics[user_id]
        
        if analytics["total_changes"] == 0:
            return {"message": "No behavioral data available"}
        
        # Calculate time-based metrics
        cutoff_time = datetime.now() - timedelta(days=time_window_days)
        
        trends = {
            "overall_metrics": {
                "total_preference_changes": analytics["total_changes"],
                "successful_adaptations": analytics["successful_adaptations"],
                "success_rate": analytics["successful_adaptations"] / max(analytics["total_changes"], 1),
                "average_satisfaction": np.mean(analytics["satisfaction_scores"]) if analytics["satisfaction_scores"] and np else 0.5
            },
            "context_analysis": self._analyze_context_usage(analytics["context_usage"]),
            "preference_effectiveness": dict(analytics["preference_effectiveness"]),
            "behavioral_insights": self._generate_behavioral_insights(user_id, analytics),
            "trend_predictions": self._predict_behavioral_trends(user_id, analytics)
        }
        
        return trends
    
    def get_contextual_recommendations(self, user_id: str, current_context: Dict[str, Any]) -> List[AdaptiveRecommendation]:
        """Get behavioral-based recommendations for current context."""
        recommendations = []
        
        if user_id not in self.behavior_patterns:
            return recommendations
        
        patterns = self.behavior_patterns[user_id]
        current_activity = current_context.get("activity", "unknown")
        
        # Find successful patterns for this context
        if current_activity in patterns:
            activity_patterns = patterns[current_activity]
            
            # Recommend high-performing preferences for this context
            for preference_path, effectiveness in activity_patterns.get("effective_preferences", {}).items():
                if effectiveness > 0.7:
                    recommendations.append(AdaptiveRecommendation(
                        preference_path=preference_path,
                        recommended_value=activity_patterns.get("optimal_values", {}).get(preference_path),
                        confidence=effectiveness,
                        reasoning=f"This preference has {effectiveness:.1%} effectiveness in {current_activity} context",
                        expected_benefit="Improved user experience based on behavioral patterns",
                        category="behavioral",
                        priority=int(effectiveness * 10)
                    ))
        
        # Sort by confidence and return top recommendations
        return sorted(recommendations, key=lambda x: x.confidence, reverse=True)[:5]
    
    def detect_usage_anomalies(self, user_id: str) -> Dict[str, Any]:
        """Detect anomalous patterns in preference usage behavior."""
        analytics = self.usage_analytics[user_id]
        
        if analytics["total_changes"] < 10:
            return {"message": "Insufficient data for anomaly detection"}
        
        anomalies = []
        
        # Detect sudden satisfaction drops
        satisfaction_scores = analytics["satisfaction_scores"]
        if len(satisfaction_scores) >= 10:
            recent_satisfaction = np.mean(satisfaction_scores[-5:]) if np else 0.5
            historical_satisfaction = np.mean(satisfaction_scores[:-5]) if np else 0.5
            
            if recent_satisfaction < historical_satisfaction * 0.7:
                anomalies.append({
                    "type": "satisfaction_drop",
                    "severity": "high",
                    "description": f"Satisfaction dropped from {historical_satisfaction:.2f} to {recent_satisfaction:.2f}",
                    "recommendation": "Review recent preference changes and consider reverting problematic settings"
                })
        
        # Detect unusual context usage
        context_usage = analytics["context_usage"]
        if context_usage:
            total_usage = sum(context_usage.values())
            for context, usage_count in context_usage.items():
                usage_percentage = usage_count / total_usage
                if usage_percentage > 0.8:  # One context dominates
                    anomalies.append({
                        "type": "context_dominance",
                        "severity": "medium",
                        "description": f"Context '{context}' represents {usage_percentage:.1%} of all usage",
                        "recommendation": "Consider optimizing preferences for diverse usage contexts"
                    })
        
        return {
            "total_anomalies": len(anomalies),
            "detected_anomalies": anomalies,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _update_behavior_patterns(self, user_id: str, usage_data: Dict[str, Any]):
        """Update behavioral patterns based on usage data."""
        patterns = self.behavior_patterns[user_id]
        
        activity = usage_data["context"].get("activity", "unknown")
        preference_path = usage_data["preference_path"]
        satisfaction = usage_data.get("satisfaction_score")
        
        # Initialize activity pattern if not exists
        if activity not in patterns:
            patterns[activity] = {
                "usage_count": 0,
                "effective_preferences": defaultdict(float),
                "optimal_values": {},
                "satisfaction_history": []
            }
        
        activity_pattern = patterns[activity]
        activity_pattern["usage_count"] += 1
        
        if satisfaction is not None:
            activity_pattern["satisfaction_history"].append(satisfaction)
            
            # Update preference effectiveness for this activity
            current_effectiveness = activity_pattern["effective_preferences"][preference_path]
            new_effectiveness = (current_effectiveness + satisfaction) / 2
            activity_pattern["effective_preferences"][preference_path] = new_effectiveness
    
    def _update_real_time_metrics(self, user_id: str, usage_data: Dict[str, Any]):
        """Update real-time behavioral metrics."""
        metrics = self.real_time_metrics[user_id]
        
        # Update recent activity
        metrics["last_activity"] = usage_data["timestamp"]
        metrics["recent_preferences"] = metrics.get("recent_preferences", [])
        metrics["recent_preferences"].append(usage_data["preference_path"])
        
        # Keep only last 10 preferences
        if len(metrics["recent_preferences"]) > 10:
            metrics["recent_preferences"] = metrics["recent_preferences"][-10:]
        
        # Update satisfaction trend
        if usage_data.get("satisfaction_score") is not None:
            metrics["recent_satisfaction"] = metrics.get("recent_satisfaction", [])
            metrics["recent_satisfaction"].append(usage_data["satisfaction_score"])
            
            if len(metrics["recent_satisfaction"]) > 5:
                metrics["recent_satisfaction"] = metrics["recent_satisfaction"][-5:]
            
            metrics["current_satisfaction_trend"] = np.mean(metrics["recent_satisfaction"]) if np else 0.5


class IntelligentRecommendationEngine:
    """
    Advanced AI-powered recommendation engine that combines behavioral analysis,
    machine learning insights, and contextual understanding to provide personalized
    preference optimization suggestions.
    """
    
    def __init__(self):
        self.recommendation_models = {}
        self.user_profiles = defaultdict(dict)
        self.contextual_preferences = defaultdict(dict)
        self.recommendation_history = defaultdict(list)
        
        # Recommendation scoring weights
        self.scoring_weights = {
            "behavioral_fit": 0.3,
            "satisfaction_prediction": 0.25,
            "contextual_relevance": 0.2,
            "performance_impact": 0.15,
            "novelty_factor": 0.1
        }
        
        logging.info("Intelligent Recommendation Engine initialized")
    
    def generate_comprehensive_recommendations(self, user_id: str, context: Dict[str, Any], 
                                             preference_history: List[PreferenceChange]) -> List[AdaptiveRecommendation]:
        """Generate comprehensive AI-powered preference recommendations."""
        recommendations = []
        
        # Behavioral recommendations
        behavioral_recs = self._generate_behavioral_recommendations(user_id, context, preference_history)
        recommendations.extend(behavioral_recs)
        
        # Context-aware recommendations
        contextual_recs = self._generate_contextual_recommendations(user_id, context)
        recommendations.extend(contextual_recs)
        
        # Performance optimization recommendations
        performance_recs = self._generate_performance_recommendations(user_id, context)
        recommendations.extend(performance_recs)
        
        # Accessibility recommendations
        accessibility_recs = self._generate_accessibility_recommendations(user_id, context)
        recommendations.extend(accessibility_recs)
        
        # Novelty and exploration recommendations
        novelty_recs = self._generate_novelty_recommendations(user_id, preference_history)
        recommendations.extend(novelty_recs)
        
        # Score and rank all recommendations
        scored_recommendations = self._score_recommendations(recommendations, user_id, context)
        
        # Remove duplicates and return top recommendations
        unique_recommendations = self._deduplicate_recommendations(scored_recommendations)
        
        return sorted(unique_recommendations, key=lambda x: x.confidence, reverse=True)[:10]
    
    def _generate_behavioral_recommendations(self, user_id: str, context: Dict[str, Any], 
                                           preference_history: List[PreferenceChange]) -> List[AdaptiveRecommendation]:
        """Generate recommendations based on behavioral analysis."""
        recommendations = []
        
        if not preference_history:
            return recommendations
        
        # Analyze successful preference changes
        successful_changes = [change for change in preference_history 
                            if change.satisfaction_impact and change.satisfaction_impact > 0.7]
        
        if successful_changes:
            # Find common patterns in successful changes
            preference_effectiveness = defaultdict(list)
            for change in successful_changes:
                preference_effectiveness[change.preference_path].append(change.satisfaction_impact)
            
            # Recommend highly effective preferences
            for preference_path, effectiveness_scores in preference_effectiveness.items():
                if len(effectiveness_scores) >= 2:  # Minimum sample size
                    avg_effectiveness = np.mean(effectiveness_scores) if np else 0.5
                    
                    if avg_effectiveness > 0.8:
                        recommendations.append(AdaptiveRecommendation(
                            preference_path=preference_path,
                            recommended_value=self._get_most_effective_value(preference_path, successful_changes),
                            confidence=avg_effectiveness,
                            reasoning=f"This preference has consistently high satisfaction ({avg_effectiveness:.1%})",
                            expected_benefit="Improved user experience based on historical success",
                            category="behavioral",
                            priority=8
                        ))
        
        return recommendations
    
    def _generate_contextual_recommendations(self, user_id: str, context: Dict[str, Any]) -> List[AdaptiveRecommendation]:
        """Generate context-aware recommendations."""
        recommendations = []
        
        activity = context.get("activity", "")
        time_of_day = datetime.now().hour
        environment = context.get("environment", "")
        
        # Time-based recommendations
        if 22 <= time_of_day or time_of_day <= 6:  # Night time
            recommendations.append(AdaptiveRecommendation(
                preference_path="appearance.theme",
                recommended_value="dark",
                confidence=0.85,
                reasoning="Dark theme reduces eye strain during nighttime usage",
                expected_benefit="Better comfort and reduced eye fatigue",
                category="contextual",
                priority=7
            ))
            
            recommendations.append(AdaptiveRecommendation(
                preference_path="voice.volume",
                recommended_value=0.6,
                confidence=0.75,
                reasoning="Lower volume is appropriate for nighttime use",
                expected_benefit="Reduced disturbance to others",
                category="contextual",
                priority=6
            ))
        
        # Activity-based recommendations
        if activity == "work" or activity == "study":
            recommendations.append(AdaptiveRecommendation(
                preference_path="notifications.do_not_disturb_hours",
                recommended_value=[time_of_day, min(time_of_day + 2, 23)],
                confidence=0.8,
                reasoning="Focus mode reduces distractions during work",
                expected_benefit="Improved concentration and productivity",
                category="contextual",
                priority=8
            ))
        
        elif activity == "entertainment" or activity == "gaming":
            recommendations.append(AdaptiveRecommendation(
                preference_path="appearance.particle_effects",
                recommended_value=True,
                confidence=0.7,
                reasoning="Enhanced visual effects improve entertainment experience",
                expected_benefit="More immersive and engaging interface",
                category="contextual",
                priority=6
            ))
        
        return recommendations
    
    def _generate_performance_recommendations(self, user_id: str, context: Dict[str, Any]) -> List[AdaptiveRecommendation]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check system performance indicators
        performance_mode = context.get("performance_mode", "balanced")
        
        if performance_mode == "low" or context.get("battery_low", False):
            recommendations.extend([
                AdaptiveRecommendation(
                    preference_path="appearance.animations_enabled",
                    recommended_value=False,
                    confidence=0.9,
                    reasoning="Disabling animations improves performance on low-power systems",
                    expected_benefit="Better system responsiveness and battery life",
                    category="performance",
                    priority=9
                ),
                AdaptiveRecommendation(
                    preference_path="appearance.particle_effects",
                    recommended_value=False,
                    confidence=0.85,
                    reasoning="Reducing visual effects conserves system resources",
                    expected_benefit="Improved performance and energy efficiency",
                    category="performance",
                    priority=8
                )
            ])
        
        return recommendations
    
    def _generate_accessibility_recommendations(self, user_id: str, context: Dict[str, Any]) -> List[AdaptiveRecommendation]:
        """Generate accessibility-focused recommendations."""
        recommendations = []
        
        # Check for accessibility needs indicators
        if context.get("low_vision", False) or context.get("vision_impaired", False):
            recommendations.extend([
                AdaptiveRecommendation(
                    preference_path="accessibility.high_contrast",
                    recommended_value=True,
                    confidence=0.95,
                    reasoning="High contrast improves visibility for users with vision impairments",
                    expected_benefit="Enhanced readability and interface clarity",
                    category="accessibility",
                    priority=10
                ),
                AdaptiveRecommendation(
                    preference_path="accessibility.large_text",
                    recommended_value=True,
                    confidence=0.9,
                    reasoning="Larger text is easier to read for users with vision difficulties",
                    expected_benefit="Improved text readability and reduced eye strain",
                    category="accessibility",
                    priority=9
                )
            ])
        
        if context.get("motor_impaired", False):
            recommendations.append(AdaptiveRecommendation(
                preference_path="gesture.gesture_timeout",
                recommended_value=5.0,
                confidence=0.85,
                reasoning="Extended gesture timeout accommodates users with motor difficulties",
                expected_benefit="More reliable gesture recognition and reduced frustration",
                category="accessibility",
                priority=8
            ))
        
        return recommendations
    
    def _generate_novelty_recommendations(self, user_id: str, preference_history: List[PreferenceChange]) -> List[AdaptiveRecommendation]:
        """Generate recommendations that introduce beneficial novelty."""
        recommendations = []
        
        if not preference_history:
            return recommendations
        
        # Find preferences that haven't been changed recently
        recent_changes = [change.preference_path for change in preference_history[-10:]]
        all_possible_preferences = [
            "appearance.theme", "appearance.color_scheme", "voice.speech_rate",
            "gesture.gesture_sensitivity", "performance.performance_mode"
        ]
        
        unexplored_preferences = [pref for pref in all_possible_preferences if pref not in recent_changes]
        
        # Recommend exploring new preferences
        for preference_path in unexplored_preferences[:3]:  # Limit to 3 novelty recommendations
            recommendations.append(AdaptiveRecommendation(
                preference_path=preference_path,
                recommended_value=self._suggest_novelty_value(preference_path),
                confidence=0.6,
                reasoning=f"Exploring {preference_path} might reveal new optimization opportunities",
                expected_benefit="Potential discovery of better settings and enhanced experience",
                category="exploration",
                priority=4
            ))
        
        return recommendations
    
    def _score_recommendations(self, recommendations: List[AdaptiveRecommendation], 
                             user_id: str, context: Dict[str, Any]) -> List[AdaptiveRecommendation]:
        """Score recommendations using multiple factors."""
        for rec in recommendations:
            # Calculate composite score
            behavioral_score = self._calculate_behavioral_fit_score(rec, user_id)
            satisfaction_score = self._predict_satisfaction_score(rec, user_id)
            contextual_score = self._calculate_contextual_relevance_score(rec, context)
            performance_score = self._calculate_performance_impact_score(rec)
            novelty_score = self._calculate_novelty_score(rec, user_id)
            
            # Weighted composite score
            composite_score = (
                self.scoring_weights["behavioral_fit"] * behavioral_score +
                self.scoring_weights["satisfaction_prediction"] * satisfaction_score +
                self.scoring_weights["contextual_relevance"] * contextual_score +
                self.scoring_weights["performance_impact"] * performance_score +
                self.scoring_weights["novelty_factor"] * novelty_score
            )
            
            # Update confidence with composite score
            rec.confidence = min(1.0, max(0.0, composite_score))
        
        return recommendations
    
    def _deduplicate_recommendations(self, recommendations: List[AdaptiveRecommendation]) -> List[AdaptiveRecommendation]:
        """Remove duplicate recommendations, keeping the highest confidence version."""
        seen_preferences = {}
        
        for rec in recommendations:
            if rec.preference_path not in seen_preferences or rec.confidence > seen_preferences[rec.preference_path].confidence:
                seen_preferences[rec.preference_path] = rec
        
        return list(seen_preferences.values())
    
    # Helper methods for recommendation scoring
    def _calculate_behavioral_fit_score(self, recommendation: AdaptiveRecommendation, user_id: str) -> float:
        """Calculate how well the recommendation fits user's behavioral patterns."""
        # Simplified behavioral fit calculation
        return 0.7  # Placeholder - would use actual behavioral analysis
    
    def _predict_satisfaction_score(self, recommendation: AdaptiveRecommendation, user_id: str) -> float:
        """Predict user satisfaction with the recommendation."""
        # Use recommendation confidence as baseline satisfaction prediction
        return recommendation.confidence
    
    def _calculate_contextual_relevance_score(self, recommendation: AdaptiveRecommendation, context: Dict[str, Any]) -> float:
        """Calculate how relevant the recommendation is to current context."""
        if recommendation.category == "contextual":
            return 0.9
        elif recommendation.category == "accessibility" and context.get("accessibility_needs", False):
            return 0.95
        elif recommendation.category == "performance" and context.get("performance_mode") == "low":
            return 0.85
        else:
            return 0.5
    
    def _calculate_performance_impact_score(self, recommendation: AdaptiveRecommendation) -> float:
        """Calculate the performance impact of the recommendation."""
        if recommendation.category == "performance":
            return 0.9
        else:
            return 0.6  # Neutral impact
    
    def _calculate_novelty_score(self, recommendation: AdaptiveRecommendation, user_id: str) -> float:
        """Calculate the novelty/exploration value of the recommendation."""
        if recommendation.category == "exploration":
            return 0.8
        else:
            return 0.3  # Low novelty for other categories
    
    def _get_most_effective_value(self, preference_path: str, successful_changes: List[PreferenceChange]) -> Any:
        """Find the most effective value for a preference based on successful changes."""
        # Filter changes for this preference
        relevant_changes = [change for change in successful_changes if change.preference_path == preference_path]
        
        if not relevant_changes:
            return None
        
        # Find the value with highest satisfaction impact
        best_change = max(relevant_changes, key=lambda x: x.satisfaction_impact or 0)
        return best_change.new_value
    
    def _suggest_novelty_value(self, preference_path: str) -> Any:
        """Suggest a novel value to explore for a preference."""
        # Provide sensible default exploration values based on preference type
        novelty_suggestions = {
            "appearance.theme": "adaptive_auto",
            "appearance.color_scheme": "green",
            "voice.speech_rate": 1.2,
            "gesture.gesture_sensitivity": 0.9,
            "performance.performance_mode": "optimized"
        }
        
        return novelty_suggestions.get(preference_path, None)


# Helper functions for the enhanced preference management system
def _get_common_preferences(changes: List[PreferenceChange]) -> List[str]:
    """Get common preferences from a list of changes."""
    preference_count = defaultdict(int)
    for change in changes:
        preference_count[change.preference_path] += 1
    
    # Return top 5 most common preferences
    sorted_prefs = sorted(preference_count.items(), key=lambda x: x[1], reverse=True)
    return [pref for pref, count in sorted_prefs[:5]]


def _analyze_cluster_timing(changes: List[PreferenceChange]) -> Dict[str, Any]:
    """Analyze timing patterns in a cluster of changes."""
    if not changes:
        return {}
    
    # Extract hours from timestamps
    hours = []
    for change in changes:
        try:
            timestamp = datetime.fromisoformat(change.timestamp)
            hours.append(timestamp.hour)
        except:
            continue
    
    if not hours:
        return {}
    
    # Find most common hours
    hour_count = defaultdict(int)
    for hour in hours:
        hour_count[hour] += 1
    
    most_common_hour = max(hour_count.items(), key=lambda x: x[1])[0] if hour_count else 12
    
    return {
        "most_common_hour": most_common_hour,
        "hour_distribution": dict(hour_count),
        "peak_usage_hours": [hour for hour, count in hour_count.items() if count >= 2]
    }


def _characterize_preference_cluster(changes: List[PreferenceChange]) -> Dict[str, str]:
    """Characterize a cluster of preference changes."""
    if not changes:
        return {}
    
    # Analyze satisfaction impacts
    satisfaction_scores = [change.satisfaction_impact for change in changes if change.satisfaction_impact is not None]
    
    if satisfaction_scores:
        avg_satisfaction = np.mean(satisfaction_scores) if np else 0.5
        
        if avg_satisfaction > 0.8:
            satisfaction_level = "high"
        elif avg_satisfaction > 0.6:
            satisfaction_level = "moderate"
        else:
            satisfaction_level = "low"
    else:
        satisfaction_level = "unknown"
    
    # Analyze preference types
    preference_types = [change.preference_path.split('.')[0] for change in changes]
    most_common_type = max(set(preference_types), key=preference_types.count) if preference_types else "unknown"
    
    return {
        "satisfaction_level": satisfaction_level,
        "dominant_preference_type": most_common_type,
        "cluster_size": len(changes),
        "diversity": len(set(change.preference_path for change in changes))
    }


def _analyze_context_usage(context_usage: Dict[str, int]) -> Dict[str, Any]:
    """Analyze context usage patterns."""
    if not context_usage:
        return {"message": "No context usage data"}
    
    total_usage = sum(context_usage.values())
    
    analysis = {
        "total_contexts": len(context_usage),
        "total_usage": total_usage,
        "most_used_context": max(context_usage.items(), key=lambda x: x[1])[0],
        "context_distribution": {
            context: {"count": count, "percentage": count / total_usage}
            for context, count in context_usage.items()
        },
        "diversity_score": len(context_usage) / max(total_usage, 1)  # Higher is more diverse
    }
    
    return analysis


def _generate_behavioral_insights(user_id: str, analytics: Dict[str, Any]) -> List[str]:
    """Generate behavioral insights from analytics data."""
    insights = []
    
    # Success rate insights
    success_rate = analytics.get("successful_adaptations", 0) / max(analytics.get("total_changes", 1), 1)
    
    if success_rate > 0.8:
        insights.append("User shows excellent adaptation to preference changes")
    elif success_rate < 0.4:
        insights.append("User may benefit from more conservative preference adaptations")
    
    # Satisfaction insights
    satisfaction_scores = analytics.get("satisfaction_scores", [])
    if satisfaction_scores and len(satisfaction_scores) >= 5:
        if np:
            recent_trend = np.mean(satisfaction_scores[-5:]) - np.mean(satisfaction_scores[:-5])
            if recent_trend > 0.1:
                insights.append("User satisfaction is improving over time")
            elif recent_trend < -0.1:
                insights.append("User satisfaction has declined recently - review recent changes")
    
    # Context usage insights
    context_usage = analytics.get("context_usage", {})
    if context_usage:
        total_contexts = len(context_usage)
        if total_contexts == 1:
            insights.append("User primarily uses system in one context - consider context-specific optimizations")
        elif total_contexts > 5:
            insights.append("User has diverse usage contexts - adaptive preferences would be beneficial")
    
    return insights


def _predict_behavioral_trends(user_id: str, analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Predict future behavioral trends based on analytics."""
    predictions = {}
    
    # Predict satisfaction trend
    satisfaction_scores = analytics.get("satisfaction_scores", [])
    if len(satisfaction_scores) >= 10 and np:
        # Simple linear trend
        recent_scores = satisfaction_scores[-10:]
        time_indices = list(range(len(recent_scores)))
        
        if len(time_indices) > 1:
            trend_slope = np.polyfit(time_indices, recent_scores, 1)[0]
            
            if trend_slope > 0.05:
                predictions["satisfaction_trend"] = "improving"
            elif trend_slope < -0.05:
                predictions["satisfaction_trend"] = "declining"
            else:
                predictions["satisfaction_trend"] = "stable"
    
    # Predict usage patterns
    context_usage = analytics.get("context_usage", {})
    if context_usage:
        most_used_context = max(context_usage.items(), key=lambda x: x[1])[0]
        predictions["dominant_context"] = most_used_context
        predictions["usage_diversity"] = len(context_usage) / sum(context_usage.values())
    
    return predictions


# Additional helper methods for ML analysis
def _analyze_sequential_patterns(history: List[PreferenceChange]) -> Dict[str, Any]:
    """Analyze sequential patterns in preference changes."""
    if len(history) < 3:
        return {"message": "Insufficient data for sequential analysis"}
    
    # Look for common sequences
    sequences = []
    for i in range(len(history) - 2):
        sequence = [
            history[i].preference_path,
            history[i + 1].preference_path,
            history[i + 2].preference_path
        ]
        sequences.append(tuple(sequence))
    
    # Count sequence frequencies
    sequence_counts = defaultdict(int)
    for seq in sequences:
        sequence_counts[seq] += 1
    
    # Find most common sequences
    common_sequences = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "total_sequences": len(sequences),
        "unique_sequences": len(sequence_counts),
        "most_common_sequences": [
            {"sequence": list(seq), "frequency": count}
            for seq, count in common_sequences
        ]
    }


def _analyze_temporal_preference_patterns(history: List[PreferenceChange]) -> Dict[str, Any]:
    """Analyze temporal patterns in preference changes."""
    if not history:
        return {}
    
    # Group by hour of day
    hourly_changes = defaultdict(list)
    daily_changes = defaultdict(list)
    
    for change in history:
        try:
            timestamp = datetime.fromisoformat(change.timestamp)
            hourly_changes[timestamp.hour].append(change)
            daily_changes[timestamp.strftime('%A')].append(change)
        except:
            continue
    
    # Find peak hours and days
    peak_hour = max(hourly_changes.items(), key=lambda x: len(x[1]))[0] if hourly_changes else None
    peak_day = max(daily_changes.items(), key=lambda x: len(x[1]))[0] if daily_changes else None
    
    return {
        "peak_hour": peak_hour,
        "peak_day": peak_day,
        "hourly_distribution": {hour: len(changes) for hour, changes in hourly_changes.items()},
        "daily_distribution": {day: len(changes) for day, changes in daily_changes.items()}
    }


def _analyze_contextual_preference_patterns(history: List[PreferenceChange]) -> Dict[str, Any]:
    """Analyze context-based preference patterns."""
    context_patterns = defaultdict(lambda: defaultdict(int))
    
    for change in history:
        context = change.context or {}
        activity = context.get("activity", "unknown")
        preference_type = change.preference_path.split('.')[0]
        
        context_patterns[activity][preference_type] += 1
    
    return {
        "context_preferences": dict(context_patterns),
        "contexts_analyzed": len(context_patterns)
    }


def _analyze_satisfaction_patterns(history: List[PreferenceChange]) -> Dict[str, Any]:
    """Analyze patterns in user satisfaction with preference changes."""
    satisfaction_by_preference = defaultdict(list)
    satisfaction_by_context = defaultdict(list)
    
    for change in history:
        if change.satisfaction_impact is not None:
            satisfaction_by_preference[change.preference_path].append(change.satisfaction_impact)
            
            context = change.context or {}
            activity = context.get("activity", "unknown")
            satisfaction_by_context[activity].append(change.satisfaction_impact)
    
    # Calculate averages
    avg_satisfaction_by_preference = {
        pref: np.mean(scores) if np and scores else 0.5
        for pref, scores in satisfaction_by_preference.items()
    }
    
    avg_satisfaction_by_context = {
        context: np.mean(scores) if np and scores else 0.5
        for context, scores in satisfaction_by_context.items()
    }
    
    return {
        "preference_satisfaction": avg_satisfaction_by_preference,
        "context_satisfaction": avg_satisfaction_by_context,
        "most_satisfying_preference": max(avg_satisfaction_by_preference.items(), key=lambda x: x[1])[0] if avg_satisfaction_by_preference else None,
        "most_satisfying_context": max(avg_satisfaction_by_context.items(), key=lambda x: x[1])[0] if avg_satisfaction_by_context else None
    }


def _predict_next_preference_changes(features: Any, history: List[PreferenceChange]) -> List[str]:
    """Predict likely next preference changes."""
    if len(history) < 5:
        return []
    
    # Simple prediction based on recent patterns
    recent_preferences = [change.preference_path for change in history[-5:]]
    preference_frequency = defaultdict(int)
    
    for pref in recent_preferences:
        preference_frequency[pref] += 1
    
    # Return most frequently changed preferences as predictions
    sorted_prefs = sorted(preference_frequency.items(), key=lambda x: x[1], reverse=True)
    return [pref for pref, count in sorted_prefs[:3]]


def _predict_satisfaction_trends(history: List[PreferenceChange]) -> Dict[str, Any]:
    """Predict satisfaction trends based on historical data."""
    satisfaction_scores = [change.satisfaction_impact for change in history if change.satisfaction_impact is not None]
    
    if len(satisfaction_scores) < 5:
        return {"message": "Insufficient satisfaction data for trend prediction"}
    
    # Simple trend analysis
    if np:
        recent_avg = np.mean(satisfaction_scores[-5:])
        historical_avg = np.mean(satisfaction_scores[:-5]) if len(satisfaction_scores) > 5 else recent_avg
        
        trend = "improving" if recent_avg > historical_avg * 1.1 else "declining" if recent_avg < historical_avg * 0.9 else "stable"
    else:
        trend = "stable"
    
    return {
        "trend": trend,
        "recent_average": float(np.mean(satisfaction_scores[-5:])) if np else 0.5,
        "historical_average": float(np.mean(satisfaction_scores[:-5])) if np and len(satisfaction_scores) > 5 else 0.5
    }


def _predict_contextual_optimizations(history: List[PreferenceChange]) -> Dict[str, Any]:
    """Predict optimal preferences for different contexts."""
    context_optimizations = {}
    
    # Group changes by context
    context_groups = defaultdict(list)
    for change in history:
        context = change.context or {}
        activity = context.get("activity", "unknown")
        context_groups[activity].append(change)
    
    # Find optimal preferences for each context
    for activity, changes in context_groups.items():
        if len(changes) >= 3:
            # Find preferences with highest satisfaction in this context
            satisfaction_by_pref = defaultdict(list)
            for change in changes:
                if change.satisfaction_impact is not None:
                    satisfaction_by_pref[change.preference_path].append(change.satisfaction_impact)
            
            optimal_prefs = {}
            for pref, scores in satisfaction_by_pref.items():
                if len(scores) >= 2:
                    avg_satisfaction = np.mean(scores) if np else 0.5
                    if avg_satisfaction > 0.7:
                        optimal_prefs[pref] = avg_satisfaction
            
            if optimal_prefs:
                context_optimizations[activity] = optimal_prefs
    
    return context_optimizations


def _analyze_anomaly_reasons(anomaly_change: PreferenceChange, history: List[PreferenceChange]) -> List[str]:
    """Analyze potential reasons for an anomalous preference change."""
    reasons = []
    
    # Check if it's an unusual time
    try:
        timestamp = datetime.fromisoformat(anomaly_change.timestamp)
        hour = timestamp.hour
        
        # Collect all change hours
        all_hours = []
        for change in history:
            try:
                change_time = datetime.fromisoformat(change.timestamp)
                all_hours.append(change_time.hour)
            except:
                continue
        
        if all_hours:
            common_hours = set(h for h in all_hours if all_hours.count(h) >= 2)
            if hour not in common_hours:
                reasons.append(f"Unusual time of change: {hour}:00")
    except:
        pass
    
    # Check if it's an unusual preference
    all_preferences = [change.preference_path for change in history]
    if all_preferences.count(anomaly_change.preference_path) <= 1:
        reasons.append("Rarely changed preference")
    
    # Check satisfaction impact
    if anomaly_change.satisfaction_impact is not None and anomaly_change.satisfaction_impact < 0.3:
        reasons.append("Low satisfaction impact")
    
    return reasons or ["Unknown anomaly reason"]


def _summarize_anomalies(anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize detected preference anomalies."""
    if not anomalies:
        return {"message": "No anomalies detected"}
    
    # Categorize anomalies by type
    anomaly_categories = defaultdict(int)
    for anomaly in anomalies:
        reasons = anomaly.get("potential_reasons", [])
        for reason in reasons:
            if "time" in reason.lower():
                anomaly_categories["temporal"] += 1
            elif "preference" in reason.lower():
                anomaly_categories["preference_type"] += 1
            elif "satisfaction" in reason.lower():
                anomaly_categories["satisfaction"] += 1
            else:
                anomaly_categories["other"] += 1
    
    return {
        "total_anomalies": len(anomalies),
        "anomaly_categories": dict(anomaly_categories),
        "severity_distribution": {
            "high": len([a for a in anomalies if a.get("anomaly_score", 0) < -0.5]),
            "medium": len([a for a in anomalies if -0.5 <= a.get("anomaly_score", 0) < 0]),
            "low": len([a for a in anomalies if a.get("anomaly_score", 0) >= 0])
        }
    }


# Optimization strategy methods
def _optimize_for_performance(user_id: str, current_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize settings for maximum performance."""
    optimized = current_settings.copy()
    
    # Performance-focused optimizations
    performance_optimizations = {
        "appearance.animations_enabled": False,
        "appearance.particle_effects": False,
        "appearance.blur_effects": False,
        "appearance.transparency": 1.0,
        "performance.gpu_acceleration": False,
        "performance.cache_enabled": True,
        "performance.background_processing": False
    }
    
    # Apply optimizations
    for key, value in performance_optimizations.items():
        keys = key.split('.')
        current = optimized
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    return optimized


def _optimize_for_accessibility(user_id: str, current_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize settings for accessibility."""
    optimized = current_settings.copy()
    
    accessibility_optimizations = {
        "accessibility.high_contrast": True,
        "accessibility.large_text": True,
        "accessibility.reduced_motion": True,
        "accessibility.screen_reader_compatible": True,
        "appearance.animations_enabled": False,
        "appearance.font_size": "large",
        "voice.speech_rate": 0.9,
        "gesture.gesture_timeout": 5.0
    }
    
    # Apply optimizations
    for key, value in accessibility_optimizations.items():
        keys = key.split('.')
        current = optimized
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    return optimized


def _optimize_for_satisfaction(user_id: str, current_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize settings for user satisfaction."""
    optimized = current_settings.copy()
    
    # Satisfaction-focused optimizations (based on common preferences)
    satisfaction_optimizations = {
        "appearance.theme": "adaptive_auto",
        "interaction.auto_suggestions": True,
        "automation.smart_suggestions": True,
        "automation.predictive_actions": True,
        "notifications.achievement_notifications": True,
        "voice.voice_feedback": True
    }
    
    # Apply optimizations
    for key, value in satisfaction_optimizations.items():
        keys = key.split('.')
        current = optimized
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    return optimized


def _optimize_for_energy(user_id: str, current_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize settings for energy efficiency."""
    optimized = current_settings.copy()
    
    energy_optimizations = {
        "performance.optimize_battery": True,
        "appearance.animations_enabled": False,
        "appearance.particle_effects": False,
        "appearance.holographic_elements": False,
        "performance.gpu_acceleration": False,
        "performance.background_processing": False,
        "ar_interface.hologram_quality": "medium",
        "voice.background_listening": False
    }
    
    # Apply optimizations
    for key, value in energy_optimizations.items():
        keys = key.split('.')
        current = optimized
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    return optimized


def _optimize_for_productivity(user_id: str, current_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize settings for productivity."""
    optimized = current_settings.copy()
    
    productivity_optimizations = {
        "appearance.theme": "focus",
        "automation.smart_suggestions": True,
        "automation.predictive_actions": True,
        "automation.proactive_assistance": True,
        "notifications.system_notifications": False,
        "notifications.achievement_notifications": False,
        "appearance.animations_enabled": False,
        "appearance.particle_effects": False,
        "interaction.confirmation_level": "low"
    }
    
    # Apply optimizations
    for key, value in productivity_optimizations.items():
        keys = key.split('.')
        current = optimized
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    return optimized


def _validate_optimizations(user_id: str, original: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that optimizations are beneficial and safe."""
    validation = {
        "changes_made": 0,
        "potential_conflicts": [],
        "safety_checks": [],
        "overall_confidence": 0.8
    }
    
    # Count changes
    def count_differences(orig, opt, path=""):
        count = 0
        for key, value in opt.items():
            if isinstance(value, dict) and key in orig and isinstance(orig[key], dict):
                count += count_differences(orig[key], value, f"{path}.{key}" if path else key)
            elif key not in orig or orig[key] != value:
                count += 1
        return count
    
    validation["changes_made"] = count_differences(original, optimized)
    
    # Check for potential conflicts
    if optimized.get("appearance", {}).get("animations_enabled") == False and \
       optimized.get("appearance", {}).get("theme") == "cyberpunk":
        validation["potential_conflicts"].append("Disabled animations may reduce cyberpunk theme effectiveness")
    
    # Safety checks
    if validation["changes_made"] > 10:
        validation["safety_checks"].append("Large number of changes - consider gradual application")
        validation["overall_confidence"] *= 0.8
    
    return validation


if __name__ == "__main__":
    # Test the enhanced preference management system
    print("Testing Enhanced Preference Management System...")
    
    # Initialize preference manager
    pref_manager = PreferenceManager()
    
    # Test user preferences
    test_user_id = "test_user_prefs"
    
    # Get default preferences
    default_prefs = pref_manager.get_user_preferences(test_user_id)
    print(f"Default preferences loaded: {len(default_prefs)} categories")
    
    # Test preference updates
    pref_manager.update_preference(test_user_id, "appearance.theme", "dark")
    pref_manager.update_preference(test_user_id, "voice.speech_rate", 1.2)
    
    # Test AI recommendations
    context = {"activity": "work", "time_of_day": "evening"}
    recommendations = pref_manager.get_ai_recommendations(test_user_id, context)
    print(f"AI recommendations generated: {len(recommendations)}")
    
    # Test adaptive optimizations
    adaptations = pref_manager.apply_adaptive_optimizations(test_user_id, context)
    print(f"Adaptive optimizations applied: {adaptations['total_adaptations']}")
    
    # Test pattern analysis
    analysis = pref_manager.analyze_preference_patterns(test_user_id)
    print(f"Pattern analysis completed: {analysis.get('summary', {}).get('overall_health', 'unknown')}")
    
    # Test contextual preferences
    contextual_prefs = pref_manager.get_contextual_preferences(test_user_id, context)
    print(f"Contextual preferences optimized: {len(contextual_prefs['applied_recommendations'])} recommendations applied")
    
    print("Enhanced Preference Management System test completed!")