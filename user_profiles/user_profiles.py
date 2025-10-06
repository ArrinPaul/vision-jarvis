import json
import os
import hashlib
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import threading
import logging

class UserProfileManager:
    """
    Advanced User Profiles with AI-powered personalization, behavior learning, and adaptive responses.
    Features: behavioral analysis, preference learning, activity patterns, and intelligent recommendations.
    """
    def __init__(self, profile_path="user_profiles.json", behavior_path="user_behaviors.json"):
        self.profile_path = profile_path
        self.behavior_path = behavior_path
        self.profiles = self._load_profiles()
        self.behaviors = self._load_behaviors()
        
        # Learning systems
        self.behavior_analyzer = BehaviorAnalyzer()
        self.preference_learner = PreferenceLearner()
        self.activity_tracker = ActivityTracker()
        self.recommendation_engine = RecommendationEngine()
        
        # Session tracking
        self.current_sessions = {}
        self.interaction_history = defaultdict(list)
        
        # Personalization models
        self.user_models = {}
        self.context_aware = True
        
        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 5  # Minimum interactions for adaptation
        
        # Background tasks
        self.background_thread = threading.Thread(target=self._background_learning, daemon=True)
        self.background_thread.start()
        
        logging.info("Advanced User Profile Manager initialized")

    def _load_profiles(self):
        """Load user profiles from storage."""
        if os.path.exists(self.profile_path):
            with open(self.profile_path, "r") as f:
                return json.load(f)
        return {}

    def _save_profiles(self):
        """Save user profiles to storage."""
        with open(self.profile_path, "w") as f:
            json.dump(self.profiles, f, indent=2)

    def _load_behaviors(self):
        """Load behavior data from storage."""
        if os.path.exists(self.behavior_path):
            with open(self.behavior_path, "r") as f:
                return json.load(f)
        return {}

    def _save_behaviors(self):
        """Save behavior data to storage."""
        with open(self.behavior_path, "w") as f:
            json.dump(self.behaviors, f, indent=2)

    def create_profile(self, user_info):
        """Create a comprehensive user profile with learning capabilities."""
        user_id = user_info.get("id")
        if not user_id:
            raise ValueError("User info must include 'id'")
            
        # Create comprehensive profile structure
        profile = {
            "id": user_id,
            "personal_info": {
                "name": user_info.get("name", ""),
                "age": user_info.get("age"),
                "occupation": user_info.get("occupation", ""),
                "location": user_info.get("location", ""),
                "timezone": user_info.get("timezone", "UTC"),
                "language": user_info.get("language", "en"),
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "preferences": {
                "communication_style": user_info.get("communication_style", "professional"),
                "response_length": user_info.get("response_length", "medium"),
                "interaction_mode": user_info.get("interaction_mode", "conversational"),
                "privacy_level": user_info.get("privacy_level", "medium"),
                "learning_enabled": user_info.get("learning_enabled", True),
                "voice_settings": {
                    "voice_id": user_info.get("voice_id", "default"),
                    "speech_rate": user_info.get("speech_rate", 1.0),
                    "pitch": user_info.get("pitch", 1.0),
                    "volume": user_info.get("volume", 0.8)
                },
                "ui_preferences": {
                    "theme": user_info.get("theme", "dark"),
                    "font_size": user_info.get("font_size", "medium"),
                    "animations": user_info.get("animations", True),
                    "ar_overlays": user_info.get("ar_overlays", True)
                }
            },
            "interests": user_info.get("interests", []),
            "goals": user_info.get("goals", []),
            "accessibility": {
                "visual_impairment": user_info.get("visual_impairment", False),
                "hearing_impairment": user_info.get("hearing_impairment", False),
                "motor_impairment": user_info.get("motor_impairment", False),
                "cognitive_assistance": user_info.get("cognitive_assistance", False)
            },
            "behavioral_patterns": {
                "active_hours": [],
                "interaction_frequency": {},
                "preferred_features": [],
                "usage_patterns": {},
                "learning_progress": {}
            },
            "security": {
                "authentication_methods": ["face", "voice"],
                "security_level": user_info.get("security_level", "medium"),
                "biometric_data": {}
            },
            "customization": {
                "shortcuts": {},
                "macros": {},
                "custom_commands": {},
                "automation_rules": []
            },
            "stats": {
                "total_interactions": 0,
                "successful_commands": 0,
                "failed_commands": 0,
                "time_saved": 0,  # in seconds
                "favorite_features": [],
                "satisfaction_score": 0.0
            }
        }
        
        self.profiles[user_id] = profile
        
        # Initialize behavior tracking
        self.behaviors[user_id] = {
            "interaction_history": [],
            "learned_patterns": {},
            "adaptation_history": [],
            "context_preferences": {},
            "feedback_history": []
        }
        
        # Initialize user model
        self.user_models[user_id] = UserPersonalizationModel(user_id)
        
        self._save_profiles()
        self._save_behaviors()
        
        logging.info(f"Advanced profile created for user {user_id}")
        return profile

    def get_profile(self, user_id):
        """Retrieve comprehensive user profile."""
        return self.profiles.get(user_id, None)

    def update_profile(self, user_id, updates):
        """Update user profile with nested data support."""
        if user_id not in self.profiles:
            raise ValueError(f"User {user_id} not found")
            
        profile = self.profiles[user_id]
        
        # Deep update function
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
                    
        deep_update(profile, updates)
        profile["personal_info"]["last_updated"] = datetime.now().isoformat()
        
        self._save_profiles()
        logging.info(f"Profile updated for user {user_id}")

    def start_session(self, user_id, context=None):
        """Start a new interaction session."""
        if user_id not in self.profiles:
            logging.warning(f"Starting session for unknown user {user_id}")
            
        session = {
            "user_id": user_id,
            "start_time": datetime.now(),
            "context": context or {},
            "interactions": [],
            "mood_indicators": [],
            "performance_metrics": {},
            "adaptation_triggers": []
        }
        
        session_id = self._generate_session_id(user_id)
        self.current_sessions[session_id] = session
        
        # Update user stats
        if user_id in self.profiles:
            self.profiles[user_id]["stats"]["total_interactions"] += 1
            
        return session_id

    def end_session(self, session_id):
        """End interaction session and process learning."""
        if session_id not in self.current_sessions:
            return
            
        session = self.current_sessions[session_id]
        user_id = session["user_id"]
        
        # Calculate session metrics
        session_duration = (datetime.now() - session["start_time"]).total_seconds()
        session["duration"] = session_duration
        session["end_time"] = datetime.now()
        
        # Process session for learning
        if user_id in self.profiles and self.profiles[user_id]["preferences"]["learning_enabled"]:
            self._process_session_learning(session)
            
        # Store in behavior history
        if user_id in self.behaviors:
            self.behaviors[user_id]["interaction_history"].append({
                "session_id": session_id,
                "duration": session_duration,
                "interactions": len(session["interactions"]),
                "context": session["context"],
                "timestamp": datetime.now().isoformat()
            })
            
        # Clean up
        del self.current_sessions[session_id]
        self._save_behaviors()

    def record_interaction(self, session_id, interaction_data):
        """Record an interaction within a session."""
        if session_id not in self.current_sessions:
            return False
            
        session = self.current_sessions[session_id]
        user_id = session["user_id"]
        
        # Enhance interaction data
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "type": interaction_data.get("type", "unknown"),
            "command": interaction_data.get("command", ""),
            "success": interaction_data.get("success", True),
            "response_time": interaction_data.get("response_time", 0),
            "user_satisfaction": interaction_data.get("satisfaction"),
            "context": interaction_data.get("context", {}),
            "adaptations_applied": interaction_data.get("adaptations", [])
        }
        
        session["interactions"].append(interaction)
        
        # Update user stats
        if user_id in self.profiles:
            if interaction["success"]:
                self.profiles[user_id]["stats"]["successful_commands"] += 1
            else:
                self.profiles[user_id]["stats"]["failed_commands"] += 1
                
        # Real-time learning
        if self.context_aware:
            self._update_real_time_learning(user_id, interaction)
            
        return True

    def get_personalized_response(self, user_id, base_response, context=None):
        """Generate personalized response based on user profile."""
        if user_id not in self.profiles:
            return base_response
            
        profile = self.profiles[user_id]
        preferences = profile["preferences"]
        
        # Apply communication style
        response = self._adapt_communication_style(base_response, preferences["communication_style"])
        
        # Apply response length preference
        response = self._adapt_response_length(response, preferences["response_length"])
        
        # Apply personalization based on learned patterns
        if user_id in self.user_models:
            response = self.user_models[user_id].personalize_response(response, context)
            
        return response

    def get_recommendations(self, user_id, category=None):
        """Get personalized recommendations for user."""
        if user_id not in self.profiles:
            return []
            
        return self.recommendation_engine.get_recommendations(
            user_id, 
            self.profiles[user_id], 
            self.behaviors.get(user_id, {}),
            category
        )

    def analyze_user_behavior(self, user_id):
        """Analyze user behavior patterns."""
        if user_id not in self.behaviors:
            return {}
            
        return self.behavior_analyzer.analyze(self.behaviors[user_id])

    def get_user_insights(self, user_id):
        """Get comprehensive user insights."""
        if user_id not in self.profiles:
            return {}
            
        profile = self.profiles[user_id]
        behavior_data = self.behaviors.get(user_id, {})
        
        insights = {
            "profile_completeness": self._calculate_profile_completeness(profile),
            "engagement_level": self._calculate_engagement_level(behavior_data),
            "learning_progress": self._calculate_learning_progress(user_id),
            "adaptation_effectiveness": self._calculate_adaptation_effectiveness(user_id),
            "usage_patterns": self.activity_tracker.get_patterns(user_id, behavior_data),
            "recommendations": self.get_recommendations(user_id),
            "satisfaction_trends": self._get_satisfaction_trends(user_id),
            "feature_adoption": self._get_feature_adoption(user_id)
        }
        
        return insights

    def _adapt_communication_style(self, response, style):
        """Adapt response based on communication style."""
        adaptations = {
            "formal": {
                "greetings": {"hi": "Good day", "hey": "Greetings", "yo": "Hello"},
                "tone": "formal"
            },
            "casual": {
                "greetings": {"good day": "hi", "greetings": "hey"},
                "tone": "casual"
            },
            "professional": {
                "tone": "professional"
            },
            "friendly": {
                "tone": "warm"
            }
        }
        
        if style in adaptations:
            # Apply style-specific adaptations
            adaptation = adaptations[style]
            
            # Replace greetings
            if "greetings" in adaptation:
                for old, new in adaptation["greetings"].items():
                    response = response.replace(old, new)
                    
        return response

    def _adapt_response_length(self, response, length_pref):
        """Adapt response length based on preference."""
        if length_pref == "short":
            # Shorten response by removing explanatory text
            sentences = response.split('. ')
            if len(sentences) > 2:
                response = '. '.join(sentences[:2]) + '.'
        elif length_pref == "detailed":
            # Could add more detail if base response is short
            if len(response.split()) < 10:
                response += " Would you like me to provide more details about this?"
                
        return response

    def _process_session_learning(self, session):
        """Process session data for machine learning insights."""
        user_id = session["user_id"]
        
        # Extract learning features
        features = {
            "session_duration": session["duration"],
            "interaction_count": len(session["interactions"]),
            "success_rate": sum(1 for i in session["interactions"] if i["success"]) / max(len(session["interactions"]), 1),
            "avg_response_time": np.mean([i["response_time"] for i in session["interactions"]]) if session["interactions"] else 0,
            "context_features": self._extract_context_features(session["context"]),
            "time_of_day": session["start_time"].hour,
            "day_of_week": session["start_time"].weekday()
        }
        
        # Update user model
        if user_id in self.user_models:
            self.user_models[user_id].update_model(features, session["interactions"])

    def _update_real_time_learning(self, user_id, interaction):
        """Update learning models in real-time."""
        if user_id not in self.user_models:
            return
            
        # Update preference learning
        self.preference_learner.update(user_id, interaction)
        
        # Update activity tracking
        self.activity_tracker.record_activity(user_id, interaction)

    def _background_learning(self):
        """Background thread for continuous learning."""
        while True:
            try:
                # Run learning updates every 5 minutes
                time.sleep(300)
                
                for user_id in self.profiles.keys():
                    if self.profiles[user_id]["preferences"]["learning_enabled"]:
                        self._update_user_model(user_id)
                        
            except Exception as e:
                logging.error(f"Background learning error: {e}")

    def _update_user_model(self, user_id):
        """Update comprehensive user model."""
        if user_id not in self.user_models:
            return
            
        # Collect recent behavior data
        recent_data = self._get_recent_behavior_data(user_id)
        
        if len(recent_data) >= self.adaptation_threshold:
            # Update model with recent data
            self.user_models[user_id].retrain(recent_data)
            
            # Update behavioral patterns in profile
            patterns = self.behavior_analyzer.extract_patterns(recent_data)
            self.profiles[user_id]["behavioral_patterns"].update(patterns)
            
            self._save_profiles()

    def _get_recent_behavior_data(self, user_id):
        """Get recent behavior data for learning."""
        if user_id not in self.behaviors:
            return []
            
        cutoff_date = datetime.now() - timedelta(days=7)  # Last week
        recent_data = []
        
        for session in self.behaviors[user_id]["interaction_history"]:
            session_date = datetime.fromisoformat(session["timestamp"])
            if session_date > cutoff_date:
                recent_data.append(session)
                
        return recent_data

    def _calculate_profile_completeness(self, profile):
        """Calculate how complete a user profile is."""
        total_fields = 0
        filled_fields = 0
        
        def count_fields(obj, prefix=""):
            nonlocal total_fields, filled_fields
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, dict):
                        count_fields(value, f"{prefix}.{key}")
                    else:
                        total_fields += 1
                        if value is not None and value != "" and value != []:
                            filled_fields += 1
                            
        count_fields(profile)
        return filled_fields / total_fields if total_fields > 0 else 0

    def _calculate_engagement_level(self, behavior_data):
        """Calculate user engagement level."""
        if not behavior_data.get("interaction_history"):
            return 0.0
            
        recent_sessions = behavior_data["interaction_history"][-10:]  # Last 10 sessions
        
        if not recent_sessions:
            return 0.0
            
        avg_duration = np.mean([s.get("duration", 0) for s in recent_sessions])
        avg_interactions = np.mean([s.get("interactions", 0) for s in recent_sessions])
        
        # Normalize to 0-1 scale
        engagement = min(1.0, (avg_duration / 300 + avg_interactions / 10) / 2)
        return engagement

    def _calculate_learning_progress(self, user_id):
        """Calculate how well the system is learning about the user."""
        if user_id not in self.user_models:
            return 0.0
            
        return self.user_models[user_id].get_learning_confidence()

    def _calculate_adaptation_effectiveness(self, user_id):
        """Calculate how effective personalization adaptations are."""
        if user_id not in self.behaviors:
            return 0.0
            
        behavior_data = self.behaviors[user_id]
        
        # Look at satisfaction scores over time
        if "feedback_history" in behavior_data and behavior_data["feedback_history"]:
            recent_feedback = behavior_data["feedback_history"][-10:]
            satisfaction_scores = [f.get("satisfaction", 0) for f in recent_feedback if "satisfaction" in f]
            
            if satisfaction_scores:
                return np.mean(satisfaction_scores)
                
        return 0.5  # Neutral if no feedback

    def _generate_session_id(self, user_id):
        """Generate unique session ID."""
        timestamp = str(time.time())
        return hashlib.md5(f"{user_id}_{timestamp}".encode()).hexdigest()[:12]

    def _extract_context_features(self, context):
        """Extract numerical features from context."""
        features = {}
        
        # Common context features
        features["has_location"] = 1 if "location" in context else 0
        features["has_time_context"] = 1 if "time" in context else 0
        features["device_type"] = hash(context.get("device", "unknown")) % 100
        features["app_context"] = hash(context.get("app", "unknown")) % 100
        
        return features

    def _get_satisfaction_trends(self, user_id):
        """Get satisfaction trends over time."""
        if user_id not in self.behaviors:
            return []
            
        behavior_data = self.behaviors[user_id]
        feedback = behavior_data.get("feedback_history", [])
        
        # Group by time periods
        trends = []
        for i in range(0, len(feedback), 10):  # Group by 10s
            chunk = feedback[i:i+10]
            if chunk:
                avg_satisfaction = np.mean([f.get("satisfaction", 0) for f in chunk])
                trends.append(avg_satisfaction)
                
        return trends

    def _get_feature_adoption(self, user_id):
        """Get feature adoption statistics."""
        if user_id not in self.behaviors:
            return {}
            
        behavior_data = self.behaviors[user_id]
        interactions = behavior_data.get("interaction_history", [])
        
        feature_usage = defaultdict(int)
        for session in interactions:
            context = session.get("context", {})
            if "feature" in context:
                feature_usage[context["feature"]] += 1
                
        return dict(feature_usage)

# Supporting Classes

class UserPersonalizationModel:
    """Machine learning model for individual user personalization."""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.features = []
        self.labels = []
        self.model = None
        self.confidence = 0.0
        self.last_training = None
        
    def update_model(self, features, interactions):
        """Update model with new data."""
        # Extract satisfaction scores as labels
        satisfaction_scores = [i.get("user_satisfaction", 0.5) for i in interactions if "user_satisfaction" in i]
        
        if satisfaction_scores:
            self.features.append(list(features.values()))
            self.labels.append(np.mean(satisfaction_scores))
            
            # Retrain if enough data
            if len(self.features) >= 10:
                self._train_model()
                
    def _train_model(self):
        """Train the personalization model."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            X = np.array(self.features[-50:])  # Use last 50 data points
            y = np.array(self.labels[-50:])
            
            self.model = RandomForestRegressor(n_estimators=10, random_state=42)
            self.model.fit(X, y)
            
            # Calculate confidence based on model performance
            if len(X) > 5:
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(self.model, X, y, cv=min(5, len(X)))
                self.confidence = np.mean(scores)
                
            self.last_training = datetime.now()
            
        except Exception as e:
            logging.error(f"Model training error for user {self.user_id}: {e}")
            
    def personalize_response(self, response, context):
        """Personalize response using trained model."""
        if self.model is None or self.confidence < 0.3:
            return response
            
        # Apply model-based personalization
        # This would involve more sophisticated response modification
        return response
        
    def get_learning_confidence(self):
        """Get confidence in learned patterns."""
        return self.confidence
        
    def retrain(self, recent_data):
        """Retrain model with recent data."""
        # Extract features from recent data for retraining
        for session in recent_data:
            features = {
                "duration": session.get("duration", 0),
                "interactions": session.get("interactions", 0),
                "context_size": len(session.get("context", {}))
            }
            self.update_model(features, [])

class BehaviorAnalyzer:
    """
    Advanced AI-powered behavior analysis for deep user understanding.
    Features: pattern recognition, anomaly detection, predictive modeling, and adaptive learning.
    """
    
    def __init__(self):
        self.pattern_models = {}
        self.anomaly_detectors = {}
        self.temporal_patterns = defaultdict(list)
        self.behavioral_clusters = {}
        self.adaptation_history = defaultdict(list)
        
        # ML models for pattern recognition
        self.scaler = StandardScaler() if 'StandardScaler' in globals() else None
        self.clustering_models = {}
        
        # Behavioral metrics tracking
        self.interaction_metrics = defaultdict(dict)
        self.cognitive_load_tracker = {}
        self.efficiency_tracker = {}
        
        logging.info("Advanced Behavior Analyzer initialized")
    
    def analyze(self, behavior_data):
        """Comprehensive behavior analysis with AI insights."""
        patterns = {}
        
        interaction_history = behavior_data.get("interaction_history", [])
        
        if interaction_history:
            # Core pattern analysis
            patterns["temporal_patterns"] = self._analyze_temporal_patterns(interaction_history)
            patterns["interaction_patterns"] = self._analyze_interaction_patterns(interaction_history) 
            patterns["cognitive_patterns"] = self._analyze_cognitive_patterns(interaction_history)
            patterns["efficiency_patterns"] = self._analyze_efficiency_patterns(interaction_history)
            
            # Advanced analysis
            patterns["behavioral_clusters"] = self._perform_behavioral_clustering(interaction_history)
            patterns["anomaly_detection"] = self._detect_behavioral_anomalies(interaction_history)
            patterns["predictive_insights"] = self._generate_predictive_insights(interaction_history)
            patterns["adaptation_opportunities"] = self._identify_adaptation_opportunities(interaction_history)
            
            # Contextual analysis
            patterns["context_patterns"] = self._analyze_context_patterns(interaction_history)
            patterns["multi_modal_patterns"] = self._analyze_multimodal_interactions(interaction_history)
            
        return patterns
    
    def _analyze_temporal_patterns(self, history):
        """Advanced temporal pattern analysis."""
        if not history:
            return {}
            
        patterns = {}
        
        # Extract temporal features
        timestamps = []
        durations = []
        intervals = []
        
        for i, session in enumerate(history):
            try:
                dt = datetime.fromisoformat(session["timestamp"])
                timestamps.append(dt)
                durations.append(session.get("duration", 0))
                
                if i > 0:
                    prev_dt = datetime.fromisoformat(history[i-1]["timestamp"])
                    intervals.append((dt - prev_dt).total_seconds())
            except:
                continue
        
        if timestamps:
            # Daily patterns
            hour_patterns = defaultdict(list)
            day_patterns = defaultdict(list)
            
            for dt, duration in zip(timestamps, durations):
                hour_patterns[dt.hour].append(duration)
                day_patterns[dt.weekday()].append(duration)
                
            patterns["hourly_preferences"] = {
                hour: {
                    "avg_duration": np.mean(durations),
                    "frequency": len(durations),
                    "engagement_score": np.mean(durations) * len(durations)
                }
                for hour, durations in hour_patterns.items()
            }
            
            patterns["daily_preferences"] = {
                day: {
                    "avg_duration": np.mean(durations),
                    "frequency": len(durations),
                    "consistency": 1.0 - (np.std(durations) / max(np.mean(durations), 1))
                }
                for day, durations in day_patterns.items()
            }
            
            # Rhythm analysis
            if len(intervals) > 5:
                patterns["interaction_rhythm"] = {
                    "avg_interval": np.mean(intervals),
                    "rhythm_consistency": 1.0 - (np.std(intervals) / max(np.mean(intervals), 1)),
                    "peak_activity_periods": self._find_activity_peaks(timestamps, durations)
                }
        
        return patterns
    
    def _analyze_interaction_patterns(self, history):
        """Analyze interaction patterns and user behavior."""
        if not history:
            return {}
            
        patterns = {}
        
        # Command patterns
        command_types = [s.get("type", "unknown") for s in history]
        command_sequences = []
        error_patterns = []
        success_patterns = []
        
        for i in range(len(history) - 1):
            current = history[i]
            next_cmd = history[i + 1]
            
            command_sequences.append((current.get("type"), next_cmd.get("type")))
            
            if current.get("success", True):
                success_patterns.append(current)
            else:
                error_patterns.append(current)
        
        # Most common command sequences
        from collections import Counter
        sequence_counts = Counter(command_sequences)
        patterns["command_sequences"] = dict(sequence_counts.most_common(10))
        
        # Error analysis
        if error_patterns:
            error_types = [e.get("error_type", "unknown") for e in error_patterns]
            error_contexts = [e.get("context", {}) for e in error_patterns]
            
            patterns["error_analysis"] = {
                "error_rate": len(error_patterns) / len(history),
                "common_errors": dict(Counter(error_types).most_common(5)),
                "error_contexts": self._analyze_error_contexts(error_contexts)
            }
        
        # Success pattern analysis
        if success_patterns:
            patterns["success_patterns"] = {
                "success_rate": len(success_patterns) / len(history),
                "optimal_conditions": self._identify_optimal_conditions(success_patterns)
            }
        
        return patterns
    
    def _analyze_cognitive_patterns(self, history):
        """Analyze cognitive load and learning patterns."""
        patterns = {}
        
        if not history:
            return patterns
            
        # Cognitive load indicators
        response_times = [s.get("response_time", 0) for s in history]
        complexity_scores = [s.get("complexity", 1) for s in history]
        retry_counts = [s.get("retries", 0) for s in history]
        
        if response_times:
            patterns["cognitive_load"] = {
                "avg_response_time": np.mean(response_times),
                "response_consistency": 1.0 - (np.std(response_times) / max(np.mean(response_times), 1)),
                "learning_curve": self._calculate_learning_curve(response_times),
                "cognitive_fatigue": self._detect_cognitive_fatigue(response_times, timestamps=history)
            }
        
        # Learning progression
        if len(history) >= 10:
            patterns["learning_progression"] = {
                "skill_improvement": self._measure_skill_improvement(history),
                "concept_mastery": self._assess_concept_mastery(history),
                "adaptation_speed": self._measure_adaptation_speed(history)
            }
        
        return patterns
    
    def _analyze_efficiency_patterns(self, history):
        """Analyze user efficiency and optimization opportunities."""
        patterns = {}
        
        if not history:
            return patterns
            
        # Efficiency metrics
        task_times = []
        shortcut_usage = []
        automation_opportunities = []
        
        for session in history:
            task_times.append(session.get("task_completion_time", 0))
            
            context = session.get("context", {})
            if context.get("used_shortcuts"):
                shortcut_usage.append(1)
            else:
                shortcut_usage.append(0)
                
            # Identify repetitive tasks
            if session.get("repetitive_task", False):
                automation_opportunities.append(session)
        
        patterns["efficiency_metrics"] = {
            "avg_task_time": np.mean(task_times) if task_times else 0,
            "shortcut_adoption": np.mean(shortcut_usage) if shortcut_usage else 0,
            "efficiency_trend": self._calculate_efficiency_trend(task_times),
            "automation_potential": len(automation_opportunities)
        }
        
        return patterns
    
    def _perform_behavioral_clustering(self, history):
        """Cluster similar behavioral patterns."""
        if len(history) < 10 or not self.scaler:
            return {}
            
        try:
            # Extract features for clustering
            features = []
            for session in history:
                feature_vector = [
                    session.get("duration", 0),
                    session.get("interactions", 0),
                    session.get("response_time", 0),
                    session.get("complexity", 1),
                    session.get("success", True) * 1.0,
                    len(session.get("context", {}))
                ]
                features.append(feature_vector)
            
            features_array = np.array(features)
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Perform clustering
            n_clusters = min(5, len(history) // 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Analyze clusters
            cluster_analysis = {}
            for i in range(n_clusters):
                cluster_indices = np.where(clusters == i)[0]
                cluster_sessions = [history[idx] for idx in cluster_indices]
                
                cluster_analysis[f"cluster_{i}"] = {
                    "size": len(cluster_sessions),
                    "characteristics": self._analyze_cluster_characteristics(cluster_sessions),
                    "representative_sessions": cluster_indices[:3].tolist()
                }
            
            return cluster_analysis
            
        except Exception as e:
            logging.error(f"Behavioral clustering error: {e}")
            return {}
    
    def _detect_behavioral_anomalies(self, history):
        """Detect unusual behavioral patterns."""
        if len(history) < 20:
            return {}
            
        anomalies = []
        
        # Statistical anomaly detection
        durations = [s.get("duration", 0) for s in history]
        response_times = [s.get("response_time", 0) for s in history]
        
        # Z-score based anomaly detection
        for i, session in enumerate(history):
            duration_z = abs((session.get("duration", 0) - np.mean(durations)) / max(np.std(durations), 1))
            response_z = abs((session.get("response_time", 0) - np.mean(response_times)) / max(np.std(response_times), 1))
            
            if duration_z > 2.5 or response_z > 2.5:
                anomalies.append({
                    "session_index": i,
                    "anomaly_type": "statistical_outlier",
                    "z_scores": {"duration": duration_z, "response_time": response_z},
                    "timestamp": session.get("timestamp")
                })
        
        # Pattern-based anomaly detection
        command_sequences = [(history[i].get("type"), history[i+1].get("type")) 
                           for i in range(len(history)-1)]
        
        from collections import Counter
        common_sequences = Counter(command_sequences)
        
        for i, seq in enumerate(command_sequences):
            if common_sequences[seq] == 1 and len(common_sequences) > 10:
                anomalies.append({
                    "session_index": i,
                    "anomaly_type": "unusual_sequence",
                    "sequence": seq,
                    "timestamp": history[i].get("timestamp")
                })
        
        return {
            "anomaly_count": len(anomalies),
            "anomaly_rate": len(anomalies) / len(history),
            "detected_anomalies": anomalies[:10]  # Top 10 anomalies
        }
    
    def _generate_predictive_insights(self, history):
        """Generate predictive insights about user behavior."""
        if len(history) < 15:
            return {}
            
        insights = {}
        
        # Predict next likely actions
        command_transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(history) - 1):
            current_cmd = history[i].get("type", "unknown")
            next_cmd = history[i + 1].get("type", "unknown") 
            command_transitions[current_cmd][next_cmd] += 1
        
        predictions = {}
        for cmd, transitions in command_transitions.items():
            if transitions:
                most_likely = max(transitions.items(), key=lambda x: x[1])
                predictions[cmd] = {
                    "next_command": most_likely[0],
                    "probability": most_likely[1] / sum(transitions.values())
                }
        
        insights["command_predictions"] = predictions
        
        # Predict optimal interaction times
        timestamps = []
        successes = []
        
        for session in history:
            try:
                dt = datetime.fromisoformat(session["timestamp"])
                timestamps.append(dt.hour)
                successes.append(session.get("success", True))
            except:
                continue
        
        if timestamps and successes:
            hourly_success = defaultdict(list)
            for hour, success in zip(timestamps, successes):
                hourly_success[hour].append(success)
            
            optimal_hours = []
            for hour, success_list in hourly_success.items():
                if np.mean(success_list) > 0.8 and len(success_list) >= 3:
                    optimal_hours.append(hour)
            
            insights["optimal_interaction_hours"] = sorted(optimal_hours)
        
        return insights

    # Helper methods for advanced behavioral analysis
    def _find_activity_peaks(self, timestamps, durations):
        """Find peak activity periods."""
        if len(timestamps) < 5:
            return []
            
        # Group by 2-hour windows
        time_windows = defaultdict(list)
        for dt, duration in zip(timestamps, durations):
            window = dt.hour // 2
            time_windows[window].append(duration)
        
        # Find windows with high activity
        peaks = []
        for window, window_durations in time_windows.items():
            if len(window_durations) >= 3:  # Minimum activities for peak
                avg_duration = np.mean(window_durations)
                total_activity = sum(window_durations)
                
                peaks.append({
                    "time_window": f"{window*2:02d}:00-{(window*2+2):02d}:00",
                    "activity_count": len(window_durations),
                    "avg_duration": avg_duration,
                    "total_activity": total_activity,
                    "intensity_score": len(window_durations) * avg_duration
                })
        
        # Sort by intensity
        return sorted(peaks, key=lambda x: x["intensity_score"], reverse=True)[:3]

    def _analyze_error_contexts(self, error_contexts):
        """Analyze contexts where errors commonly occur."""
        if not error_contexts:
            return {}
            
        context_patterns = defaultdict(int)
        
        for context in error_contexts:
            for key, value in context.items():
                context_key = f"{key}:{value}"
                context_patterns[context_key] += 1
                
        # Return most common error contexts
        sorted_contexts = sorted(context_patterns.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_contexts[:5])

    def _identify_optimal_conditions(self, success_patterns):
        """Identify conditions that lead to successful interactions."""
        if not success_patterns:
            return {}
            
        # Analyze contexts of successful interactions
        successful_contexts = defaultdict(list)
        
        for pattern in success_patterns:
            context = pattern.get("context", {})
            for key, value in context.items():
                successful_contexts[key].append(value)
                
        # Find most common successful conditions
        optimal_conditions = {}
        for key, values in successful_contexts.items():
            if values:
                from collections import Counter
                most_common = Counter(values).most_common(1)[0]
                optimal_conditions[key] = {
                    "value": most_common[0],
                    "frequency": most_common[1],
                    "success_rate": most_common[1] / len(success_patterns)
                }
                
        return optimal_conditions

    def _calculate_learning_curve(self, response_times):
        """Calculate learning curve from response times."""
        if len(response_times) < 10:
            return {"trend": "insufficient_data"}
            
        # Split into early and recent periods
        mid_point = len(response_times) // 2
        early_times = response_times[:mid_point]
        recent_times = response_times[mid_point:]
        
        early_avg = np.mean(early_times)
        recent_avg = np.mean(recent_times)
        
        improvement = (early_avg - recent_avg) / early_avg if early_avg > 0 else 0
        
        return {
            "trend": "improving" if improvement > 0.1 else "stable" if abs(improvement) <= 0.1 else "declining",
            "improvement_rate": improvement,
            "early_avg": early_avg,
            "recent_avg": recent_avg
        }

    def _detect_cognitive_fatigue(self, response_times, timestamps):
        """Detect signs of cognitive fatigue."""
        if len(response_times) < 10:
            return {"fatigue_detected": False}
            
        # Look for increasing response times within sessions
        session_groups = []
        current_session = []
        
        for i, session in enumerate(timestamps):
            if i == 0:
                current_session.append(response_times[i])
            else:
                # Check if within same session (< 30 minutes gap)
                try:
                    current_time = datetime.fromisoformat(session["timestamp"])
                    prev_time = datetime.fromisoformat(timestamps[i-1]["timestamp"])
                    
                    if (current_time - prev_time).total_seconds() <= 1800:  # 30 minutes
                        current_session.append(response_times[i])
                    else:
                        if len(current_session) >= 5:
                            session_groups.append(current_session)
                        current_session = [response_times[i]]
                except:
                    current_session.append(response_times[i])
        
        # Add final session
        if len(current_session) >= 5:
            session_groups.append(current_session)
            
        # Analyze sessions for fatigue patterns
        fatigue_sessions = 0
        for session in session_groups:
            if len(session) >= 5:
                # Check if response times increase significantly over session
                first_half = session[:len(session)//2]
                second_half = session[len(session)//2:]
                
                if np.mean(second_half) > np.mean(first_half) * 1.3:  # 30% increase
                    fatigue_sessions += 1
        
        fatigue_rate = fatigue_sessions / len(session_groups) if session_groups else 0
        
        return {
            "fatigue_detected": fatigue_rate > 0.3,
            "fatigue_rate": fatigue_rate,
            "sessions_analyzed": len(session_groups)
        }

    def _measure_skill_improvement(self, history):
        """Measure skill improvement over time."""
        if len(history) < 20:
            return {"improvement": "insufficient_data"}
            
        # Split history into quarters
        quarter_size = len(history) // 4
        quarters = [
            history[i*quarter_size:(i+1)*quarter_size] 
            for i in range(4)
        ]
        
        # Calculate metrics for each quarter
        quarter_metrics = []
        for quarter in quarters:
            success_rate = np.mean([s.get("success", True) for s in quarter])
            avg_response_time = np.mean([s.get("response_time", 0) for s in quarter])
            avg_satisfaction = np.mean([s.get("satisfaction", 0.5) for s in quarter])
            
            quarter_metrics.append({
                "success_rate": success_rate,
                "response_time": avg_response_time,
                "satisfaction": avg_satisfaction,
                "skill_score": (success_rate + avg_satisfaction) / 2
            })
        
        # Analyze improvement trend
        skill_scores = [q["skill_score"] for q in quarter_metrics]
        
        if len(skill_scores) >= 2:
            improvement = skill_scores[-1] - skill_scores[0]
            trend = "improving" if improvement > 0.1 else "stable" if abs(improvement) <= 0.1 else "declining"
        else:
            improvement = 0
            trend = "stable"
            
        return {
            "improvement": trend,
            "improvement_score": improvement,
            "quarter_metrics": quarter_metrics,
            "overall_trajectory": skill_scores
        }

    def _assess_concept_mastery(self, history):
        """Assess mastery of different concepts/commands."""
        concept_mastery = defaultdict(lambda: {"attempts": 0, "successes": 0, "avg_time": 0})
        
        for session in history:
            concept = session.get("type", "unknown")
            success = session.get("success", True)
            response_time = session.get("response_time", 0)
            
            concept_mastery[concept]["attempts"] += 1
            if success:
                concept_mastery[concept]["successes"] += 1
            
            # Update average time
            current_avg = concept_mastery[concept]["avg_time"]
            attempts = concept_mastery[concept]["attempts"]
            concept_mastery[concept]["avg_time"] = (current_avg * (attempts - 1) + response_time) / attempts
        
        # Calculate mastery levels
        mastery_assessment = {}
        for concept, data in concept_mastery.items():
            if data["attempts"] >= 3:  # Minimum attempts for assessment
                success_rate = data["successes"] / data["attempts"]
                
                if success_rate >= 0.9 and data["avg_time"] <= 2.0:
                    mastery_level = "expert"
                elif success_rate >= 0.8 and data["avg_time"] <= 3.0:
                    mastery_level = "proficient"
                elif success_rate >= 0.6:
                    mastery_level = "competent"
                else:
                    mastery_level = "learning"
                    
                mastery_assessment[concept] = {
                    "level": mastery_level,
                    "success_rate": success_rate,
                    "avg_response_time": data["avg_time"],
                    "total_attempts": data["attempts"]
                }
        
        return mastery_assessment

    def _measure_adaptation_speed(self, history):
        """Measure how quickly user adapts to new features."""
        if len(history) < 10:
            return {"adaptation": "insufficient_data"}
            
        # Track first occurrences of each command type
        first_occurrences = {}
        command_learning = defaultdict(list)
        
        for i, session in enumerate(history):
            command_type = session.get("type", "unknown")
            
            if command_type not in first_occurrences:
                first_occurrences[command_type] = i
            
            # Track performance for each command from first occurrence
            attempts_since_first = i - first_occurrences[command_type]
            if attempts_since_first <= 10:  # Look at first 10 attempts
                command_learning[command_type].append({
                    "attempt": attempts_since_first,
                    "success": session.get("success", True),
                    "response_time": session.get("response_time", 0)
                })
        
        # Analyze adaptation patterns
        adaptation_scores = []
        for command, learning_data in command_learning.items():
            if len(learning_data) >= 5:  # Minimum data for analysis
                # Calculate improvement over attempts
                early_attempts = learning_data[:3]
                later_attempts = learning_data[3:]
                
                early_success = np.mean([a["success"] for a in early_attempts])
                later_success = np.mean([a["success"] for a in later_attempts])
                
                adaptation_score = later_success - early_success
                adaptation_scores.append(adaptation_score)
        
        if adaptation_scores:
            avg_adaptation = np.mean(adaptation_scores)
            adaptation_speed = "fast" if avg_adaptation > 0.3 else "moderate" if avg_adaptation > 0.1 else "slow"
        else:
            avg_adaptation = 0
            adaptation_speed = "insufficient_data"
            
        return {
            "adaptation_speed": adaptation_speed,
            "adaptation_score": avg_adaptation,
            "commands_analyzed": len(adaptation_scores)
        }

    def _calculate_efficiency_trend(self, task_times):
        """Calculate efficiency trend over time."""
        if len(task_times) < 10:
            return {"trend": "insufficient_data"}
            
        # Split into two halves
        mid_point = len(task_times) // 2
        early_times = task_times[:mid_point]
        recent_times = task_times[mid_point:]
        
        early_avg = np.mean(early_times)
        recent_avg = np.mean(recent_times)
        
        if early_avg > 0:
            efficiency_improvement = (early_avg - recent_avg) / early_avg
        else:
            efficiency_improvement = 0
            
        trend = "improving" if efficiency_improvement > 0.1 else "stable" if abs(efficiency_improvement) <= 0.1 else "declining"
        
        return {
            "trend": trend,
            "improvement_rate": efficiency_improvement,
            "early_avg_time": early_avg,
            "recent_avg_time": recent_avg
        }

    def _analyze_cluster_characteristics(self, cluster_sessions):
        """Analyze characteristics of a behavioral cluster."""
        if not cluster_sessions:
            return {}
            
        # Calculate cluster characteristics
        durations = [s.get("duration", 0) for s in cluster_sessions]
        response_times = [s.get("response_time", 0) for s in cluster_sessions]
        success_rates = [s.get("success", True) for s in cluster_sessions]
        complexities = [s.get("complexity", 1) for s in cluster_sessions]
        
        command_types = [s.get("type", "unknown") for s in cluster_sessions]
        from collections import Counter
        common_commands = Counter(command_types).most_common(3)
        
        return {
            "avg_duration": np.mean(durations),
            "avg_response_time": np.mean(response_times),
            "success_rate": np.mean(success_rates),
            "avg_complexity": np.mean(complexities),
            "common_commands": dict(common_commands),
            "session_count": len(cluster_sessions)
        }
        
    def _find_peak_hours(self, history):
        """Find peak usage hours."""
        hour_counts = defaultdict(int)
        
        for session in history:
            if "timestamp" in session:
                try:
                    dt = datetime.fromisoformat(session["timestamp"])
                    hour_counts[dt.hour] += 1
                except:
                    continue
                    
        if hour_counts:
            peak_hour = max(hour_counts.items(), key=lambda x: x[1])
            return {"peak_hour": peak_hour[0], "usage_count": peak_hour[1]}
            
        return {}
        
    def _analyze_session_patterns(self, history):
        """Analyze session duration and interaction patterns."""
        if not history:
            return {}
            
        durations = [s.get("duration", 0) for s in history]
        interactions = [s.get("interactions", 0) for s in history]
        
        return {
            "avg_session_duration": np.mean(durations),
            "avg_interactions_per_session": np.mean(interactions),
            "session_consistency": 1.0 - (np.std(durations) / max(np.mean(durations), 1))
        }
        
    def _analyze_interaction_trends(self, history):
        """Analyze trends in user interactions."""
        if len(history) < 5:
            return {}
            
        # Calculate trend over time
        recent = history[-10:]
        older = history[-20:-10] if len(history) >= 20 else []
        
        if older:
            recent_avg = np.mean([s.get("interactions", 0) for s in recent])
            older_avg = np.mean([s.get("interactions", 0) for s in older])
            
            trend = "increasing" if recent_avg > older_avg else "decreasing"
            return {"interaction_trend": trend, "change_rate": abs(recent_avg - older_avg)}
            
        return {}
    
    def extract_patterns(self, recent_data):
        """Extract behavioral patterns from recent data."""
        patterns = {}
        
        if recent_data:
            # Time patterns
            times = []
            for session in recent_data:
                if "timestamp" in session:
                    try:
                        dt = datetime.fromisoformat(session["timestamp"])
                        times.append(dt.hour)
                    except:
                        continue
                        
            if times:
                patterns["preferred_hours"] = list(set(times))
                patterns["most_active_hour"] = max(set(times), key=times.count)
                
        return patterns

class PreferenceLearner:
    """
    Advanced AI-powered preference learning system.
    Features: contextual preferences, adaptive learning, preference prediction, and personalization.
    """
    
    def __init__(self):
        self.preference_models = defaultdict(dict)
        self.contextual_preferences = defaultdict(lambda: defaultdict(dict))
        self.preference_weights = defaultdict(dict)
        self.learning_history = defaultdict(list)
        
        # Advanced preference tracking
        self.multi_dimensional_preferences = defaultdict(dict)
        self.temporal_preferences = defaultdict(list)
        self.interaction_preferences = defaultdict(dict)
        self.adaptation_preferences = defaultdict(dict)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        self.confidence_threshold = 0.7
        self.min_samples_for_learning = 3
        
        logging.info("Advanced Preference Learner initialized")
    
    def update(self, user_id, interaction):
        """Advanced preference learning from multi-dimensional interactions."""
        # Extract comprehensive interaction features
        features = self._extract_interaction_features(interaction)
        
        # Learn preferences across multiple dimensions
        self._learn_command_preferences(user_id, interaction, features)
        self._learn_contextual_preferences(user_id, interaction, features)
        self._learn_temporal_preferences(user_id, interaction, features)
        self._learn_interface_preferences(user_id, interaction, features)
        self._learn_behavioral_preferences(user_id, interaction, features)
        
        # Update preference weights based on success/feedback
        self._update_preference_weights(user_id, interaction, features)
        
        # Store learning history
        self.learning_history[user_id].append({
            "timestamp": interaction.get("timestamp", datetime.now().isoformat()),
            "features": features,
            "feedback": interaction.get("feedback", None),
            "success": interaction.get("success", False)
        })
        
        # Limit history size
        if len(self.learning_history[user_id]) > 1000:
            self.learning_history[user_id] = self.learning_history[user_id][-1000:]
    
    def _extract_interaction_features(self, interaction):
        """Extract comprehensive features from interaction."""
        features = {
            "command_type": interaction.get("type", "unknown"),
            "success": interaction.get("success", False),
            "response_time": interaction.get("response_time", 0),
            "context": interaction.get("context", {}),
            "modality": interaction.get("modality", "unknown"),  # voice, gesture, touch, etc.
            "complexity": interaction.get("complexity", 1),
            "user_effort": interaction.get("user_effort", 1),
            "satisfaction": interaction.get("satisfaction", 0.5),
            "retry_count": interaction.get("retries", 0),
            "timestamp": interaction.get("timestamp"),
            "environment": interaction.get("environment", {}),
            "device_context": interaction.get("device_context", {})
        }
        
        # Extract temporal context
        try:
            dt = datetime.fromisoformat(features["timestamp"])
            features["hour"] = dt.hour
            features["day_of_week"] = dt.weekday()
            features["is_weekend"] = dt.weekday() >= 5
        except:
            features["hour"] = 12
            features["day_of_week"] = 0
            features["is_weekend"] = False
        
        return features
    
    def _learn_command_preferences(self, user_id, interaction, features):
        """Learn preferences for specific commands."""
        command_type = features["command_type"]
        
        if command_type not in self.preference_models[user_id]:
            self.preference_models[user_id][command_type] = {
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "preferred_modalities": defaultdict(int),
                "satisfaction_score": 0.0,
                "usage_count": 0,
                "last_updated": datetime.now().isoformat()
            }
        
        model = self.preference_models[user_id][command_type]
        
        # Update metrics with exponential moving average
        alpha = self.learning_rate
        model["success_rate"] = (1 - alpha) * model["success_rate"] + alpha * features["success"]
        model["avg_response_time"] = (1 - alpha) * model["avg_response_time"] + alpha * features["response_time"]
        model["satisfaction_score"] = (1 - alpha) * model["satisfaction_score"] + alpha * features["satisfaction"]
        model["usage_count"] += 1
        model["last_updated"] = datetime.now().isoformat()
        
        # Track modality preferences
        modality = features["modality"]
        model["preferred_modalities"][modality] += 1
    
    def _learn_contextual_preferences(self, user_id, interaction, features):
        """Learn context-dependent preferences."""
        context = features["context"]
        command_type = features["command_type"]
        
        # Learn preferences for different contexts
        for context_key, context_value in context.items():
            context_signature = f"{context_key}:{context_value}"
            
            if context_signature not in self.contextual_preferences[user_id][command_type]:
                self.contextual_preferences[user_id][command_type][context_signature] = {
                    "success_rate": 0.0,
                    "preference_score": 0.0,
                    "count": 0
                }
            
            ctx_pref = self.contextual_preferences[user_id][command_type][context_signature]
            alpha = self.learning_rate
            
            ctx_pref["success_rate"] = (1 - alpha) * ctx_pref["success_rate"] + alpha * features["success"]
            ctx_pref["preference_score"] = (1 - alpha) * ctx_pref["preference_score"] + alpha * features["satisfaction"]
            ctx_pref["count"] += 1
    
    def _learn_temporal_preferences(self, user_id, interaction, features):
        """Learn time-based preferences."""
        hour = features["hour"]
        day_of_week = features["day_of_week"]
        is_weekend = features["is_weekend"]
        command_type = features["command_type"]
        
        temporal_key = f"hour_{hour}"
        day_key = f"day_{day_of_week}"
        weekend_key = f"weekend_{is_weekend}"
        
        for temp_key in [temporal_key, day_key, weekend_key]:
            if temp_key not in self.multi_dimensional_preferences[user_id]:
                self.multi_dimensional_preferences[user_id][temp_key] = defaultdict(dict)
            
            if command_type not in self.multi_dimensional_preferences[user_id][temp_key]:
                self.multi_dimensional_preferences[user_id][temp_key][command_type] = {
                    "preference_score": 0.0,
                    "usage_frequency": 0,
                    "success_rate": 0.0
                }
            
            temp_pref = self.multi_dimensional_preferences[user_id][temp_key][command_type]
            alpha = self.learning_rate
            
            temp_pref["preference_score"] = (1 - alpha) * temp_pref["preference_score"] + alpha * features["satisfaction"]
            temp_pref["usage_frequency"] += 1
            temp_pref["success_rate"] = (1 - alpha) * temp_pref["success_rate"] + alpha * features["success"]
    
    def _learn_interface_preferences(self, user_id, interaction, features):
        """Learn interface and interaction mode preferences."""
        if "interface_preferences" not in self.interaction_preferences[user_id]:
            self.interaction_preferences[user_id]["interface_preferences"] = {
                "modality_preferences": defaultdict(float),
                "response_format_preferences": defaultdict(float),
                "verbosity_preferences": defaultdict(float),
                "interaction_speed_preferences": defaultdict(float)
            }
        
        interface_prefs = self.interaction_preferences[user_id]["interface_preferences"]
        
        # Learn modality preferences
        modality = features["modality"]
        satisfaction = features["satisfaction"]
        success = features["success"]
        
        preference_score = (satisfaction + success) / 2
        alpha = self.learning_rate
        
        current_score = interface_prefs["modality_preferences"][modality]
        interface_prefs["modality_preferences"][modality] = (1 - alpha) * current_score + alpha * preference_score
        
        # Learn response time preferences
        response_time = features["response_time"]
        if response_time > 0:
            if response_time < 1.0:
                speed_category = "fast"
            elif response_time < 3.0:
                speed_category = "medium"
            else:
                speed_category = "slow"
            
            current_speed_score = interface_prefs["interaction_speed_preferences"][speed_category]
            interface_prefs["interaction_speed_preferences"][speed_category] = (1 - alpha) * current_speed_score + alpha * preference_score
    
    def _learn_behavioral_preferences(self, user_id, interaction, features):
        """Learn behavioral and adaptive preferences."""
        if "behavioral_preferences" not in self.adaptation_preferences[user_id]:
            self.adaptation_preferences[user_id]["behavioral_preferences"] = {
                "error_tolerance": 0.5,
                "help_seeking_tendency": 0.5,
                "exploration_tendency": 0.5,
                "automation_preference": 0.5,
                "customization_preference": 0.5
            }
        
        behavioral_prefs = self.adaptation_preferences[user_id]["behavioral_preferences"]
        
        # Learn error tolerance
        if features["retry_count"] > 0:
            error_tolerance_indicator = min(1.0, 1.0 / (features["retry_count"] + 1))
            alpha = self.learning_rate
            behavioral_prefs["error_tolerance"] = (1 - alpha) * behavioral_prefs["error_tolerance"] + alpha * error_tolerance_indicator
        
        # Learn exploration tendency based on command variety
        command_variety = len(set([cmd for cmd in self.preference_models[user_id].keys()]))
        exploration_score = min(1.0, command_variety / 20.0)  # Normalize by expected max commands
        behavioral_prefs["exploration_tendency"] = (1 - alpha) * behavioral_prefs["exploration_tendency"] + alpha * exploration_score
    
    def _update_preference_weights(self, user_id, interaction, features):
        """Update preference weights based on feedback."""
        command_type = features["command_type"]
        success = features["success"]
        satisfaction = features["satisfaction"]
        
        if command_type not in self.preference_weights[user_id]:
            self.preference_weights[user_id][command_type] = {
                "importance": 0.5,
                "confidence": 0.0,
                "stability": 0.5
            }
        
        weights = self.preference_weights[user_id][command_type]
        
        # Update importance based on usage frequency and satisfaction
        usage_count = self.preference_models[user_id].get(command_type, {}).get("usage_count", 0)
        importance_score = min(1.0, (usage_count * satisfaction) / 10.0)
        
        alpha = self.learning_rate
        weights["importance"] = (1 - alpha) * weights["importance"] + alpha * importance_score
        
        # Update confidence based on consistency
        if success:
            weights["confidence"] = min(1.0, weights["confidence"] + 0.1)
        else:
            weights["confidence"] = max(0.0, weights["confidence"] - 0.05)
        
        # Update stability (how much preferences change)
        recent_interactions = self.learning_history[user_id][-10:]
        if len(recent_interactions) >= 5:
            recent_satisfactions = [int.get("feedback", {}).get("satisfaction", 0.5) for int in recent_interactions if int.get("features", {}).get("command_type") == command_type]
            if recent_satisfactions:
                stability = 1.0 - (np.std(recent_satisfactions) / max(np.mean(recent_satisfactions), 0.1))
                weights["stability"] = (1 - alpha) * weights["stability"] + alpha * stability
    
    def get_preferences(self, user_id, context=None):
        """Get comprehensive user preferences with context awareness."""
        if user_id not in self.preference_models:
            return self._get_default_preferences()
        
        preferences = {
            "command_preferences": self.preference_models[user_id],
            "contextual_preferences": dict(self.contextual_preferences[user_id]),
            "temporal_preferences": dict(self.multi_dimensional_preferences[user_id]),
            "interface_preferences": self.interaction_preferences.get(user_id, {}),
            "behavioral_preferences": self.adaptation_preferences.get(user_id, {}),
            "preference_weights": self.preference_weights[user_id]
        }
        
        # Apply context if provided
        if context:
            preferences = self._apply_contextual_filtering(preferences, context)
        
        return preferences
    
    def predict_preference(self, user_id, command_type, context=None):
        """Predict user preference for a specific command in given context."""
        if user_id not in self.preference_models:
            return {"preference_score": 0.5, "confidence": 0.0}
        
        base_preferences = self.preference_models[user_id].get(command_type, {})
        
        # Base preference score
        base_score = base_preferences.get("satisfaction_score", 0.5)
        confidence = self.preference_weights[user_id].get(command_type, {}).get("confidence", 0.0)
        
        # Apply contextual adjustments
        if context and command_type in self.contextual_preferences[user_id]:
            contextual_adjustments = []
            
            for context_key, context_value in context.items():
                context_signature = f"{context_key}:{context_value}"
                if context_signature in self.contextual_preferences[user_id][command_type]:
                    ctx_pref = self.contextual_preferences[user_id][command_type][context_signature]
                    if ctx_pref["count"] >= self.min_samples_for_learning:
                        contextual_adjustments.append(ctx_pref["preference_score"])
            
            if contextual_adjustments:
                context_score = np.mean(contextual_adjustments)
                adjusted_score = 0.7 * base_score + 0.3 * context_score
            else:
                adjusted_score = base_score
        else:
            adjusted_score = base_score
        
        return {
            "preference_score": adjusted_score,
            "confidence": confidence,
            "base_score": base_score,
            "context_adjusted": context is not None
        }
    
    def _get_default_preferences(self):
        """Get default preferences for new users."""
        return {
            "command_preferences": {},
            "contextual_preferences": {},
            "temporal_preferences": {},
            "interface_preferences": {
                "modality_preferences": {"voice": 0.6, "gesture": 0.4, "touch": 0.3},
                "response_format_preferences": {"conversational": 0.7, "brief": 0.3},
                "verbosity_preferences": {"detailed": 0.4, "concise": 0.6},
                "interaction_speed_preferences": {"medium": 0.6, "fast": 0.3, "slow": 0.1}
            },
            "behavioral_preferences": {
                "error_tolerance": 0.5,
                "help_seeking_tendency": 0.5,
                "exploration_tendency": 0.5,
                "automation_preference": 0.5
            },
            "preference_weights": {}
        }
    
    def _apply_contextual_filtering(self, preferences, context):
        """Apply contextual filtering to preferences."""
        filtered_preferences = preferences.copy()
        
        # Extract current time context
        current_time = context.get("current_time", datetime.now())
        if isinstance(current_time, str):
            current_time = datetime.fromisoformat(current_time)
        
        current_hour = current_time.hour
        current_day = current_time.weekday()
        is_weekend = current_day >= 5
        
        # Apply temporal context filtering
        temporal_prefs = preferences.get("temporal_preferences", {})
        
        # Boost preferences that align with current temporal context
        for temp_key in [f"hour_{current_hour}", f"day_{current_day}", f"weekend_{is_weekend}"]:
            if temp_key in temporal_prefs:
                # Add temporal boost to command preferences
                for cmd_type, temp_data in temporal_prefs[temp_key].items():
                    if temp_data.get("usage_frequency", 0) >= 2:  # Minimum usage threshold
                        boost = temp_data.get("preference_score", 0.5) * 0.2  # 20% boost
                        if cmd_type in filtered_preferences["command_preferences"]:
                            current_score = filtered_preferences["command_preferences"][cmd_type].get("satisfaction_score", 0.5)
                            filtered_preferences["command_preferences"][cmd_type]["satisfaction_score"] = min(1.0, current_score + boost)
        
        return filtered_preferences

class ActivityTracker:
    """
    Comprehensive activity tracking and pattern analysis system.
    Features: multi-dimensional activity analysis, productivity metrics, goal tracking, and insights.
    """
    
    def __init__(self):
        self.activity_data = defaultdict(list)
        self.activity_sessions = defaultdict(list)
        self.productivity_metrics = defaultdict(dict)
        self.goal_tracking = defaultdict(dict)
        self.activity_insights = defaultdict(dict)
        
        # Advanced tracking features
        self.workflow_patterns = defaultdict(list)
        self.efficiency_tracking = defaultdict(dict)
        self.learning_progress = defaultdict(dict)
        self.habit_formation = defaultdict(dict)
        
        # Analytics parameters
        self.session_timeout = 1800  # 30 minutes
        self.productivity_window = 7  # days for productivity analysis
        self.habit_formation_threshold = 21  # days
        
        logging.info("Advanced Activity Tracker initialized")
    
    def record_activity(self, user_id, interaction):
        """Advanced activity recording with comprehensive data capture."""
        timestamp = interaction.get("timestamp", datetime.now().isoformat())
        
        # Create rich activity record
        activity = {
            "timestamp": timestamp,
            "type": interaction.get("type", "unknown"),
            "success": interaction.get("success", True),
            "duration": interaction.get("response_time", 0),
            "task_completion_time": interaction.get("task_completion_time", 0),
            "complexity": interaction.get("complexity", 1),
            "context": interaction.get("context", {}),
            "modality": interaction.get("modality", "unknown"),
            "user_effort": interaction.get("user_effort", 1),
            "satisfaction": interaction.get("satisfaction", 0.5),
            "retry_count": interaction.get("retries", 0),
            "goal_contribution": interaction.get("goal_contribution", 0),
            "environment": interaction.get("environment", {}),
            "workflow_step": interaction.get("workflow_step", None)
        }
        
        # Add activity to records
        self.activity_data[user_id].append(activity)
        
        # Session management
        self._update_activity_sessions(user_id, activity)
        
        # Real-time analysis updates
        self._update_productivity_metrics(user_id, activity)
        self._track_goal_progress(user_id, activity)
        self._analyze_workflow_patterns(user_id, activity)
        self._track_habit_formation(user_id, activity)
        
        # Maintain data size limits
        if len(self.activity_data[user_id]) > 2000:
            self.activity_data[user_id] = self.activity_data[user_id][-2000:]
        
        logging.debug(f"Activity recorded for user {user_id}: {activity['type']}")
    
    def _update_activity_sessions(self, user_id, activity):
        """Manage activity sessions for session-based analysis."""
        current_time = datetime.fromisoformat(activity["timestamp"])
        
        # Check if this continues an existing session or starts a new one
        if self.activity_sessions[user_id]:
            last_session = self.activity_sessions[user_id][-1]
            last_activity_time = datetime.fromisoformat(last_session["activities"][-1]["timestamp"])
            
            if (current_time - last_activity_time).total_seconds() <= self.session_timeout:
                # Continue existing session
                last_session["activities"].append(activity)
                last_session["end_time"] = current_time
                last_session["duration"] = (current_time - last_session["start_time"]).total_seconds()
                last_session["activity_count"] += 1
            else:
                # Start new session
                self._create_new_session(user_id, activity, current_time)
        else:
            # First session
            self._create_new_session(user_id, activity, current_time)
    
    def _create_new_session(self, user_id, activity, current_time):
        """Create a new activity session."""
        session = {
            "session_id": len(self.activity_sessions[user_id]) + 1,
            "start_time": current_time,
            "end_time": current_time,
            "duration": 0,
            "activities": [activity],
            "activity_count": 1,
            "primary_goal": activity.get("goal_contribution", 0),
            "context": activity.get("context", {}),
            "productivity_score": 0.0
        }
        
        self.activity_sessions[user_id].append(session)
        
        # Limit session history
        if len(self.activity_sessions[user_id]) > 100:
            self.activity_sessions[user_id] = self.activity_sessions[user_id][-100:]
    
    def _update_productivity_metrics(self, user_id, activity):
        """Update real-time productivity metrics."""
        current_date = datetime.fromisoformat(activity["timestamp"]).date().isoformat()
        
        if current_date not in self.productivity_metrics[user_id]:
            self.productivity_metrics[user_id][current_date] = {
                "total_activities": 0,
                "successful_activities": 0,
                "total_time": 0.0,
                "productive_time": 0.0,
                "efficiency_score": 0.0,
                "goal_progress": 0.0,
                "satisfaction_score": 0.0,
                "focus_score": 0.0
            }
        
        daily_metrics = self.productivity_metrics[user_id][current_date]
        
        # Update metrics
        daily_metrics["total_activities"] += 1
        if activity["success"]:
            daily_metrics["successful_activities"] += 1
        
        daily_metrics["total_time"] += activity["duration"]
        
        # Calculate productive time (successful activities with good satisfaction)
        if activity["success"] and activity["satisfaction"] > 0.6:
            daily_metrics["productive_time"] += activity["duration"]
        
        daily_metrics["goal_progress"] += activity["goal_contribution"]
        daily_metrics["satisfaction_score"] = (
            (daily_metrics["satisfaction_score"] * (daily_metrics["total_activities"] - 1) + activity["satisfaction"]) /
            daily_metrics["total_activities"]
        )
        
        # Calculate efficiency score
        if daily_metrics["total_time"] > 0:
            daily_metrics["efficiency_score"] = daily_metrics["productive_time"] / daily_metrics["total_time"]
        
        # Calculate focus score (based on task completion without retries)
        focus_contribution = 1.0 if activity["retry_count"] == 0 else max(0.0, 1.0 - activity["retry_count"] * 0.2)
        daily_metrics["focus_score"] = (
            (daily_metrics["focus_score"] * (daily_metrics["total_activities"] - 1) + focus_contribution) /
            daily_metrics["total_activities"]
        )
    
    def _track_goal_progress(self, user_id, activity):
        """Track progress towards user goals."""
        goal_contribution = activity.get("goal_contribution", 0)
        
        if goal_contribution > 0:
            activity_type = activity["type"]
            current_date = datetime.fromisoformat(activity["timestamp"]).date().isoformat()
            
            if activity_type not in self.goal_tracking[user_id]:
                self.goal_tracking[user_id][activity_type] = {
                    "total_contribution": 0.0,
                    "activity_count": 0,
                    "weekly_progress": defaultdict(float),
                    "monthly_progress": defaultdict(float),
                    "success_rate": 0.0,
                    "average_contribution": 0.0
                }
            
            goal_data = self.goal_tracking[user_id][activity_type]
            goal_data["total_contribution"] += goal_contribution
            goal_data["activity_count"] += 1
            goal_data["average_contribution"] = goal_data["total_contribution"] / goal_data["activity_count"]
            
            # Update success rate
            previous_success_rate = goal_data["success_rate"]
            current_success = 1.0 if activity["success"] else 0.0
            goal_data["success_rate"] = (
                (previous_success_rate * (goal_data["activity_count"] - 1) + current_success) /
                goal_data["activity_count"]
            )
            
            # Track weekly and monthly progress
            current_dt = datetime.fromisoformat(activity["timestamp"])
            week_key = f"{current_dt.year}-W{current_dt.isocalendar()[1]}"
            month_key = f"{current_dt.year}-{current_dt.month:02d}"
            
            goal_data["weekly_progress"][week_key] += goal_contribution
            goal_data["monthly_progress"][month_key] += goal_contribution
    
    def _analyze_workflow_patterns(self, user_id, activity):
        """Analyze workflow patterns and sequences."""
        if len(self.activity_data[user_id]) >= 2:
            # Get previous activity for sequence analysis
            previous_activity = self.activity_data[user_id][-2]
            current_activity = activity
            
            sequence = (previous_activity["type"], current_activity["type"])
            
            # Find or create workflow pattern entry
            pattern_found = False
            for workflow in self.workflow_patterns[user_id]:
                if workflow["sequence"] == sequence:
                    workflow["count"] += 1
                    workflow["avg_time_between"] = (
                        (workflow["avg_time_between"] * (workflow["count"] - 1) + 
                         (datetime.fromisoformat(current_activity["timestamp"]) - 
                          datetime.fromisoformat(previous_activity["timestamp"])).total_seconds()) /
                        workflow["count"]
                    )
                    workflow["success_rate"] = (
                        (workflow["success_rate"] * (workflow["count"] - 1) + 
                         (1.0 if current_activity["success"] else 0.0)) /
                        workflow["count"]
                    )
                    pattern_found = True
                    break
            
            if not pattern_found:
                time_between = (datetime.fromisoformat(current_activity["timestamp"]) - 
                               datetime.fromisoformat(previous_activity["timestamp"])).total_seconds()
                
                self.workflow_patterns[user_id].append({
                    "sequence": sequence,
                    "count": 1,
                    "avg_time_between": time_between,
                    "success_rate": 1.0 if current_activity["success"] else 0.0,
                    "last_seen": current_activity["timestamp"]
                })
            
            # Limit workflow patterns
            if len(self.workflow_patterns[user_id]) > 50:
                # Keep most frequent patterns
                self.workflow_patterns[user_id].sort(key=lambda x: x["count"], reverse=True)
                self.workflow_patterns[user_id] = self.workflow_patterns[user_id][:50]
    
    def _track_habit_formation(self, user_id, activity):
        """Track habit formation and consistency."""
        activity_type = activity["type"]
        current_date = datetime.fromisoformat(activity["timestamp"]).date()
        
        if activity_type not in self.habit_formation[user_id]:
            self.habit_formation[user_id][activity_type] = {
                "streak_count": 0,
                "max_streak": 0,
                "last_activity_date": None,
                "habit_strength": 0.0,
                "consistency_score": 0.0,
                "daily_occurrences": defaultdict(int)
            }
        
        habit_data = self.habit_formation[user_id][activity_type]
        
        # Update daily occurrence
        habit_data["daily_occurrences"][current_date.isoformat()] += 1
        
        # Update streak
        if habit_data["last_activity_date"]:
            last_date = datetime.fromisoformat(habit_data["last_activity_date"]).date()
            days_diff = (current_date - last_date).days
            
            if days_diff == 1:
                # Consecutive day - continue streak
                habit_data["streak_count"] += 1
            elif days_diff == 0:
                # Same day - no change to streak
                pass
            else:
                # Gap in activity - reset streak
                habit_data["streak_count"] = 1
        else:
            habit_data["streak_count"] = 1
        
        habit_data["last_activity_date"] = current_date.isoformat()
        habit_data["max_streak"] = max(habit_data["max_streak"], habit_data["streak_count"])
        
        # Calculate habit strength (based on streak and consistency)
        days_tracked = len(habit_data["daily_occurrences"])
        if days_tracked > 0:
            habit_data["consistency_score"] = min(1.0, habit_data["streak_count"] / self.habit_formation_threshold)
            habit_data["habit_strength"] = (
                0.6 * habit_data["consistency_score"] + 
                0.4 * min(1.0, days_tracked / 30.0)  # 30 days for full strength
            )
    
    def get_patterns(self, user_id, behavior_data=None):
        """Get comprehensive activity patterns and insights."""
        if user_id not in self.activity_data:
            return {}
        
        activities = self.activity_data[user_id]
        patterns = {}
        
        # Basic activity analysis
        patterns["activity_summary"] = self._get_activity_summary(activities)
        patterns["temporal_patterns"] = self._get_temporal_patterns(activities)
        patterns["productivity_analysis"] = self._get_productivity_analysis(user_id)
        patterns["goal_analysis"] = self._get_goal_analysis(user_id)
        patterns["workflow_analysis"] = self._get_workflow_analysis(user_id)
        patterns["habit_analysis"] = self._get_habit_analysis(user_id)
        patterns["session_analysis"] = self._get_session_analysis(user_id)
        
        # Advanced insights
        patterns["efficiency_insights"] = self._generate_efficiency_insights(user_id)
        patterns["improvement_suggestions"] = self._generate_improvement_suggestions(user_id)
        patterns["trend_analysis"] = self._perform_trend_analysis(user_id)
        
        return patterns
    
    def _get_activity_summary(self, activities):
        """Get basic activity summary statistics."""
        if not activities:
            return {}
        
        activity_types = [a["type"] for a in activities]
        success_rates = [a["success"] for a in activities]
        durations = [a["duration"] for a in activities]
        satisfactions = [a["satisfaction"] for a in activities]
        
        from collections import Counter
        type_counts = Counter(activity_types)
        
        return {
            "total_activities": len(activities),
            "unique_activity_types": len(set(activity_types)),
            "overall_success_rate": np.mean(success_rates),
            "average_duration": np.mean(durations),
            "average_satisfaction": np.mean(satisfactions),
            "most_common_activities": dict(type_counts.most_common(10)),
            "activity_diversity": len(set(activity_types)) / len(activities) if activities else 0
        }
    
    def _get_temporal_patterns(self, activities):
        """Analyze temporal patterns in activities."""
        if not activities:
            return {}
        
        hourly_activity = defaultdict(int)
        daily_activity = defaultdict(int)
        weekly_activity = defaultdict(int)
        
        for activity in activities:
            try:
                dt = datetime.fromisoformat(activity["timestamp"])
                hourly_activity[dt.hour] += 1
                daily_activity[dt.date().isoformat()] += 1
                weekly_activity[dt.weekday()] += 1
            except:
                continue
        
        # Find peak activity periods
        peak_hour = max(hourly_activity.items(), key=lambda x: x[1]) if hourly_activity else (12, 0)
        peak_day = max(weekly_activity.items(), key=lambda x: x[1]) if weekly_activity else (0, 0)
        
        return {
            "hourly_distribution": dict(hourly_activity),
            "daily_distribution": dict(daily_activity),
            "weekly_distribution": dict(weekly_activity),
            "peak_hour": {"hour": peak_hour[0], "count": peak_hour[1]},
            "peak_weekday": {"day": peak_day[0], "count": peak_day[1]},
            "activity_span_days": len(daily_activity)
        }
    
    def _get_productivity_analysis(self, user_id):
        """Get comprehensive productivity analysis."""
        return dict(self.productivity_metrics[user_id])
    
    def _get_goal_analysis(self, user_id):
        """Get goal tracking analysis."""
        return dict(self.goal_tracking[user_id])
    
    def _get_workflow_analysis(self, user_id):
        """Get workflow pattern analysis."""
        if not self.workflow_patterns[user_id]:
            return {}
        
        # Sort patterns by frequency
        sorted_patterns = sorted(self.workflow_patterns[user_id], key=lambda x: x["count"], reverse=True)
        
        return {
            "total_patterns": len(sorted_patterns),
            "most_common_patterns": sorted_patterns[:10],
            "average_success_rate": np.mean([p["success_rate"] for p in sorted_patterns]),
            "workflow_efficiency": np.mean([p["success_rate"] for p in sorted_patterns if p["count"] >= 3])
        }
    
    def _get_habit_analysis(self, user_id):
        """Get habit formation analysis."""
        habits = self.habit_formation[user_id]
        
        if not habits:
            return {}
        
        # Analyze habit strength
        strong_habits = {k: v for k, v in habits.items() if v["habit_strength"] > 0.7}
        developing_habits = {k: v for k, v in habits.items() if 0.3 < v["habit_strength"] <= 0.7}
        weak_habits = {k: v for k, v in habits.items() if v["habit_strength"] <= 0.3}
        
        return {
            "total_tracked_habits": len(habits),
            "strong_habits": len(strong_habits),
            "developing_habits": len(developing_habits),
            "weak_habits": len(weak_habits),
            "habit_details": {
                "strong": dict(strong_habits),
                "developing": dict(developing_habits),
                "weak": dict(weak_habits)
            },
            "overall_habit_strength": np.mean([h["habit_strength"] for h in habits.values()])
        }
    
    def _get_session_analysis(self, user_id):
        """Analyze activity sessions."""
        sessions = self.activity_sessions[user_id]
        
        if not sessions:
            return {}
        
        session_durations = [s["duration"] for s in sessions]
        session_activities = [s["activity_count"] for s in sessions]
        
        return {
            "total_sessions": len(sessions),
            "average_session_duration": np.mean(session_durations),
            "average_activities_per_session": np.mean(session_activities),
            "longest_session": max(session_durations) if session_durations else 0,
            "most_productive_session": max(sessions, key=lambda x: x["activity_count"]) if sessions else None,
            "session_consistency": 1.0 - (np.std(session_durations) / max(np.mean(session_durations), 1)) if session_durations else 0
        }

    # Helper methods for ActivityTracker insights
    def _generate_efficiency_insights(self, user_id):
        """Generate insights about user efficiency."""
        insights = []
        
        # Analyze productivity metrics
        productivity_data = self.productivity_metrics.get(user_id, {})
        if productivity_data:
            recent_days = sorted(productivity_data.keys())[-7:]  # Last 7 days
            
            if recent_days:
                recent_efficiency = [productivity_data[day]["efficiency_score"] for day in recent_days if "efficiency_score" in productivity_data[day]]
                
                if recent_efficiency:
                    avg_efficiency = np.mean(recent_efficiency)
                    
                    if avg_efficiency < 0.6:
                        insights.append({
                            "type": "efficiency",
                            "priority": "high",
                            "message": f"Your efficiency score is {avg_efficiency:.1%}. Consider taking breaks and focusing on high-value tasks.",
                            "suggestions": ["Take regular breaks", "Prioritize important tasks", "Minimize distractions"]
                        })
                    elif avg_efficiency > 0.8:
                        insights.append({
                            "type": "efficiency",
                            "priority": "low",
                            "message": f"Excellent efficiency! You're performing at {avg_efficiency:.1%}.",
                            "suggestions": ["Keep up the great work", "Share your strategies with others"]
                        })
        
        # Analyze goal progress
        goal_data = self.goal_tracking.get(user_id, {})
        for goal_type, data in goal_data.items():
            if data["success_rate"] < 0.5:
                insights.append({
                    "type": "goal_performance",
                    "priority": "medium",
                    "message": f"Low success rate for {goal_type}: {data['success_rate']:.1%}",
                    "suggestions": ["Break down complex tasks", "Practice more", "Seek help if needed"]
                })
        
        return insights

    def _generate_improvement_suggestions(self, user_id):
        """Generate personalized improvement suggestions."""
        suggestions = []
        
        # Analyze habit formation
        habits = self.habit_formation.get(user_id, {})
        weak_habits = {k: v for k, v in habits.items() if v["habit_strength"] < 0.3}
        
        for habit_type, habit_data in weak_habits.items():
            suggestions.append({
                "type": "habit_improvement",
                "priority": "medium",
                "action": f"Strengthen {habit_type} habit",
                "current_strength": habit_data["habit_strength"],
                "target_strength": 0.7,
                "strategies": [
                    "Set daily reminders",
                    "Start with smaller goals",
                    "Track progress daily",
                    "Reward consistent behavior"
                ]
            })
        
        # Analyze workflow patterns
        workflows = self.workflow_patterns.get(user_id, [])
        inefficient_workflows = [w for w in workflows if w["success_rate"] < 0.7 and w["count"] >= 3]
        
        for workflow in inefficient_workflows:
            suggestions.append({
                "type": "workflow_optimization",
                "priority": "high",
                "action": f"Optimize {workflow['sequence'][0]} → {workflow['sequence'][1]} workflow",
                "current_success_rate": workflow["success_rate"],
                "frequency": workflow["count"],
                "strategies": [
                    "Review the process steps",
                    "Identify common failure points",
                    "Consider alternative approaches",
                    "Practice the workflow"
                ]
            })
        
        return suggestions

    def _perform_trend_analysis(self, user_id):
        """Perform comprehensive trend analysis."""
        trends = {}
        
        # Activity volume trend
        activities = self.activity_data.get(user_id, [])
        if len(activities) >= 14:  # At least 2 weeks of data
            # Group by day
            daily_counts = defaultdict(int)
            for activity in activities:
                try:
                    day = datetime.fromisoformat(activity["timestamp"]).date().isoformat()
                    daily_counts[day] += 1
                except:
                    continue
            
            if len(daily_counts) >= 7:
                sorted_days = sorted(daily_counts.keys())
                recent_week = sorted_days[-7:]
                previous_week = sorted_days[-14:-7] if len(sorted_days) >= 14 else []
                
                if previous_week:
                    recent_avg = np.mean([daily_counts[day] for day in recent_week])
                    previous_avg = np.mean([daily_counts[day] for day in previous_week])
                    
                    change = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
                    
                    trends["activity_volume"] = {
                        "trend": "increasing" if change > 0.1 else "decreasing" if change < -0.1 else "stable",
                        "change_percentage": change,
                        "recent_avg": recent_avg,
                        "previous_avg": previous_avg
                    }
        
        # Productivity trend
        productivity_data = self.productivity_metrics.get(user_id, {})
        if len(productivity_data) >= 7:
            sorted_dates = sorted(productivity_data.keys())
            recent_productivity = [productivity_data[date]["efficiency_score"] for date in sorted_dates[-7:] if "efficiency_score" in productivity_data[date]]
            
            if len(recent_productivity) >= 5:
                # Linear regression for trend
                x = np.arange(len(recent_productivity))
                slope = np.polyfit(x, recent_productivity, 1)[0]
                
                trends["productivity"] = {
                    "trend": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable",
                    "slope": slope,
                    "recent_scores": recent_productivity
                }
        
        # Goal achievement trend
        goal_data = self.goal_tracking.get(user_id, {})
        improving_goals = []
        declining_goals = []
        
        for goal_type, data in goal_data.items():
            if data["activity_count"] >= 10:
                # Analyze recent vs older success rates (simplified)
                if data["success_rate"] > 0.7:
                    improving_goals.append(goal_type)
                elif data["success_rate"] < 0.4:
                    declining_goals.append(goal_type)
        
        trends["goal_achievement"] = {
            "improving_goals": improving_goals,
            "declining_goals": declining_goals,
            "overall_trend": "positive" if len(improving_goals) > len(declining_goals) else "negative" if len(declining_goals) > len(improving_goals) else "mixed"
        }
        
        return trends

class RecommendationEngine:
    """Generate personalized recommendations."""
    
    def get_recommendations(self, user_id, profile, behavior_data, category=None):
        """Generate recommendations based on user data."""
        recommendations = []
        
        # Feature-based recommendations
        unused_features = self._find_unused_features(behavior_data)
        if unused_features:
            recommendations.extend([
                {
                    "type": "feature",
                    "title": f"Try {feature}",
                    "description": f"Based on your usage patterns, you might find {feature} useful",
                    "priority": "medium"
                }
                for feature in unused_features[:3]
            ])
            
        # Efficiency recommendations
        efficiency_tips = self._get_efficiency_recommendations(profile, behavior_data)
        recommendations.extend(efficiency_tips)
        
        # Personalization recommendations
        personalization_tips = self._get_personalization_recommendations(profile)
        recommendations.extend(personalization_tips)
        
        return recommendations[:10]  # Limit to top 10
        
    def _find_unused_features(self, behavior_data):
        """Find features user hasn't tried."""
        all_features = [
            "voice_commands", "gesture_control", "smart_home_integration",
            "automation_rules", "face_recognition", "ar_interface"
        ]
        
        used_features = set()
        for session in behavior_data.get("interaction_history", []):
            context = session.get("context", {})
            if "feature" in context:
                used_features.add(context["feature"])
                
        return [f for f in all_features if f not in used_features]
        
    def _get_efficiency_recommendations(self, profile, behavior_data):
        """Get recommendations to improve efficiency."""
        recommendations = []
        
        # Check for repeated manual tasks
        interaction_history = behavior_data.get("interaction_history", [])
        if len(interaction_history) > 10:
            # Look for automation opportunities
            recommendations.append({
                "type": "automation",
                "title": "Create Automation Rules",
                "description": "You perform similar tasks regularly. Consider setting up automation rules.",
                "priority": "high"
            })
            
        return recommendations
        
    def _get_personalization_recommendations(self, profile):
        """Get recommendations for better personalization."""
        recommendations = []
        
        # Check profile completeness
        if not profile.get("interests"):
            recommendations.append({
                "type": "profile",
                "title": "Add Your Interests",
                "description": "Adding interests helps me provide more relevant responses",
                "priority": "low"
            })
            
        return recommendations
