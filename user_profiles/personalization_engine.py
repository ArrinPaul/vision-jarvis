"""
Advanced AI-Powered Personalization Engine for JARVIS User Profiles.
Provides intelligent personalization with adaptive learning, contextual recommendations,
multi-modal fusion, behavioral analysis, and predictive optimization.
"""

import json
import os
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

# Optional imports for enhanced ML features
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA, NMF
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = DBSCAN = StandardScaler = MinMaxScaler = None
    PCA = NMF = RandomForestRegressor = IsolationForest = None
    MLPRegressor = cosine_similarity = None

try:
    import scipy.stats as stats
    from scipy.spatial.distance import euclidean, cosine
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = euclidean = cosine = None

@dataclass
class PersonalizationProfile:
    """Comprehensive user personalization profile."""
    user_id: str
    created_at: str
    last_updated: str
    
    # Core characteristics
    personality_traits: Dict[str, float]
    behavioral_patterns: Dict[str, Any]
    preference_evolution: List[Dict[str, Any]]
    context_adaptations: Dict[str, Dict[str, Any]]
    
    # Learning metrics
    adaptation_speed: float
    consistency_score: float
    exploration_tendency: float
    satisfaction_history: List[float]
    
    # Multi-modal preferences
    visual_preferences: Dict[str, Any]
    auditory_preferences: Dict[str, Any]
    interaction_preferences: Dict[str, Any]
    cognitive_preferences: Dict[str, Any]

@dataclass
class PersonalizationRecommendation:
    """AI-generated personalization recommendation."""
    recommendation_id: str
    user_id: str
    category: str
    type: str
    priority: int
    confidence: float
    
    # Recommendation details
    title: str
    description: str
    expected_benefit: str
    implementation_complexity: str
    
    # Target changes
    preference_changes: Dict[str, Any]
    behavioral_adaptations: Dict[str, Any]
    interface_modifications: Dict[str, Any]
    
    # Supporting evidence
    reasoning: str
    supporting_data: Dict[str, Any]
    success_probability: float
    estimated_impact: Dict[str, float]

@dataclass
class ContextualState:
    """Current contextual state for personalization."""
    timestamp: str
    user_id: str
    
    # Environmental context
    time_of_day: str
    day_of_week: str
    location: Optional[str]
    environment_type: str
    ambient_conditions: Dict[str, Any]
    
    # Activity context
    current_activity: str
    activity_duration: float
    recent_activities: List[str]
    activity_intensity: str
    
    # Emotional context
    detected_mood: Optional[str]
    stress_level: float
    engagement_level: float
    satisfaction_level: float
    
    # Technical context
    device_type: str
    connection_quality: str
    performance_metrics: Dict[str, float]
    available_modalities: List[str]


class PersonalizationEngine:
    """
    Advanced AI-powered personalization engine that creates highly tailored user experiences
    through multi-modal learning, behavioral analysis, and predictive optimization.
    """
    
    def __init__(self, personalization_dir="personalization_data"):
        self.personalization_dir = personalization_dir
        self.user_profiles = {}
        self.active_contexts = {}
        self.recommendation_cache = defaultdict(list)
        
        # Core AI components
        self.behavioral_analyzer = BehavioralPersonalizationAnalyzer()
        self.preference_predictor = PreferencePredictor()
        self.context_fusion_engine = ContextFusionEngine()
        self.adaptive_optimizer = AdaptiveOptimizer()
        self.multi_modal_learner = MultiModalLearner()
        
        # Recommendation engines
        self.content_recommender = ContentRecommendationEngine()
        self.interface_recommender = InterfaceRecommendationEngine()
        self.workflow_recommender = WorkflowRecommendationEngine()
        self.feature_recommender = FeatureRecommendationEngine()
        
        # Real-time processing
        self.interaction_stream = deque(maxlen=1000)
        self.personalization_events = deque(maxlen=500)
        self.learning_queue = deque(maxlen=200)
        
        # ML models
        self.personality_models = {}
        self.preference_models = {}
        self.satisfaction_predictors = {}
        self.context_classifiers = {}
        
        # Performance tracking
        self.recommendation_metrics = defaultdict(dict)
        self.personalization_effectiveness = defaultdict(float)
        self.user_satisfaction_trends = defaultdict(list)
        
        # Background processing
        self.processing_thread = threading.Thread(target=self._background_learning, daemon=True)
        self.learning_enabled = True
        
        # Initialize system
        self._initialize_personalization_system()
        
        logging.info("Advanced Personalization Engine initialized")
    
    def create_user_profile(self, user_id: str, initial_data: Dict[str, Any] = None) -> PersonalizationProfile:
        """Create a comprehensive personalization profile for a new user."""
        if initial_data is None:
            initial_data = {}
        
        # Initialize personality traits using Big Five model
        personality_traits = {
            "openness": initial_data.get("openness", 0.5),
            "conscientiousness": initial_data.get("conscientiousness", 0.5),
            "extraversion": initial_data.get("extraversion", 0.5),
            "agreeableness": initial_data.get("agreeableness", 0.5),
            "neuroticism": initial_data.get("neuroticism", 0.5)
        }
        
        # Initialize behavioral patterns
        behavioral_patterns = {
            "interaction_frequency": initial_data.get("interaction_frequency", "moderate"),
            "feature_exploration": initial_data.get("feature_exploration", "balanced"),
            "feedback_responsiveness": initial_data.get("feedback_responsiveness", "average"),
            "customization_tendency": initial_data.get("customization_tendency", "moderate"),
            "help_seeking_behavior": initial_data.get("help_seeking_behavior", "when_needed")
        }
        
        # Initialize multi-modal preferences
        visual_preferences = {
            "color_preference": initial_data.get("color_preference", "adaptive"),
            "contrast_sensitivity": initial_data.get("contrast_sensitivity", 0.5),
            "motion_sensitivity": initial_data.get("motion_sensitivity", 0.5),
            "visual_density": initial_data.get("visual_density", "balanced"),
            "icon_vs_text": initial_data.get("icon_vs_text", 0.5)
        }
        
        auditory_preferences = {
            "volume_preference": initial_data.get("volume_preference", 0.7),
            "speech_rate": initial_data.get("speech_rate", 1.0),
            "voice_type": initial_data.get("voice_type", "neutral"),
            "background_sounds": initial_data.get("background_sounds", False),
            "audio_feedback": initial_data.get("audio_feedback", True)
        }
        
        interaction_preferences = {
            "input_modality": initial_data.get("input_modality", "mixed"),
            "confirmation_level": initial_data.get("confirmation_level", "moderate"),
            "automation_tolerance": initial_data.get("automation_tolerance", 0.6),
            "error_handling": initial_data.get("error_handling", "guided"),
            "pace_preference": initial_data.get("pace_preference", "user_controlled")
        }
        
        cognitive_preferences = {
            "information_density": initial_data.get("information_density", "moderate"),
            "explanation_depth": initial_data.get("explanation_depth", "contextual"),
            "learning_style": initial_data.get("learning_style", "mixed"),
            "decision_support": initial_data.get("decision_support", "moderate"),
            "cognitive_load": initial_data.get("cognitive_load", "adaptive")
        }
        
        # Create profile
        profile = PersonalizationProfile(
            user_id=user_id,
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            personality_traits=personality_traits,
            behavioral_patterns=behavioral_patterns,
            preference_evolution=[],
            context_adaptations={},
            adaptation_speed=0.5,
            consistency_score=0.5,
            exploration_tendency=0.5,
            satisfaction_history=[],
            visual_preferences=visual_preferences,
            auditory_preferences=auditory_preferences,
            interaction_preferences=interaction_preferences,
            cognitive_preferences=cognitive_preferences
        )
        
        # Store profile
        self.user_profiles[user_id] = profile
        self._save_user_profile(user_id)
        
        # Initialize ML models for this user
        self._initialize_user_models(user_id)
        
        logging.info(f"Created personalization profile for user {user_id}")
        return profile
    
    def update_context(self, user_id: str, context: ContextualState):
        """Update current contextual state for personalization."""
        self.active_contexts[user_id] = context
        
        # Trigger contextual adaptations
        if user_id in self.user_profiles:
            adaptations = self.context_fusion_engine.generate_adaptations(
                self.user_profiles[user_id], context
            )
            
            if adaptations:
                self._apply_contextual_adaptations(user_id, adaptations)
        
        logging.debug(f"Updated context for user {user_id}: {context.current_activity}")
    
    def get_personalized_recommendations(self, user_id: str, category: str = "all", 
                                       max_recommendations: int = 10) -> List[PersonalizationRecommendation]:
        """Get AI-powered personalized recommendations for a user."""
        if user_id not in self.user_profiles:
            logging.warning(f"No profile found for user {user_id}")
            return []
        
        profile = self.user_profiles[user_id]
        context = self.active_contexts.get(user_id)
        
        all_recommendations = []
        
        # Generate different types of recommendations
        if category in ["all", "interface"]:
            interface_recs = self.interface_recommender.generate_recommendations(profile, context)
            all_recommendations.extend(interface_recs)
        
        if category in ["all", "content"]:
            content_recs = self.content_recommender.generate_recommendations(profile, context)
            all_recommendations.extend(content_recs)
        
        if category in ["all", "workflow"]:
            workflow_recs = self.workflow_recommender.generate_recommendations(profile, context)
            all_recommendations.extend(workflow_recs)
        
        if category in ["all", "features"]:
            feature_recs = self.feature_recommender.generate_recommendations(profile, context)
            all_recommendations.extend(feature_recs)
        
        # Score and rank recommendations
        scored_recommendations = self._score_recommendations(all_recommendations, profile, context)
        
        # Remove duplicates and limit results
        unique_recommendations = self._deduplicate_recommendations(scored_recommendations)
        top_recommendations = sorted(unique_recommendations, key=lambda x: x.confidence, reverse=True)[:max_recommendations]
        
        # Cache recommendations
        self.recommendation_cache[user_id] = top_recommendations
        
        logging.info(f"Generated {len(top_recommendations)} personalized recommendations for user {user_id}")
        return top_recommendations
    
    def learn_from_interaction(self, user_id: str, interaction_data: Dict[str, Any]):
        """Learn from user interaction to improve personalization."""
        if user_id not in self.user_profiles:
            return
        
        # Add to learning queue
        learning_event = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "interaction_data": interaction_data,
            "context": self.active_contexts.get(user_id)
        }
        
        self.learning_queue.append(learning_event)
        
        # Immediate learning for high-impact interactions
        if interaction_data.get("impact", "low") == "high":
            self._process_immediate_learning(learning_event)
        
        logging.debug(f"Queued learning event for user {user_id}")
    
    def predict_user_satisfaction(self, user_id: str, proposed_changes: Dict[str, Any]) -> float:
        """Predict user satisfaction with proposed changes using ML."""
        if user_id not in self.user_profiles or user_id not in self.satisfaction_predictors:
            return 0.5  # Neutral prediction
        
        try:
            profile = self.user_profiles[user_id]
            predictor = self.satisfaction_predictors[user_id]
            
            # Extract features for prediction
            features = self._extract_satisfaction_features(profile, proposed_changes)
            
            # Predict satisfaction
            satisfaction_score = predictor.predict([features])[0] if SKLEARN_AVAILABLE else 0.5
            
            return max(0.0, min(1.0, satisfaction_score))
            
        except Exception as e:
            logging.error(f"Satisfaction prediction error for user {user_id}: {e}")
            return 0.5
    
    def optimize_user_experience(self, user_id: str) -> Dict[str, Any]:
        """Perform comprehensive user experience optimization."""
        if user_id not in self.user_profiles:
            return {"error": "User profile not found"}
        
        profile = self.user_profiles[user_id]
        context = self.active_contexts.get(user_id)
        
        # Multi-dimensional optimization
        optimization_results = {
            "user_id": user_id,
            "optimization_timestamp": datetime.now().isoformat(),
            "optimizations_applied": [],
            "performance_improvements": {},
            "satisfaction_predictions": {}
        }
        
        # Interface optimization
        interface_optimizations = self.adaptive_optimizer.optimize_interface(profile, context)
        optimization_results["optimizations_applied"].extend(interface_optimizations)
        
        # Workflow optimization
        workflow_optimizations = self.adaptive_optimizer.optimize_workflows(profile, context)
        optimization_results["optimizations_applied"].extend(workflow_optimizations)
        
        # Content personalization optimization
        content_optimizations = self.adaptive_optimizer.optimize_content_delivery(profile, context)
        optimization_results["optimizations_applied"].extend(content_optimizations)
        
        # Predictive optimization
        predictive_optimizations = self.adaptive_optimizer.optimize_predictively(profile, context)
        optimization_results["optimizations_applied"].extend(predictive_optimizations)
        
        # Calculate expected improvements
        for optimization in optimization_results["optimizations_applied"]:
            category = optimization.get("category", "general")
            expected_improvement = optimization.get("expected_improvement", 0.1)
            
            if category not in optimization_results["performance_improvements"]:
                optimization_results["performance_improvements"][category] = 0.0
            
            optimization_results["performance_improvements"][category] += expected_improvement
        
        # Update user profile with optimizations
        self._apply_optimizations(user_id, optimization_results["optimizations_applied"])
        
        logging.info(f"Performed comprehensive optimization for user {user_id}: {len(optimization_results['optimizations_applied'])} optimizations applied")
        return optimization_results
    
    def analyze_personalization_effectiveness(self, user_id: str, time_window_days: int = 30) -> Dict[str, Any]:
        """Analyze the effectiveness of personalization for a user."""
        if user_id not in self.user_profiles:
            return {"error": "User profile not found"}
        
        profile = self.user_profiles[user_id]
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        
        analysis = {
            "user_id": user_id,
            "analysis_period": f"{time_window_days} days",
            "analysis_timestamp": datetime.now().isoformat(),
            "effectiveness_metrics": {},
            "satisfaction_analysis": {},
            "adaptation_analysis": {},
            "recommendation_performance": {},
            "areas_for_improvement": []
        }
        
        # Analyze satisfaction trends
        recent_satisfaction = [s for s in profile.satisfaction_history if len(profile.satisfaction_history) > 0]
        if recent_satisfaction:
            analysis["satisfaction_analysis"] = {
                "average_satisfaction": np.mean(recent_satisfaction) if np else 0.5,
                "satisfaction_trend": self._calculate_satisfaction_trend(recent_satisfaction),
                "satisfaction_volatility": np.std(recent_satisfaction) if np else 0.0,
                "peak_satisfaction": max(recent_satisfaction),
                "lowest_satisfaction": min(recent_satisfaction)
            }
        
        # Analyze adaptation effectiveness
        analysis["adaptation_analysis"] = {
            "adaptation_speed": profile.adaptation_speed,
            "consistency_score": profile.consistency_score,
            "exploration_tendency": profile.exploration_tendency,
            "behavioral_stability": self._calculate_behavioral_stability(profile)
        }
        
        # Analyze recommendation performance
        if user_id in self.recommendation_metrics:
            metrics = self.recommendation_metrics[user_id]
            analysis["recommendation_performance"] = {
                "acceptance_rate": metrics.get("acceptance_rate", 0.0),
                "implementation_success": metrics.get("implementation_success", 0.0),
                "user_benefit_score": metrics.get("user_benefit_score", 0.0),
                "recommendation_accuracy": metrics.get("recommendation_accuracy", 0.0)
            }
        
        # Calculate overall effectiveness
        effectiveness_components = []
        if "average_satisfaction" in analysis["satisfaction_analysis"]:
            effectiveness_components.append(analysis["satisfaction_analysis"]["average_satisfaction"])
        if "acceptance_rate" in analysis["recommendation_performance"]:
            effectiveness_components.append(analysis["recommendation_performance"]["acceptance_rate"])
        
        analysis["effectiveness_metrics"]["overall_effectiveness"] = np.mean(effectiveness_components) if effectiveness_components and np else 0.5
        
        # Identify improvement areas
        analysis["areas_for_improvement"] = self._identify_improvement_areas(analysis)
        
        return analysis
    
    def export_personalization_data(self, user_id: str) -> Dict[str, Any]:
        """Export comprehensive personalization data for a user."""
        if user_id not in self.user_profiles:
            return {"error": "User profile not found"}
        
        export_data = {
            "user_id": user_id,
            "export_timestamp": datetime.now().isoformat(),
            "profile_data": asdict(self.user_profiles[user_id]),
            "recent_recommendations": self.recommendation_cache.get(user_id, []),
            "effectiveness_metrics": self.personalization_effectiveness.get(user_id, 0.0),
            "model_summaries": self._get_model_summaries(user_id),
            "export_version": "2.0"
        }
        
        return export_data
    
    def import_personalization_data(self, import_data: Dict[str, Any]) -> bool:
        """Import personalization data for a user."""
        try:
            user_id = import_data.get("user_id")
            if not user_id:
                return False
            
            # Import profile data
            if "profile_data" in import_data:
                profile_dict = import_data["profile_data"]
                profile = PersonalizationProfile(**profile_dict)
                self.user_profiles[user_id] = profile
            
            # Import recommendations
            if "recent_recommendations" in import_data:
                self.recommendation_cache[user_id] = import_data["recent_recommendations"]
            
            # Import effectiveness metrics
            if "effectiveness_metrics" in import_data:
                self.personalization_effectiveness[user_id] = import_data["effectiveness_metrics"]
            
            # Reinitialize models
            self._initialize_user_models(user_id)
            
            # Save imported data
            self._save_user_profile(user_id)
            
            logging.info(f"Successfully imported personalization data for user {user_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error importing personalization data: {e}")
            return False
    
    def _initialize_personalization_system(self):
        """Initialize the personalization system."""
        # Create directories
        os.makedirs(self.personalization_dir, exist_ok=True)
        os.makedirs(f"{self.personalization_dir}/profiles", exist_ok=True)
        os.makedirs(f"{self.personalization_dir}/models", exist_ok=True)
        os.makedirs(f"{self.personalization_dir}/analytics", exist_ok=True)
        
        # Load existing profiles
        self._load_all_profiles()
        
        # Initialize ML models
        self._initialize_global_models()
        
        # Start background learning
        self.processing_thread.start()
    
    def _initialize_user_models(self, user_id: str):
        """Initialize ML models for a specific user."""
        if not SKLEARN_AVAILABLE:
            return
        
        try:
            # Personality prediction model
            self.personality_models[user_id] = MLPRegressor(
                hidden_layer_sizes=(50, 25),
                max_iter=1000,
                random_state=42
            )
            
            # Preference evolution model
            self.preference_models[user_id] = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
            
            # Satisfaction predictor
            self.satisfaction_predictors[user_id] = RandomForestRegressor(
                n_estimators=50,
                random_state=42
            )
            
            # Context classifier
            self.context_classifiers[user_id] = KMeans(
                n_clusters=5,
                random_state=42
            )
            
            logging.debug(f"Initialized ML models for user {user_id}")
            
        except Exception as e:
            logging.error(f"Error initializing user models for {user_id}: {e}")
    
    def _apply_contextual_adaptations(self, user_id: str, adaptations: List[Dict[str, Any]]):
        """Apply contextual adaptations to user profile."""
        if user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[user_id]
        current_context = self.active_contexts.get(user_id)
        
        if not current_context:
            return
        
        # Store adaptations in profile
        context_key = f"{current_context.current_activity}_{current_context.time_of_day}"
        
        if context_key not in profile.context_adaptations:
            profile.context_adaptations[context_key] = {}
        
        for adaptation in adaptations:
            adaptation_type = adaptation.get("type", "general")
            profile.context_adaptations[context_key][adaptation_type] = adaptation
        
        # Update profile timestamp
        profile.last_updated = datetime.now().isoformat()
        
        # Save profile
        self._save_user_profile(user_id)
        
        logging.debug(f"Applied {len(adaptations)} contextual adaptations for user {user_id}")
    
    def _score_recommendations(self, recommendations: List[PersonalizationRecommendation], 
                             profile: PersonalizationProfile, context: Optional[ContextualState]) -> List[PersonalizationRecommendation]:
        """Score recommendations using multiple factors."""
        scoring_weights = {
            "relevance": 0.25,
            "feasibility": 0.20,
            "impact": 0.20,
            "user_preference_alignment": 0.15,
            "context_appropriateness": 0.10,
            "novelty": 0.10
        }
        
        for rec in recommendations:
            # Calculate individual scores
            relevance_score = self._calculate_relevance_score(rec, profile)
            feasibility_score = self._calculate_feasibility_score(rec)
            impact_score = self._calculate_impact_score(rec, profile)
            alignment_score = self._calculate_preference_alignment_score(rec, profile)
            context_score = self._calculate_context_appropriateness_score(rec, context) if context else 0.5
            novelty_score = self._calculate_novelty_score(rec, profile)
            
            # Weighted composite score
            composite_score = (
                scoring_weights["relevance"] * relevance_score +
                scoring_weights["feasibility"] * feasibility_score +
                scoring_weights["impact"] * impact_score +
                scoring_weights["user_preference_alignment"] * alignment_score +
                scoring_weights["context_appropriateness"] * context_score +
                scoring_weights["novelty"] * novelty_score
            )
            
            # Update recommendation confidence
            rec.confidence = min(1.0, max(0.0, composite_score))
        
        return recommendations
    
    def _deduplicate_recommendations(self, recommendations: List[PersonalizationRecommendation]) -> List[PersonalizationRecommendation]:
        """Remove duplicate recommendations, keeping the highest confidence version."""
        seen_recommendations = {}
        
        for rec in recommendations:
            key = f"{rec.category}_{rec.type}_{hash(str(rec.preference_changes))}"
            
            if key not in seen_recommendations or rec.confidence > seen_recommendations[key].confidence:
                seen_recommendations[key] = rec
        
        return list(seen_recommendations.values())
    
    def _background_learning(self):
        """Background thread for continuous learning and optimization."""
        while self.learning_enabled:
            try:
                # Process learning queue
                if self.learning_queue:
                    batch_size = min(10, len(self.learning_queue))
                    learning_batch = [self.learning_queue.popleft() for _ in range(batch_size)]
                    self._process_learning_batch(learning_batch)
                
                # Update ML models
                self._update_ml_models()
                
                # Optimize user experiences
                self._background_optimization()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(60)  # Process every minute
                
            except Exception as e:
                logging.error(f"Background learning error: {e}")
                time.sleep(60)
    
    def _process_learning_batch(self, learning_batch: List[Dict[str, Any]]):
        """Process a batch of learning events."""
        # Group by user
        user_events = defaultdict(list)
        for event in learning_batch:
            user_events[event["user_id"]].append(event)
        
        # Update each user's learning
        for user_id, events in user_events.items():
            if user_id in self.user_profiles:
                self._update_user_learning(user_id, events)
    
    def _update_user_learning(self, user_id: str, events: List[Dict[str, Any]]):
        """Update learning for a specific user based on events."""
        profile = self.user_profiles[user_id]
        
        for event in events:
            interaction_data = event["interaction_data"]
            
            # Update satisfaction history
            if "satisfaction" in interaction_data:
                profile.satisfaction_history.append(interaction_data["satisfaction"])
                
                # Keep only last 100 satisfaction scores
                if len(profile.satisfaction_history) > 100:
                    profile.satisfaction_history = profile.satisfaction_history[-100:]
            
            # Update behavioral patterns
            if "behavior_changes" in interaction_data:
                self._update_behavioral_patterns(profile, interaction_data["behavior_changes"])
            
            # Update preferences
            if "preference_feedback" in interaction_data:
                self._update_preference_evolution(profile, interaction_data["preference_feedback"])
        
        # Update learning metrics
        self._update_learning_metrics(profile, events)
        
        # Save updated profile
        profile.last_updated = datetime.now().isoformat()
        self._save_user_profile(user_id)
    
    def _save_user_profile(self, user_id: str):
        """Save user profile to storage."""
        try:
            profile_path = f"{self.personalization_dir}/profiles/{user_id}_profile.json"
            
            with open(profile_path, 'w') as f:
                json.dump(asdict(self.user_profiles[user_id]), f, indent=2, default=str)
            
            logging.debug(f"Saved profile for user {user_id}")
            
        except Exception as e:
            logging.error(f"Error saving profile for user {user_id}: {e}")
    
    def _load_all_profiles(self):
        """Load all user profiles from storage."""
        try:
            profiles_dir = f"{self.personalization_dir}/profiles"
            
            if os.path.exists(profiles_dir):
                for filename in os.listdir(profiles_dir):
                    if filename.endswith("_profile.json"):
                        user_id = filename.replace("_profile.json", "")
                        
                        with open(os.path.join(profiles_dir, filename), 'r') as f:
                            profile_data = json.load(f)
                            profile = PersonalizationProfile(**profile_data)
                            self.user_profiles[user_id] = profile
                            
                            # Initialize models for loaded users
                            self._initialize_user_models(user_id)
            
            logging.info(f"Loaded {len(self.user_profiles)} personalization profiles")
            
        except Exception as e:
            logging.error(f"Error loading profiles: {e}")
    
    # Helper methods for scoring
    def _calculate_relevance_score(self, rec: PersonalizationRecommendation, profile: PersonalizationProfile) -> float:
        """Calculate relevance score for a recommendation."""
        # Base relevance on user's behavioral patterns and preferences
        relevance = 0.5
        
        # Check alignment with personality traits
        if rec.category == "interface" and profile.personality_traits.get("openness", 0.5) > 0.7:
            relevance += 0.2
        
        # Check alignment with behavioral patterns
        if rec.type in profile.behavioral_patterns:
            relevance += 0.3
        
        return min(1.0, relevance)
    
    def _calculate_feasibility_score(self, rec: PersonalizationRecommendation) -> float:
        """Calculate feasibility score for a recommendation."""
        complexity_scores = {
            "low": 1.0,
            "medium": 0.7,
            "high": 0.4,
            "very_high": 0.2
        }
        
        return complexity_scores.get(rec.implementation_complexity, 0.5)
    
    def _calculate_impact_score(self, rec: PersonalizationRecommendation, profile: PersonalizationProfile) -> float:
        """Calculate expected impact score for a recommendation."""
        # Use estimated impact from recommendation
        estimated_impacts = rec.estimated_impact
        
        if estimated_impacts:
            # Weight different impact categories
            weights = {"satisfaction": 0.4, "efficiency": 0.3, "engagement": 0.3}
            
            weighted_impact = sum(
                weights.get(category, 0.1) * impact
                for category, impact in estimated_impacts.items()
            )
            
            return min(1.0, weighted_impact)
        
        return 0.5
    
    def _calculate_preference_alignment_score(self, rec: PersonalizationRecommendation, profile: PersonalizationProfile) -> float:
        """Calculate how well recommendation aligns with user preferences."""
        alignment = 0.5
        
        # Check multi-modal preference alignment
        if rec.category == "visual" and "visual_preferences" in asdict(profile):
            visual_prefs = profile.visual_preferences
            
            for change_key, change_value in rec.preference_changes.items():
                if change_key in visual_prefs:
                    # Calculate preference similarity
                    if isinstance(change_value, (int, float)) and isinstance(visual_prefs[change_key], (int, float)):
                        diff = abs(change_value - visual_prefs[change_key])
                        alignment += (1 - diff) * 0.2
        
        return min(1.0, alignment)


class BehavioralPersonalizationAnalyzer:
    """Analyzes user behavior patterns for personalization insights."""
    
    def __init__(self):
        self.behavior_models = {}
        self.pattern_detectors = {}
        
    def analyze_behavior_patterns(self, user_id: str, interaction_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze comprehensive behavior patterns for personalization."""
        if not interaction_history:
            return {"patterns": [], "insights": [], "recommendations": []}
        
        analysis = {
            "temporal_patterns": self._analyze_temporal_patterns(interaction_history),
            "interaction_patterns": self._analyze_interaction_patterns(interaction_history),
            "preference_patterns": self._analyze_preference_patterns(interaction_history),
            "learning_patterns": self._analyze_learning_patterns(interaction_history),
            "adaptation_patterns": self._analyze_adaptation_patterns(interaction_history)
        }
        
        # Generate behavioral insights
        insights = self._generate_behavioral_insights(analysis)
        
        # Generate personalization recommendations
        recommendations = self._generate_behavioral_recommendations(analysis)
        
        return {
            "analysis": analysis,
            "insights": insights,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _analyze_temporal_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal behavior patterns."""
        patterns = {
            "peak_usage_hours": [],
            "usage_frequency": {},
            "session_patterns": {},
            "weekly_patterns": {}
        }
        
        # Group interactions by hour
        hourly_usage = defaultdict(int)
        daily_usage = defaultdict(int)
        
        for interaction in history:
            if "timestamp" in interaction:
                try:
                    dt = datetime.fromisoformat(interaction["timestamp"])
                    hourly_usage[dt.hour] += 1
                    daily_usage[dt.strftime("%A")] += 1
                except:
                    continue
        
        # Find peak hours
        if hourly_usage:
            peak_hours = sorted(hourly_usage.items(), key=lambda x: x[1], reverse=True)[:3]
            patterns["peak_usage_hours"] = [hour for hour, count in peak_hours]
        
        patterns["usage_frequency"] = dict(hourly_usage)
        patterns["weekly_patterns"] = dict(daily_usage)
        
        return patterns


class PreferencePredictor:
    """Predicts user preference evolution using machine learning."""
    
    def __init__(self):
        self.prediction_models = {}
        self.preference_histories = defaultdict(list)
        
    def predict_preference_evolution(self, user_id: str, current_preferences: Dict[str, Any], 
                                   context: Optional[ContextualState] = None) -> Dict[str, Any]:
        """Predict how user preferences might evolve."""
        predictions = {
            "short_term": {},  # Next week
            "medium_term": {},  # Next month  
            "long_term": {},   # Next 3 months
            "confidence_scores": {},
            "influencing_factors": []
        }
        
        if not SKLEARN_AVAILABLE:
            return predictions
        
        try:
            # Get preference history
            history = self.preference_histories[user_id]
            
            if len(history) < 5:  # Need minimum history
                return predictions
            
            # Extract features for prediction
            features = self._extract_prediction_features(history, current_preferences, context)
            
            # Predict for different time horizons
            for time_horizon in ["short_term", "medium_term", "long_term"]:
                horizon_predictions = self._predict_for_horizon(user_id, features, time_horizon)
                predictions[time_horizon] = horizon_predictions
                
                # Calculate confidence
                predictions["confidence_scores"][time_horizon] = self._calculate_prediction_confidence(
                    horizon_predictions, history
                )
            
            # Identify influencing factors
            predictions["influencing_factors"] = self._identify_influencing_factors(features, history)
            
        except Exception as e:
            logging.error(f"Preference prediction error for user {user_id}: {e}")
        
        return predictions


class ContextFusionEngine:
    """Fuses multiple contextual signals for comprehensive understanding."""
    
    def __init__(self):
        self.fusion_models = {}
        self.context_weights = {
            "temporal": 0.25,
            "environmental": 0.20,
            "activity": 0.25,
            "emotional": 0.15,
            "technical": 0.15
        }
    
    def generate_adaptations(self, profile: PersonalizationProfile, 
                           context: ContextualState) -> List[Dict[str, Any]]:
        """Generate contextual adaptations based on fused context understanding."""
        adaptations = []
        
        # Temporal adaptations
        temporal_adaptations = self._generate_temporal_adaptations(profile, context)
        adaptations.extend(temporal_adaptations)
        
        # Activity-based adaptations
        activity_adaptations = self._generate_activity_adaptations(profile, context)
        adaptations.extend(activity_adaptations)
        
        # Emotional state adaptations
        emotional_adaptations = self._generate_emotional_adaptations(profile, context)
        adaptations.extend(emotional_adaptations)
        
        # Environmental adaptations
        environmental_adaptations = self._generate_environmental_adaptations(profile, context)
        adaptations.extend(environmental_adaptations)
        
        return adaptations
    
    def _generate_temporal_adaptations(self, profile: PersonalizationProfile, context: ContextualState) -> List[Dict[str, Any]]:
        """Generate adaptations based on temporal context."""
        adaptations = []
        
        # Time of day adaptations
        if context.time_of_day == "evening":
            adaptations.append({
                "type": "visual_comfort",
                "changes": {"theme": "dark", "brightness": 0.7},
                "reason": "Evening hours - reduce eye strain",
                "expected_improvement": 0.2
            })
        
        # Day of week adaptations
        if context.day_of_week in ["Saturday", "Sunday"]:
            adaptations.append({
                "type": "interaction_pace",
                "changes": {"pace": "relaxed", "automation": "increased"},
                "reason": "Weekend - more relaxed interaction preferred",
                "expected_improvement": 0.15
            })
        
        return adaptations


class AdaptiveOptimizer:
    """Performs adaptive optimization across multiple dimensions."""
    
    def __init__(self):
        self.optimization_history = defaultdict(list)
        self.performance_baselines = {}
        
    def optimize_interface(self, profile: PersonalizationProfile, context: Optional[ContextualState]) -> List[Dict[str, Any]]:
        """Optimize interface based on user profile and context."""
        optimizations = []
        
        # Visual optimizations
        if profile.visual_preferences.get("visual_density") == "minimal":
            optimizations.append({
                "category": "interface",
                "type": "visual_density",
                "changes": {"layout": "spacious", "elements": "minimal"},
                "expected_improvement": 0.2,
                "reasoning": "User prefers minimal visual density"
            })
        
        # Interaction optimizations
        if profile.interaction_preferences.get("automation_tolerance", 0.5) > 0.7:
            optimizations.append({
                "category": "interface", 
                "type": "automation",
                "changes": {"auto_suggestions": True, "predictive_actions": True},
                "expected_improvement": 0.25,
                "reasoning": "User is comfortable with high automation"
            })
        
        return optimizations
    
    def optimize_workflows(self, profile: PersonalizationProfile, context: Optional[ContextualState]) -> List[Dict[str, Any]]:
        """Optimize workflows based on behavioral patterns."""
        optimizations = []
        
        # Workflow efficiency optimizations
        if profile.behavioral_patterns.get("feature_exploration") == "conservative":
            optimizations.append({
                "category": "workflow",
                "type": "guided_workflows",
                "changes": {"guidance_level": "high", "shortcuts_prominent": True},
                "expected_improvement": 0.18,
                "reasoning": "Conservative users benefit from guided workflows"
            })
        
        return optimizations


class MultiModalLearner:
    """Learns across multiple interaction modalities."""
    
    def __init__(self):
        self.modality_models = {}
        self.cross_modal_patterns = {}
        
    def learn_cross_modal_patterns(self, user_id: str, multi_modal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn patterns across different interaction modalities."""
        patterns = {
            "modality_preferences": {},
            "cross_modal_correlations": {},
            "adaptation_insights": {},
            "optimization_opportunities": []
        }
        
        # Analyze modality usage patterns
        if "modality_usage" in multi_modal_data:
            usage_data = multi_modal_data["modality_usage"]
            
            total_usage = sum(usage_data.values())
            patterns["modality_preferences"] = {
                modality: usage / total_usage
                for modality, usage in usage_data.items()
            }
        
        # Find cross-modal correlations
        patterns["cross_modal_correlations"] = self._find_cross_modal_correlations(multi_modal_data)
        
        # Generate optimization opportunities
        patterns["optimization_opportunities"] = self._identify_multi_modal_optimizations(patterns)
        
        return patterns


# Recommendation engines
class ContentRecommendationEngine:
    """Recommends personalized content and information."""
    
    def generate_recommendations(self, profile: PersonalizationProfile, 
                               context: Optional[ContextualState]) -> List[PersonalizationRecommendation]:
        """Generate content-focused recommendations."""
        recommendations = []
        
        # Information density recommendations
        if profile.cognitive_preferences.get("information_density") == "high":
            rec = PersonalizationRecommendation(
                recommendation_id=f"content_density_{int(time.time())}",
                user_id=profile.user_id,
                category="content",
                type="information_density",
                priority=7,
                confidence=0.8,
                title="Increase Information Density",
                description="Show more detailed information in interface elements",
                expected_benefit="Access more information at once, reducing navigation",
                implementation_complexity="medium",
                preference_changes={"information_density": "detailed", "compact_view": True},
                behavioral_adaptations={},
                interface_modifications={"layout": "compact", "details": "expanded"},
                reasoning="User prefers high information density based on cognitive preferences",
                supporting_data={"cognitive_preferences": profile.cognitive_preferences},
                success_probability=0.75,
                estimated_impact={"efficiency": 0.2, "satisfaction": 0.15}
            )
            recommendations.append(rec)
        
        return recommendations


class InterfaceRecommendationEngine:
    """Recommends interface modifications and adaptations."""
    
    def generate_recommendations(self, profile: PersonalizationProfile,
                               context: Optional[ContextualState]) -> List[PersonalizationRecommendation]:
        """Generate interface-focused recommendations."""
        recommendations = []
        
        # Theme recommendations based on personality and context
        if profile.personality_traits.get("openness", 0.5) > 0.7:
            rec = PersonalizationRecommendation(
                recommendation_id=f"interface_theme_{int(time.time())}",
                user_id=profile.user_id,
                category="interface",
                type="theme_adaptation", 
                priority=6,
                confidence=0.7,
                title="Enable Dynamic Theme Adaptation",
                description="Use adaptive themes that change based on context and time",
                expected_benefit="Optimal visual comfort throughout the day",
                implementation_complexity="low",
                preference_changes={"theme": "adaptive_auto", "dynamic_adjustment": True},
                behavioral_adaptations={},
                interface_modifications={"theme_system": "adaptive", "context_aware": True},
                reasoning="High openness suggests receptiveness to adaptive features",
                supporting_data={"personality_traits": profile.personality_traits},
                success_probability=0.8,
                estimated_impact={"satisfaction": 0.18, "comfort": 0.25}
            )
            recommendations.append(rec)
        
        return recommendations


class WorkflowRecommendationEngine:
    """Recommends workflow optimizations and shortcuts."""
    
    def generate_recommendations(self, profile: PersonalizationProfile,
                               context: Optional[ContextualState]) -> List[PersonalizationRecommendation]:
        """Generate workflow-focused recommendations.""" 
        recommendations = []
        
        # Automation recommendations
        if profile.behavioral_patterns.get("customization_tendency") == "high":
            rec = PersonalizationRecommendation(
                recommendation_id=f"workflow_automation_{int(time.time())}",
                user_id=profile.user_id,
                category="workflow",
                type="automation_enhancement",
                priority=8,
                confidence=0.85,
                title="Enable Advanced Automation",
                description="Activate predictive actions and smart automation features",
                expected_benefit="Reduce repetitive tasks and streamline workflows",
                implementation_complexity="medium",
                preference_changes={"automation_level": "advanced", "predictive_actions": True},
                behavioral_adaptations={"task_prediction": True, "pattern_recognition": True},
                interface_modifications={"automation_controls": "visible", "smart_suggestions": True},
                reasoning="High customization tendency indicates comfort with advanced features",
                supporting_data={"behavioral_patterns": profile.behavioral_patterns},
                success_probability=0.82,
                estimated_impact={"efficiency": 0.3, "time_savings": 0.25}
            )
            recommendations.append(rec)
        
        return recommendations


class FeatureRecommendationEngine:
    """Recommends new features and capabilities."""
    
    def generate_recommendations(self, profile: PersonalizationProfile,
                               context: Optional[ContextualState]) -> List[PersonalizationRecommendation]:
        """Generate feature-focused recommendations."""
        recommendations = []
        
        # Feature exploration recommendations
        if profile.behavioral_patterns.get("feature_exploration") == "active":
            rec = PersonalizationRecommendation(
                recommendation_id=f"feature_exploration_{int(time.time())}",
                user_id=profile.user_id,
                category="features",
                type="new_feature_introduction",
                priority=5,
                confidence=0.65,
                title="Try Advanced Gesture Controls",
                description="Explore enhanced gesture recognition capabilities",
                expected_benefit="More intuitive and natural interaction methods",
                implementation_complexity="low",
                preference_changes={"gesture_enabled": True, "advanced_gestures": True},
                behavioral_adaptations={"gesture_learning": True},
                interface_modifications={"gesture_indicators": True, "tutorial_available": True},
                reasoning="Active feature exploration indicates willingness to try new capabilities",
                supporting_data={"behavioral_patterns": profile.behavioral_patterns},
                success_probability=0.7,
                estimated_impact={"engagement": 0.2, "novelty": 0.3}
            )
            recommendations.append(rec)
        
        return recommendations


# Helper functions
def _calculate_satisfaction_trend(satisfaction_history: List[float]) -> str:
    """Calculate satisfaction trend from history."""
    if len(satisfaction_history) < 3:
        return "insufficient_data"
    
    recent = satisfaction_history[-5:] if len(satisfaction_history) >= 5 else satisfaction_history
    older = satisfaction_history[:-5] if len(satisfaction_history) >= 10 else satisfaction_history[:-len(recent)//2]
    
    if not older:
        return "stable"
    
    recent_avg = np.mean(recent) if np else sum(recent) / len(recent)
    older_avg = np.mean(older) if np else sum(older) / len(older)
    
    diff = recent_avg - older_avg
    
    if diff > 0.1:
        return "improving"
    elif diff < -0.1:
        return "declining"
    else:
        return "stable"


def _calculate_behavioral_stability(profile: PersonalizationProfile) -> float:
    """Calculate behavioral stability score."""
    if len(profile.preference_evolution) < 2:
        return 0.5
    
    # Simple stability calculation based on preference change frequency
    recent_changes = len([e for e in profile.preference_evolution[-10:] if e.get("change_magnitude", 0) > 0.3])
    stability = max(0.0, 1.0 - (recent_changes / 10.0))
    
    return stability


if __name__ == "__main__":
    # Test the personalization engine
    print("Testing Advanced Personalization Engine...")
    
    # Initialize engine
    engine = PersonalizationEngine()
    
    # Create test user profile
    test_user_id = "test_user_personalization"
    initial_data = {
        "openness": 0.8,
        "conscientiousness": 0.7,
        "extraversion": 0.6,
        "feature_exploration": "active",
        "customization_tendency": "high"
    }
    
    profile = engine.create_user_profile(test_user_id, initial_data)
    print(f"Created profile for {profile.user_id}")
    
    # Create test context
    context = ContextualState(
        timestamp=datetime.now().isoformat(),
        user_id=test_user_id,
        time_of_day="evening",
        day_of_week="Friday",
        location=None,
        environment_type="home",
        ambient_conditions={},
        current_activity="work",
        activity_duration=120.0,
        recent_activities=["email", "research"],
        activity_intensity="moderate",
        detected_mood="focused",
        stress_level=0.3,
        engagement_level=0.8,
        satisfaction_level=0.7,
        device_type="desktop",
        connection_quality="excellent",
        performance_metrics={"response_time": 0.2},
        available_modalities=["voice", "gesture", "text"]
    )
    
    engine.update_context(test_user_id, context)
    
    # Get personalized recommendations
    recommendations = engine.get_personalized_recommendations(test_user_id, max_recommendations=5)
    print(f"Generated {len(recommendations)} personalized recommendations")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.title} (confidence: {rec.confidence:.2f})")
    
    # Test learning from interaction
    interaction_data = {
        "satisfaction": 0.8,
        "behavior_changes": {"automation_usage": "increased"},
        "preference_feedback": {"theme": "positive"},
        "impact": "medium"
    }
    
    engine.learn_from_interaction(test_user_id, interaction_data)
    
    # Test optimization
    optimization_results = engine.optimize_user_experience(test_user_id)
    print(f"Applied {len(optimization_results['optimizations_applied'])} optimizations")
    
    # Test effectiveness analysis
    effectiveness = engine.analyze_personalization_effectiveness(test_user_id)
    print(f"Personalization effectiveness: {effectiveness['effectiveness_metrics'].get('overall_effectiveness', 'N/A')}")
    
    print("Advanced Personalization Engine test completed!")