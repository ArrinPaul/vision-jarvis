"""
Advanced Interaction Analytics for JARVIS User Profiles.
Provides comprehensive analysis of gesture patterns, voice command usage, interaction efficiency,
machine learning insights, and behavioral analytics.
"""

import numpy as np
import json
import os
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

# Optional imports for enhanced ML features
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = DBSCAN = StandardScaler = PCA = IsolationForest = None

try:
    import scipy.stats as stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    find_peaks = None

@dataclass
class InteractionEvent:
    """Structured interaction event data."""
    timestamp: str
    event_type: str  # gesture, voice, touch, gaze, etc.
    event_data: Dict[str, Any]
    success: bool
    response_time: float
    confidence: float
    context: Dict[str, Any]
    user_satisfaction: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None

@dataclass
class GestureAnalysis:
    """Gesture pattern analysis results."""
    gesture_type: str
    accuracy_score: float
    execution_time: float
    smoothness_score: float
    consistency_score: float
    learning_progress: float
    common_errors: List[str]
    optimization_suggestions: List[str]

@dataclass
class VoiceAnalysis:
    """Voice command analysis results."""
    command_type: str
    recognition_accuracy: float
    response_time: float
    clarity_score: float
    confidence_score: float
    adaptation_progress: float
    common_mistakes: List[str]
    improvement_suggestions: List[str]

class InteractionAnalytics:
    """
    Comprehensive interaction analytics system for JARVIS.
    Provides advanced analysis of user interactions, performance metrics, and behavioral insights.
    """
    
    def __init__(self, analytics_dir="interaction_analytics"):
        self.analytics_dir = analytics_dir
        self.interaction_buffer = defaultdict(deque)  # Recent interactions
        self.analytics_cache = {}
        self.performance_metrics = defaultdict(dict)
        self.behavioral_patterns = defaultdict(dict)
        
        # Advanced analytics components
        self.gesture_analyzer = GesturePatternAnalyzer()
        self.voice_analyzer = VoiceInteractionAnalyzer()
        self.efficiency_analyzer = InteractionEfficiencyAnalyzer()
        self.behavioral_analyzer = BehavioralPatternAnalyzer()
        self.predictive_analyzer = PredictiveAnalyzer()
        
        # Machine learning models
        self.ml_models = {}
        self.anomaly_detectors = {}
        self.performance_predictors = {}
        
        # Real-time processing
        self.real_time_processor = RealTimeProcessor()
        self.event_stream = deque(maxlen=10000)  # Recent events
        
        # Create directories
        os.makedirs(analytics_dir, exist_ok=True)
        os.makedirs(f"{analytics_dir}/reports", exist_ok=True)
        os.makedirs(f"{analytics_dir}/models", exist_ok=True)
        
        # Background processing
        self.processing_thread = threading.Thread(target=self._background_processing, daemon=True)
        self.processing_thread.start()
        
        logging.info("Advanced Interaction Analytics initialized")
    
    def record_interaction(self, user_id: str, interaction_event: InteractionEvent):
        """Record a new interaction event for analysis."""
        # Add to recent interactions buffer
        self.interaction_buffer[user_id].append(interaction_event)
        
        # Add to event stream for real-time processing
        self.event_stream.append((user_id, interaction_event))
        
        # Limit buffer size
        if len(self.interaction_buffer[user_id]) > 1000:
            self.interaction_buffer[user_id].popleft()
        
        # Real-time processing
        self.real_time_processor.process_event(user_id, interaction_event)
        
        # Update performance metrics
        self._update_performance_metrics(user_id, interaction_event)
        
        logging.debug(f"Recorded interaction for user {user_id}: {interaction_event.event_type}")
    
    def analyze_gesture_patterns(self, user_id: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """Comprehensive gesture pattern analysis."""
        recent_gestures = self._get_recent_interactions(
            user_id, 
            event_types=['gesture'], 
            hours=time_window_hours
        )
        
        if not recent_gestures:
            return {"error": "No gesture data available"}
        
        analysis = self.gesture_analyzer.analyze_patterns(recent_gestures)
        
        return {
            "gesture_summary": analysis["summary"],
            "accuracy_trends": analysis["accuracy_trends"],
            "performance_metrics": analysis["performance_metrics"],
            "learning_insights": analysis["learning_insights"],
            "optimization_recommendations": analysis["recommendations"],
            "detailed_analysis": analysis["detailed_analysis"]
        }
    
    def analyze_voice_interactions(self, user_id: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """Comprehensive voice interaction analysis."""
        recent_voice = self._get_recent_interactions(
            user_id, 
            event_types=['voice'], 
            hours=time_window_hours
        )
        
        if not recent_voice:
            return {"error": "No voice interaction data available"}
        
        analysis = self.voice_analyzer.analyze_interactions(recent_voice)
        
        return {
            "voice_summary": analysis["summary"],
            "recognition_performance": analysis["recognition_performance"],
            "adaptation_progress": analysis["adaptation_progress"],
            "clarity_analysis": analysis["clarity_analysis"],
            "improvement_suggestions": analysis["improvement_suggestions"],
            "detailed_metrics": analysis["detailed_metrics"]
        }
    
    def analyze_interaction_efficiency(self, user_id: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze overall interaction efficiency and performance."""
        recent_interactions = self._get_recent_interactions(user_id, hours=time_window_hours)
        
        if not recent_interactions:
            return {"error": "No interaction data available"}
        
        analysis = self.efficiency_analyzer.analyze_efficiency(recent_interactions)
        
        return {
            "efficiency_score": analysis["overall_efficiency"],
            "performance_trends": analysis["trends"],
            "bottleneck_identification": analysis["bottlenecks"],
            "optimization_opportunities": analysis["optimizations"],
            "comparative_metrics": analysis["comparisons"],
            "predictive_insights": analysis["predictions"]
        }
    
    def analyze_behavioral_patterns(self, user_id: str, time_window_days: int = 7) -> Dict[str, Any]:
        """Advanced behavioral pattern analysis."""
        extended_interactions = self._get_recent_interactions(user_id, hours=time_window_days * 24)
        
        if not extended_interactions:
            return {"error": "No behavioral data available"}
        
        analysis = self.behavioral_analyzer.analyze_patterns(extended_interactions)
        
        return {
            "behavioral_clusters": analysis["clusters"],
            "usage_patterns": analysis["usage_patterns"],
            "adaptation_insights": analysis["adaptation_insights"],
            "anomaly_detection": analysis["anomalies"],
            "predictive_behavior": analysis["predictions"],
            "personalization_recommendations": analysis["personalization"]
        }
    
    def generate_performance_report(self, user_id: str, report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate comprehensive performance and analytics report."""
        report = {
            "user_id": user_id,
            "report_type": report_type,
            "generation_timestamp": datetime.now().isoformat(),
            "analysis_period": "last_7_days"
        }
        
        try:
            # Core analytics
            report["gesture_analysis"] = self.analyze_gesture_patterns(user_id, 168)  # 7 days
            report["voice_analysis"] = self.analyze_voice_interactions(user_id, 168)
            report["efficiency_analysis"] = self.analyze_interaction_efficiency(user_id, 168)
            report["behavioral_analysis"] = self.analyze_behavioral_patterns(user_id, 7)
            
            # Advanced insights
            report["predictive_insights"] = self.predictive_analyzer.generate_insights(user_id)
            report["ml_insights"] = self._generate_ml_insights(user_id)
            report["anomaly_report"] = self._detect_interaction_anomalies(user_id)
            
            # Performance summary
            report["performance_summary"] = self._generate_performance_summary(report)
            
            # Save report
            self._save_report(user_id, report)
            
        except Exception as e:
            logging.error(f"Error generating performance report for {user_id}: {e}")
            report["error"] = str(e)
        
        return report
    
    def get_real_time_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get real-time interaction metrics."""
        return self.real_time_processor.get_current_metrics(user_id)
    
    def predict_interaction_outcomes(self, user_id: str, interaction_context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict likely outcomes for upcoming interactions."""
        return self.predictive_analyzer.predict_outcomes(user_id, interaction_context)
    
    def _get_recent_interactions(self, user_id: str, event_types: List[str] = None, hours: int = 24) -> List[InteractionEvent]:
        """Get recent interactions for a user."""
        if user_id not in self.interaction_buffer:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_interactions = []
        
        for interaction in self.interaction_buffer[user_id]:
            interaction_time = datetime.fromisoformat(interaction.timestamp)
            if interaction_time >= cutoff_time:
                if event_types is None or interaction.event_type in event_types:
                    recent_interactions.append(interaction)
        
        return recent_interactions
    
    def _update_performance_metrics(self, user_id: str, interaction_event: InteractionEvent):
        """Update real-time performance metrics."""
        current_date = datetime.now().date().isoformat()
        
        if current_date not in self.performance_metrics[user_id]:
            self.performance_metrics[user_id][current_date] = {
                "total_interactions": 0,
                "successful_interactions": 0,
                "average_response_time": 0.0,
                "average_confidence": 0.0,
                "error_count": 0,
                "interaction_types": defaultdict(int)
            }
        
        metrics = self.performance_metrics[user_id][current_date]
        
        # Update counters
        metrics["total_interactions"] += 1
        if interaction_event.success:
            metrics["successful_interactions"] += 1
        else:
            metrics["error_count"] += 1
        
        # Update averages
        total = metrics["total_interactions"]
        metrics["average_response_time"] = (
            (metrics["average_response_time"] * (total - 1) + interaction_event.response_time) / total
        )
        metrics["average_confidence"] = (
            (metrics["average_confidence"] * (total - 1) + interaction_event.confidence) / total
        )
        
        # Update interaction type counts
        metrics["interaction_types"][interaction_event.event_type] += 1
    
    def _generate_ml_insights(self, user_id: str) -> Dict[str, Any]:
        """Generate machine learning based insights."""
        if not SKLEARN_AVAILABLE:
            return {"error": "Machine learning libraries not available"}
        
        insights = {}
        
        try:
            # Get interaction data for ML analysis
            interactions = self._get_recent_interactions(user_id, hours=168)  # 7 days
            
            if len(interactions) < 10:
                return {"error": "Insufficient data for ML analysis"}
            
            # Extract features for ML
            features = self._extract_ml_features(interactions)
            
            if features.size == 0:
                return {"error": "Could not extract features"}
            
            # Clustering analysis
            insights["interaction_clusters"] = self._perform_interaction_clustering(features)
            
            # Anomaly detection
            insights["anomaly_detection"] = self._detect_ml_anomalies(features)
            
            # Performance prediction
            insights["performance_prediction"] = self._predict_performance_trends(features)
            
        except Exception as e:
            logging.error(f"ML insights generation error: {e}")
            insights["error"] = str(e)
        
        return insights
    
    def _extract_ml_features(self, interactions: List[InteractionEvent]) -> np.ndarray:
        """Extract features for machine learning analysis."""
        if not interactions:
            return np.array([])
        
        features = []
        
        for interaction in interactions:
            feature_vector = [
                interaction.response_time,
                interaction.confidence,
                1.0 if interaction.success else 0.0,
                interaction.user_satisfaction or 0.5,
                len(interaction.context),
                hash(interaction.event_type) % 1000,  # Event type encoding
            ]
            
            # Add contextual features
            context = interaction.context
            feature_vector.extend([
                context.get("complexity", 1.0),
                context.get("noise_level", 0.0),
                context.get("user_fatigue", 0.0),
                context.get("environmental_factors", 0.0)
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _perform_interaction_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """Perform clustering analysis on interaction patterns."""
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
                cluster_features = features[cluster_mask]
                
                cluster_analysis[f"cluster_{i}"] = {
                    "size": int(np.sum(cluster_mask)),
                    "avg_response_time": float(np.mean(cluster_features[:, 0])),
                    "avg_confidence": float(np.mean(cluster_features[:, 1])),
                    "success_rate": float(np.mean(cluster_features[:, 2])),
                    "characteristics": self._interpret_cluster(cluster_features)
                }
            
            return {
                "n_clusters": n_clusters,
                "cluster_analysis": cluster_analysis,
                "cluster_labels": cluster_labels.tolist()
            }
            
        except Exception as e:
            logging.error(f"Clustering analysis error: {e}")
            return {"error": str(e)}
    
    def _detect_ml_anomalies(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in interaction patterns using ML."""
        try:
            # Isolation Forest for anomaly detection
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = isolation_forest.fit_predict(features)
            
            # Identify anomalies
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            normal_indices = np.where(anomaly_labels == 1)[0]
            
            return {
                "total_anomalies": len(anomaly_indices),
                "anomaly_rate": len(anomaly_indices) / len(features),
                "anomaly_indices": anomaly_indices.tolist(),
                "anomaly_characteristics": self._analyze_anomalies(features[anomaly_indices])
            }
            
        except Exception as e:
            logging.error(f"Anomaly detection error: {e}")
            return {"error": str(e)}
    
    def _predict_performance_trends(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict performance trends using historical data."""
        try:
            if len(features) < 20:
                return {"error": "Insufficient data for trend prediction"}
            
            # Extract time series data
            response_times = features[:, 0]
            confidence_scores = features[:, 1]
            success_rates = features[:, 2]
            
            # Simple trend analysis
            time_indices = np.arange(len(features))
            
            # Linear regression for trends
            response_trend = np.polyfit(time_indices, response_times, 1)[0]
            confidence_trend = np.polyfit(time_indices, confidence_scores, 1)[0]
            success_trend = np.polyfit(time_indices, success_rates, 1)[0]
            
            return {
                "response_time_trend": "improving" if response_trend < 0 else "declining" if response_trend > 0 else "stable",
                "confidence_trend": "improving" if confidence_trend > 0 else "declining" if confidence_trend < 0 else "stable",
                "success_rate_trend": "improving" if success_trend > 0 else "declining" if success_trend < 0 else "stable",
                "trend_strength": {
                    "response_time": abs(response_trend),
                    "confidence": abs(confidence_trend),
                    "success_rate": abs(success_trend)
                }
            }
            
        except Exception as e:
            logging.error(f"Performance prediction error: {e}")
            return {"error": str(e)}
    
    def _detect_interaction_anomalies(self, user_id: str) -> Dict[str, Any]:
        """Detect anomalies in interaction patterns."""
        interactions = self._get_recent_interactions(user_id, hours=168)  # 7 days
        
        if len(interactions) < 10:
            return {"error": "Insufficient data for anomaly detection"}
        
        anomalies = []
        
        # Statistical anomaly detection
        response_times = [i.response_time for i in interactions]
        confidences = [i.confidence for i in interactions]
        
        # Z-score based detection
        rt_mean, rt_std = np.mean(response_times), np.std(response_times)
        conf_mean, conf_std = np.mean(confidences), np.std(confidences)
        
        for i, interaction in enumerate(interactions):
            rt_zscore = abs((interaction.response_time - rt_mean) / max(rt_std, 0.1))
            conf_zscore = abs((interaction.confidence - conf_mean) / max(conf_std, 0.1))
            
            if rt_zscore > 2.5 or conf_zscore > 2.5:
                anomalies.append({
                    "index": i,
                    "timestamp": interaction.timestamp,
                    "type": interaction.event_type,
                    "anomaly_scores": {
                        "response_time_zscore": rt_zscore,
                        "confidence_zscore": conf_zscore
                    },
                    "details": {
                        "response_time": interaction.response_time,
                        "confidence": interaction.confidence,
                        "success": interaction.success
                    }
                })
        
        return {
            "total_anomalies": len(anomalies),
            "anomaly_rate": len(anomalies) / len(interactions),
            "detected_anomalies": anomalies[:10],  # Top 10 anomalies
            "analysis_period": "7_days"
        }
    
    def _generate_performance_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of performance metrics."""
        summary = {
            "overall_score": 0.0,
            "key_insights": [],
            "improvement_areas": [],
            "strengths": [],
            "recommendations": []
        }
        
        try:
            # Calculate overall performance score
            scores = []
            
            # Gesture performance
            if "gesture_analysis" in report and "performance_metrics" in report["gesture_analysis"]:
                gesture_score = report["gesture_analysis"]["performance_metrics"].get("overall_score", 0.5)
                scores.append(gesture_score)
            
            # Voice performance
            if "voice_analysis" in report and "recognition_performance" in report["voice_analysis"]:
                voice_score = report["voice_analysis"]["recognition_performance"].get("overall_score", 0.5)
                scores.append(voice_score)
            
            # Efficiency score
            if "efficiency_analysis" in report and "efficiency_score" in report["efficiency_analysis"]:
                efficiency_score = report["efficiency_analysis"]["efficiency_score"]
                scores.append(efficiency_score)
            
            if scores:
                summary["overall_score"] = np.mean(scores)
            
            # Generate insights based on performance
            if summary["overall_score"] > 0.8:
                summary["key_insights"].append("Excellent overall performance across all interaction modalities")
                summary["strengths"].append("High efficiency and accuracy")
            elif summary["overall_score"] > 0.6:
                summary["key_insights"].append("Good performance with room for optimization")
                summary["improvement_areas"].append("Focus on consistency and error reduction")
            else:
                summary["key_insights"].append("Performance below optimal levels")
                summary["improvement_areas"].append("Significant improvement needed across multiple areas")
            
            # Add specific recommendations
            summary["recommendations"] = self._generate_performance_recommendations(report)
            
        except Exception as e:
            logging.error(f"Performance summary generation error: {e}")
            summary["error"] = str(e)
        
        return summary
    
    def _generate_performance_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate specific performance recommendations."""
        recommendations = []
        
        try:
            # Gesture recommendations
            if "gesture_analysis" in report and "optimization_recommendations" in report["gesture_analysis"]:
                gesture_recs = report["gesture_analysis"]["optimization_recommendations"]
                recommendations.extend([f"Gesture: {rec}" for rec in gesture_recs[:2]])
            
            # Voice recommendations
            if "voice_analysis" in report and "improvement_suggestions" in report["voice_analysis"]:
                voice_recs = report["voice_analysis"]["improvement_suggestions"]
                recommendations.extend([f"Voice: {rec}" for rec in voice_recs[:2]])
            
            # Efficiency recommendations
            if "efficiency_analysis" in report and "optimization_opportunities" in report["efficiency_analysis"]:
                efficiency_recs = report["efficiency_analysis"]["optimization_opportunities"]
                recommendations.extend([f"Efficiency: {rec}" for rec in efficiency_recs[:2]])
            
        except Exception as e:
            logging.error(f"Recommendation generation error: {e}")
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    def _save_report(self, user_id: str, report: Dict[str, Any]):
        """Save analytics report to storage."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.analytics_dir}/reports/{user_id}_report_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logging.info(f"Analytics report saved: {filename}")
            
        except Exception as e:
            logging.error(f"Error saving report: {e}")
    
    def _background_processing(self):
        """Background processing for continuous analytics."""
        while True:
            try:
                # Process recent events
                if self.event_stream:
                    recent_events = list(self.event_stream)[-100:]  # Last 100 events
                    self._process_event_batch(recent_events)
                
                time.sleep(60)  # Process every minute
                
            except Exception as e:
                logging.error(f"Background processing error: {e}")
                time.sleep(60)
    
    def _process_event_batch(self, events: List[Tuple[str, InteractionEvent]]):
        """Process a batch of events for pattern detection."""
        # Group events by user
        user_events = defaultdict(list)
        for user_id, event in events:
            user_events[user_id].append(event)
        
        # Analyze each user's events
        for user_id, user_event_list in user_events.items():
            if len(user_event_list) >= 5:  # Minimum events for analysis
                self._update_behavioral_patterns(user_id, user_event_list)
    
    def _update_behavioral_patterns(self, user_id: str, events: List[InteractionEvent]):
        """Update behavioral patterns based on recent events."""
        current_date = datetime.now().date().isoformat()
        
        if current_date not in self.behavioral_patterns[user_id]:
            self.behavioral_patterns[user_id][current_date] = {
                "interaction_frequency": 0,
                "preferred_modalities": defaultdict(int),
                "peak_performance_hours": [],
                "error_patterns": [],
                "learning_indicators": {}
            }
        
        patterns = self.behavioral_patterns[user_id][current_date]
        
        # Update interaction frequency
        patterns["interaction_frequency"] += len(events)
        
        # Update modality preferences
        for event in events:
            patterns["preferred_modalities"][event.event_type] += 1
        
        # Analyze performance by hour
        hourly_performance = defaultdict(list)
        for event in events:
            hour = datetime.fromisoformat(event.timestamp).hour
            performance_score = event.confidence * (1.0 if event.success else 0.5)
            hourly_performance[hour].append(performance_score)
        
        # Update peak performance hours
        for hour, performances in hourly_performance.items():
            if np.mean(performances) > 0.8:  # High performance threshold
                if hour not in patterns["peak_performance_hours"]:
                    patterns["peak_performance_hours"].append(hour)


class GesturePatternAnalyzer:
    """Specialized analyzer for gesture interaction patterns."""
    
    def __init__(self):
        self.gesture_models = {}
        self.performance_history = defaultdict(list)
        
    def analyze_patterns(self, gesture_interactions: List[InteractionEvent]) -> Dict[str, Any]:
        """Analyze gesture patterns and performance."""
        analysis = {
            "summary": {},
            "accuracy_trends": {},
            "performance_metrics": {},
            "learning_insights": {},
            "recommendations": [],
            "detailed_analysis": {}
        }
        
        if not gesture_interactions:
            return analysis
        
        # Group by gesture type
        gesture_groups = defaultdict(list)
        for interaction in gesture_interactions:
            gesture_type = interaction.event_data.get("gesture_type", "unknown")
            gesture_groups[gesture_type].append(interaction)
        
        # Analyze each gesture type
        for gesture_type, interactions in gesture_groups.items():
            gesture_analysis = self._analyze_gesture_type(gesture_type, interactions)
            analysis["detailed_analysis"][gesture_type] = gesture_analysis
        
        # Generate summary
        analysis["summary"] = self._generate_gesture_summary(analysis["detailed_analysis"])
        analysis["performance_metrics"] = self._calculate_performance_metrics(gesture_interactions)
        analysis["accuracy_trends"] = self._analyze_accuracy_trends(gesture_interactions)
        analysis["learning_insights"] = self._generate_learning_insights(gesture_interactions)
        analysis["recommendations"] = self._generate_gesture_recommendations(analysis)
        
        return analysis
    
    def _analyze_gesture_type(self, gesture_type: str, interactions: List[InteractionEvent]) -> GestureAnalysis:
        """Analyze a specific gesture type."""
        # Calculate metrics
        accuracies = [i.confidence for i in interactions]
        response_times = [i.response_time for i in interactions]
        success_rates = [1.0 if i.success else 0.0 for i in interactions]
        
        # Calculate smoothness and consistency
        smoothness_scores = []
        consistency_scores = []
        
        for interaction in interactions:
            event_data = interaction.event_data
            smoothness = event_data.get("smoothness", 0.5)
            smoothness_scores.append(smoothness)
            
            consistency = 1.0 - event_data.get("variance", 0.5)
            consistency_scores.append(consistency)
        
        # Calculate learning progress
        learning_progress = self._calculate_learning_progress(interactions)
        
        # Identify common errors
        common_errors = self._identify_common_errors(interactions)
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            gesture_type, accuracies, response_times, common_errors
        )
        
        return GestureAnalysis(
            gesture_type=gesture_type,
            accuracy_score=np.mean(accuracies) if accuracies else 0.0,
            execution_time=np.mean(response_times) if response_times else 0.0,
            smoothness_score=np.mean(smoothness_scores) if smoothness_scores else 0.0,
            consistency_score=np.mean(consistency_scores) if consistency_scores else 0.0,
            learning_progress=learning_progress,
            common_errors=common_errors,
            optimization_suggestions=optimization_suggestions
        )
    
    def _calculate_learning_progress(self, interactions: List[InteractionEvent]) -> float:
        """Calculate learning progress for a gesture."""
        if len(interactions) < 5:
            return 0.0
        
        # Split into early and recent periods
        mid_point = len(interactions) // 2
        early_success = np.mean([1.0 if i.success else 0.0 for i in interactions[:mid_point]])
        recent_success = np.mean([1.0 if i.success else 0.0 for i in interactions[mid_point:]])
        
        return max(0.0, recent_success - early_success)
    
    def _identify_common_errors(self, interactions: List[InteractionEvent]) -> List[str]:
        """Identify common error patterns."""
        error_patterns = defaultdict(int)
        
        for interaction in interactions:
            if not interaction.success and interaction.error_details:
                error_type = interaction.error_details.get("error_type", "unknown")
                error_patterns[error_type] += 1
        
        # Return most common errors
        sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
        return [error for error, count in sorted_errors[:5]]
    
    def _generate_optimization_suggestions(self, gesture_type: str, accuracies: List[float], 
                                         response_times: List[float], common_errors: List[str]) -> List[str]:
        """Generate optimization suggestions for gesture performance."""
        suggestions = []
        
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        avg_response_time = np.mean(response_times) if response_times else 0.0
        
        if avg_accuracy < 0.7:
            suggestions.append(f"Practice {gesture_type} gesture for better accuracy")
        
        if avg_response_time > 2.0:
            suggestions.append(f"Work on speed for {gesture_type} gesture")
        
        for error in common_errors[:2]:
            suggestions.append(f"Focus on reducing {error} errors")
        
        if len(accuracies) > 10:
            accuracy_consistency = 1.0 - (np.std(accuracies) / max(np.mean(accuracies), 0.1))
            if accuracy_consistency < 0.7:
                suggestions.append(f"Improve consistency in {gesture_type} execution")
        
        return suggestions


class VoiceInteractionAnalyzer:
    """Specialized analyzer for voice interaction patterns."""
    
    def __init__(self):
        self.voice_models = {}
        self.adaptation_tracking = defaultdict(dict)
        
    def analyze_interactions(self, voice_interactions: List[InteractionEvent]) -> Dict[str, Any]:
        """Analyze voice interaction patterns."""
        analysis = {
            "summary": {},
            "recognition_performance": {},
            "adaptation_progress": {},
            "clarity_analysis": {},
            "improvement_suggestions": [],
            "detailed_metrics": {}
        }
        
        if not voice_interactions:
            return analysis
        
        # Group by command type
        command_groups = defaultdict(list)
        for interaction in voice_interactions:
            command_type = interaction.event_data.get("command_type", "unknown")
            command_groups[command_type].append(interaction)
        
        # Analyze each command type
        for command_type, interactions in command_groups.items():
            command_analysis = self._analyze_voice_command(command_type, interactions)
            analysis["detailed_metrics"][command_type] = asdict(command_analysis)
        
        # Generate comprehensive analysis
        analysis["summary"] = self._generate_voice_summary(analysis["detailed_metrics"])
        analysis["recognition_performance"] = self._analyze_recognition_performance(voice_interactions)
        analysis["adaptation_progress"] = self._analyze_adaptation_progress(voice_interactions)
        analysis["clarity_analysis"] = self._analyze_voice_clarity(voice_interactions)
        analysis["improvement_suggestions"] = self._generate_voice_improvements(analysis)
        
        return analysis
    
    def _analyze_voice_command(self, command_type: str, interactions: List[InteractionEvent]) -> VoiceAnalysis:
        """Analyze a specific voice command type."""
        # Calculate metrics
        recognition_scores = [i.confidence for i in interactions]
        response_times = [i.response_time for i in interactions]
        
        # Extract voice-specific metrics
        clarity_scores = []
        confidence_scores = []
        
        for interaction in interactions:
            event_data = interaction.event_data
            clarity_scores.append(event_data.get("clarity_score", 0.5))
            confidence_scores.append(event_data.get("voice_confidence", 0.5))
        
        # Calculate adaptation progress
        adaptation_progress = self._calculate_voice_adaptation(interactions)
        
        # Identify common mistakes
        common_mistakes = self._identify_voice_mistakes(interactions)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_voice_command_improvements(
            command_type, recognition_scores, clarity_scores, common_mistakes
        )
        
        return VoiceAnalysis(
            command_type=command_type,
            recognition_accuracy=np.mean(recognition_scores) if recognition_scores else 0.0,
            response_time=np.mean(response_times) if response_times else 0.0,
            clarity_score=np.mean(clarity_scores) if clarity_scores else 0.0,
            confidence_score=np.mean(confidence_scores) if confidence_scores else 0.0,
            adaptation_progress=adaptation_progress,
            common_mistakes=common_mistakes,
            improvement_suggestions=improvement_suggestions
        )
    
    def _calculate_voice_adaptation(self, interactions: List[InteractionEvent]) -> float:
        """Calculate voice adaptation progress."""
        if len(interactions) < 5:
            return 0.0
        
        # Measure improvement over time
        timestamps = [datetime.fromisoformat(i.timestamp) for i in interactions]
        confidences = [i.confidence for i in interactions]
        
        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, confidences))
        sorted_confidences = [conf for _, conf in sorted_data]
        
        # Calculate trend
        if len(sorted_confidences) >= 10:
            early_avg = np.mean(sorted_confidences[:len(sorted_confidences)//3])
            recent_avg = np.mean(sorted_confidences[-len(sorted_confidences)//3:])
            return max(0.0, recent_avg - early_avg)
        
        return 0.0
    
    def _identify_voice_mistakes(self, interactions: List[InteractionEvent]) -> List[str]:
        """Identify common voice recognition mistakes."""
        mistake_patterns = defaultdict(int)
        
        for interaction in interactions:
            if not interaction.success and interaction.error_details:
                mistake_type = interaction.error_details.get("recognition_error", "unknown")
                mistake_patterns[mistake_type] += 1
        
        # Return most common mistakes
        sorted_mistakes = sorted(mistake_patterns.items(), key=lambda x: x[1], reverse=True)
        return [mistake for mistake, count in sorted_mistakes[:5]]


class InteractionEfficiencyAnalyzer:
    """Analyzer for overall interaction efficiency and performance."""
    
    def __init__(self):
        self.efficiency_models = {}
        self.benchmark_data = {}
        
    def analyze_efficiency(self, interactions: List[InteractionEvent]) -> Dict[str, Any]:
        """Analyze overall interaction efficiency."""
        analysis = {
            "overall_efficiency": 0.0,
            "trends": {},
            "bottlenecks": [],
            "optimizations": [],
            "comparisons": {},
            "predictions": {}
        }
        
        if not interactions:
            return analysis
        
        # Calculate overall efficiency score
        analysis["overall_efficiency"] = self._calculate_efficiency_score(interactions)
        
        # Analyze trends
        analysis["trends"] = self._analyze_efficiency_trends(interactions)
        
        # Identify bottlenecks
        analysis["bottlenecks"] = self._identify_performance_bottlenecks(interactions)
        
        # Generate optimizations
        analysis["optimizations"] = self._generate_efficiency_optimizations(interactions)
        
        # Comparative analysis
        analysis["comparisons"] = self._generate_comparative_analysis(interactions)
        
        # Predictive insights
        analysis["predictions"] = self._generate_efficiency_predictions(interactions)
        
        return analysis
    
    def _calculate_efficiency_score(self, interactions: List[InteractionEvent]) -> float:
        """Calculate overall efficiency score."""
        if not interactions:
            return 0.0
        
        # Factors contributing to efficiency
        success_rate = np.mean([1.0 if i.success else 0.0 for i in interactions])
        avg_confidence = np.mean([i.confidence for i in interactions])
        avg_response_time = np.mean([i.response_time for i in interactions])
        
        # Normalize response time (lower is better)
        response_time_score = max(0.0, 1.0 - (avg_response_time / 5.0))  # 5 seconds as baseline
        
        # Calculate satisfaction score
        satisfaction_scores = [i.user_satisfaction for i in interactions if i.user_satisfaction is not None]
        avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 0.5
        
        # Weighted efficiency score
        efficiency_score = (
            0.3 * success_rate +
            0.25 * avg_confidence +
            0.25 * response_time_score +
            0.2 * avg_satisfaction
        )
        
        return min(1.0, max(0.0, efficiency_score))


class BehavioralPatternAnalyzer:
    """Analyzer for behavioral patterns and user adaptation."""
    
    def __init__(self):
        self.pattern_models = {}
        self.behavioral_clusters = {}
        
    def analyze_patterns(self, interactions: List[InteractionEvent]) -> Dict[str, Any]:
        """Analyze behavioral patterns in interactions."""
        analysis = {
            "clusters": {},
            "usage_patterns": {},
            "adaptation_insights": {},
            "anomalies": {},
            "predictions": {},
            "personalization": {}
        }
        
        if len(interactions) < 10:
            return analysis
        
        # Behavioral clustering
        analysis["clusters"] = self._perform_behavioral_clustering(interactions)
        
        # Usage pattern analysis
        analysis["usage_patterns"] = self._analyze_usage_patterns(interactions)
        
        # Adaptation insights
        analysis["adaptation_insights"] = self._generate_adaptation_insights(interactions)
        
        # Anomaly detection
        analysis["anomalies"] = self._detect_behavioral_anomalies(interactions)
        
        # Behavioral predictions
        analysis["predictions"] = self._predict_behavioral_trends(interactions)
        
        # Personalization recommendations
        analysis["personalization"] = self._generate_personalization_recommendations(interactions)
        
        return analysis
    
    def _perform_behavioral_clustering(self, interactions: List[InteractionEvent]) -> Dict[str, Any]:
        """Cluster interactions based on behavioral patterns."""
        if not SKLEARN_AVAILABLE or len(interactions) < 10:
            return {"error": "Insufficient data or ML libraries unavailable"}
        
        # Extract behavioral features
        features = []
        for interaction in interactions:
            hour = datetime.fromisoformat(interaction.timestamp).hour
            feature_vector = [
                interaction.response_time,
                interaction.confidence,
                1.0 if interaction.success else 0.0,
                hour / 24.0,  # Normalized hour
                len(interaction.context),
                interaction.user_satisfaction or 0.5
            ]
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        try:
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            # DBSCAN clustering for behavioral patterns
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            cluster_labels = dbscan.fit_predict(features_scaled)
            
            # Analyze clusters
            unique_labels = set(cluster_labels)
            cluster_analysis = {}
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                    
                cluster_mask = cluster_labels == label
                cluster_interactions = [interactions[i] for i, mask in enumerate(cluster_mask) if mask]
                
                cluster_analysis[f"pattern_{label}"] = {
                    "size": int(np.sum(cluster_mask)),
                    "avg_response_time": float(np.mean(features_array[cluster_mask, 0])),
                    "avg_confidence": float(np.mean(features_array[cluster_mask, 1])),
                    "success_rate": float(np.mean(features_array[cluster_mask, 2])),
                    "common_times": self._analyze_cluster_timing(cluster_interactions),
                    "characteristics": self._characterize_behavioral_cluster(cluster_interactions)
                }
            
            return {
                "n_clusters": len(unique_labels) - (1 if -1 in unique_labels else 0),
                "cluster_analysis": cluster_analysis,
                "noise_points": int(np.sum(cluster_labels == -1))
            }
            
        except Exception as e:
            logging.error(f"Behavioral clustering error: {e}")
            return {"error": str(e)}


class PredictiveAnalyzer:
    """Predictive analytics for interaction outcomes and user behavior."""
    
    def __init__(self):
        self.prediction_models = {}
        self.feature_importance = {}
        
    def generate_insights(self, user_id: str) -> Dict[str, Any]:
        """Generate predictive insights for a user."""
        insights = {
            "performance_predictions": {},
            "behavioral_predictions": {},
            "optimization_predictions": {},
            "risk_assessments": {},
            "adaptation_forecasts": {}
        }
        
        # This would integrate with the main analytics system
        # For now, return placeholder insights
        insights["performance_predictions"] = {
            "next_week_efficiency": 0.8,
            "improvement_areas": ["gesture_accuracy", "voice_clarity"],
            "success_probability": 0.85
        }
        
        return insights
    
    def predict_outcomes(self, user_id: str, interaction_context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcomes for upcoming interactions."""
        predictions = {
            "success_probability": 0.8,
            "expected_response_time": 1.5,
            "confidence_prediction": 0.85,
            "optimization_suggestions": [],
            "risk_factors": []
        }
        
        # Context-based predictions would be implemented here
        # This is a placeholder implementation
        
        return predictions


class RealTimeProcessor:
    """Real-time processing of interaction events."""
    
    def __init__(self):
        self.current_metrics = defaultdict(dict)
        self.event_buffer = defaultdict(deque)
        
    def process_event(self, user_id: str, event: InteractionEvent):
        """Process an event in real-time."""
        # Add to buffer
        self.event_buffer[user_id].append(event)
        
        # Limit buffer size
        if len(self.event_buffer[user_id]) > 100:
            self.event_buffer[user_id].popleft()
        
        # Update real-time metrics
        self._update_real_time_metrics(user_id, event)
    
    def get_current_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get current real-time metrics for a user."""
        return dict(self.current_metrics[user_id])
    
    def _update_real_time_metrics(self, user_id: str, event: InteractionEvent):
        """Update real-time metrics for a user."""
        metrics = self.current_metrics[user_id]
        
        # Update counters
        metrics["total_events"] = metrics.get("total_events", 0) + 1
        metrics["recent_success_rate"] = self._calculate_recent_success_rate(user_id)
        metrics["avg_response_time"] = self._calculate_avg_response_time(user_id)
        metrics["current_confidence"] = event.confidence
        metrics["last_event_time"] = event.timestamp
        metrics["event_frequency"] = self._calculate_event_frequency(user_id)
    
    def _calculate_recent_success_rate(self, user_id: str) -> float:
        """Calculate success rate for recent events."""
        recent_events = list(self.event_buffer[user_id])[-20:]  # Last 20 events
        if not recent_events:
            return 0.0
        
        success_count = sum(1 for event in recent_events if event.success)
        return success_count / len(recent_events)
    
    def _calculate_avg_response_time(self, user_id: str) -> float:
        """Calculate average response time for recent events."""
        recent_events = list(self.event_buffer[user_id])[-20:]  # Last 20 events
        if not recent_events:
            return 0.0
        
        response_times = [event.response_time for event in recent_events]
        return np.mean(response_times)
    
    def _calculate_event_frequency(self, user_id: str) -> float:
        """Calculate event frequency (events per minute)."""
        recent_events = list(self.event_buffer[user_id])
        if len(recent_events) < 2:
            return 0.0
        
        # Calculate time span of recent events
        first_time = datetime.fromisoformat(recent_events[0].timestamp)
        last_time = datetime.fromisoformat(recent_events[-1].timestamp)
        time_span_minutes = (last_time - first_time).total_seconds() / 60.0
        
        if time_span_minutes <= 0:
            return 0.0
        
        return len(recent_events) / time_span_minutes


# Helper functions for analytics
def _interpret_cluster(cluster_features: np.ndarray) -> Dict[str, str]:
    """Interpret cluster characteristics."""
    characteristics = {}
    
    avg_response_time = np.mean(cluster_features[:, 0])
    avg_confidence = np.mean(cluster_features[:, 1])
    avg_success = np.mean(cluster_features[:, 2])
    
    if avg_response_time < 1.0:
        characteristics["speed"] = "fast"
    elif avg_response_time > 3.0:
        characteristics["speed"] = "slow"
    else:
        characteristics["speed"] = "moderate"
    
    if avg_confidence > 0.8:
        characteristics["confidence"] = "high"
    elif avg_confidence < 0.6:
        characteristics["confidence"] = "low"
    else:
        characteristics["confidence"] = "moderate"
    
    if avg_success > 0.8:
        characteristics["reliability"] = "high"
    elif avg_success < 0.6:
        characteristics["reliability"] = "low"
    else:
        characteristics["reliability"] = "moderate"
    
    return characteristics


def _analyze_anomalies(anomaly_features: np.ndarray) -> Dict[str, Any]:
    """Analyze characteristics of detected anomalies."""
    if len(anomaly_features) == 0:
        return {}
    
    return {
        "avg_response_time": float(np.mean(anomaly_features[:, 0])),
        "avg_confidence": float(np.mean(anomaly_features[:, 1])),
        "success_rate": float(np.mean(anomaly_features[:, 2])),
        "anomaly_types": _classify_anomaly_types(anomaly_features)
    }


def _classify_anomaly_types(features: np.ndarray) -> List[str]:
    """Classify types of anomalies detected."""
    anomaly_types = []
    
    response_times = features[:, 0]
    confidences = features[:, 1]
    success_rates = features[:, 2]
    
    if np.mean(response_times) > 5.0:
        anomaly_types.append("slow_response")
    
    if np.mean(confidences) < 0.3:
        anomaly_types.append("low_confidence")
    
    if np.mean(success_rates) < 0.3:
        anomaly_types.append("high_failure_rate")
    
    return anomaly_types


if __name__ == "__main__":
    # Test the interaction analytics system
    print("Testing Advanced Interaction Analytics System...")
    
    # Initialize analytics
    analytics = InteractionAnalytics()
    
    # Create test interaction events
    test_events = [
        InteractionEvent(
            timestamp=datetime.now().isoformat(),
            event_type="gesture",
            event_data={"gesture_type": "swipe", "smoothness": 0.8},
            success=True,
            response_time=1.2,
            confidence=0.9,
            context={"complexity": 2, "noise_level": 0.1},
            user_satisfaction=0.8
        ),
        InteractionEvent(
            timestamp=datetime.now().isoformat(),
            event_type="voice",
            event_data={"command_type": "navigate", "clarity_score": 0.85},
            success=True,
            response_time=0.8,
            confidence=0.88,
            context={"complexity": 1, "noise_level": 0.2},
            user_satisfaction=0.9
        )
    ]
    
    # Record test events
    test_user_id = "test_user_analytics"
    for event in test_events:
        analytics.record_interaction(test_user_id, event)
    
    # Generate analytics
    print("Generating analytics reports...")
    
    gesture_analysis = analytics.analyze_gesture_patterns(test_user_id)
    print(f"Gesture Analysis: {gesture_analysis}")
    
    voice_analysis = analytics.analyze_voice_interactions(test_user_id)
    print(f"Voice Analysis: {voice_analysis}")
    
    efficiency_analysis = analytics.analyze_interaction_efficiency(test_user_id)
    print(f"Efficiency Analysis: {efficiency_analysis}")
    
    # Generate comprehensive report
    comprehensive_report = analytics.generate_performance_report(test_user_id)
    print(f"Comprehensive Report Generated: {len(comprehensive_report)} sections")
    
    print("Advanced Interaction Analytics System test completed!")