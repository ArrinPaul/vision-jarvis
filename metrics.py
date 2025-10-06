import time
import json
import logging
from typing import Dict, Any, List, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
import statistics


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    name: str
    value: float
    timestamp: float
    context: Dict[str, Any] = None


class MetricsCollector:
    """Collects and analyzes performance metrics for Jarvis-like responsiveness"""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.logger = logging.getLogger(__name__)
        
        # Metric storage
        self.metrics = defaultdict(lambda: deque(maxlen=max_samples))
        self.session_start = time.time()
        
        # Target thresholds for "Jarvis-like" performance
        self.targets = {
            "wake_to_listen": 150,      # ms - wake word to listening start
            "asr_latency": 300,         # ms - speech to text result
            "llm_response": 1000,       # ms - query to LLM response
            "tts_start": 200,           # ms - text to audio start
            "gesture_fps": 24,          # fps - gesture detection frame rate
            "command_dispatch": 50,     # ms - command to action execution
            "total_voice_latency": 2000 # ms - speech start to response start
        }
        
        # Quality thresholds
        self.quality_targets = {
            "asr_accuracy": 0.95,       # Speech recognition accuracy
            "gesture_confidence": 0.8,  # Gesture detection confidence
            "command_success_rate": 0.9 # Command execution success rate
        }
    
    def record_metric(self, name: str, value: float, context: Dict[str, Any] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(name, value, time.time(), context or {})
        self.metrics[name].append(metric)
        
        # Log if significantly above target
        target = self.targets.get(name)
        if target and value > target * 1.5:
            self.logger.warning(f"Performance alert: {name} = {value:.1f}ms (target: {target}ms)")
    
    def start_timer(self, name: str) -> str:
        """Start a timer and return a timer ID"""
        timer_id = f"{name}_{time.time()}"
        setattr(self, f"_timer_{timer_id}", time.time())
        return timer_id
    
    def end_timer(self, timer_id: str, context: Dict[str, Any] = None):
        """End a timer and record the metric"""
        start_time = getattr(self, f"_timer_{timer_id}", None)
        if start_time:
            duration = (time.time() - start_time) * 1000  # Convert to ms
            metric_name = timer_id.split("_")[0]
            self.record_metric(metric_name, duration, context)
            delattr(self, f"_timer_{timer_id}")
    
    def get_metric_stats(self, name: str, window_minutes: int = 10) -> Dict[str, float]:
        """Get statistics for a metric within a time window"""
        if name not in self.metrics:
            return {}
        
        cutoff_time = time.time() - (window_minutes * 60)
        recent_values = [
            m.value for m in self.metrics[name] 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_values:
            return {}
        
        return {
            "count": len(recent_values),
            "mean": statistics.mean(recent_values),
            "median": statistics.median(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "p95": statistics.quantiles(recent_values, n=20)[18] if len(recent_values) > 5 else max(recent_values),
            "std": statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        }
    
    def get_jarvis_readiness_score(self) -> Dict[str, Any]:
        """Calculate overall Jarvis readiness score"""
        scores = {}
        overall_score = 0
        total_weight = 0
        
        # Performance metrics (70% of score)
        perf_weights = {
            "wake_to_listen": 0.15,
            "asr_latency": 0.20,
            "llm_response": 0.15,
            "tts_start": 0.10,
            "gesture_fps": 0.05,
            "command_dispatch": 0.05
        }
        
        for metric, weight in perf_weights.items():
            stats = self.get_metric_stats(metric, window_minutes=30)
            if stats:
                target = self.targets[metric]
                if metric == "gesture_fps":
                    # Higher is better for FPS
                    score = min(100, (stats["mean"] / target) * 100)
                else:
                    # Lower is better for latency
                    score = max(0, min(100, (target / stats["mean"]) * 100))
                
                scores[metric] = {
                    "score": score,
                    "current": stats["mean"],
                    "target": target,
                    "status": "good" if score >= 80 else "fair" if score >= 60 else "poor"
                }
                
                overall_score += score * weight
                total_weight += weight
        
        # Quality metrics (30% of score)
        quality_weights = {
            "asr_accuracy": 0.15,
            "gesture_confidence": 0.10,
            "command_success_rate": 0.05
        }
        
        for metric, weight in quality_weights.items():
            stats = self.get_metric_stats(metric, window_minutes=30)
            if stats:
                target = self.quality_targets[metric]
                score = min(100, (stats["mean"] / target) * 100)
                
                scores[metric] = {
                    "score": score,
                    "current": stats["mean"],
                    "target": target,
                    "status": "good" if score >= 90 else "fair" if score >= 75 else "poor"
                }
                
                overall_score += score * weight
                total_weight += weight
        
        # Calculate final score
        final_score = overall_score / total_weight if total_weight > 0 else 0
        
        # Determine overall status
        if final_score >= 85:
            status = "excellent"
            message = "System is performing at Jarvis-like levels"
        elif final_score >= 70:
            status = "good"
            message = "System is responsive with minor optimization opportunities"
        elif final_score >= 50:
            status = "fair"
            message = "System is functional but has noticeable delays"
        else:
            status = "poor"
            message = "System needs significant optimization"
        
        return {
            "overall_score": final_score,
            "status": status,
            "message": message,
            "metrics": scores,
            "session_duration": time.time() - self.session_start,
            "recommendations": self._generate_recommendations(scores)
        }
    
    def _generate_recommendations(self, scores: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        for metric, data in scores.items():
            if data["score"] < 70:
                if metric == "wake_to_listen":
                    recommendations.append("Consider optimizing wake word detection or using a lighter model")
                elif metric == "asr_latency":
                    recommendations.append("Try faster-whisper with a smaller model or optimize audio preprocessing")
                elif metric == "llm_response":
                    recommendations.append("Consider using a faster LLM model or implementing response caching")
                elif metric == "tts_start":
                    recommendations.append("Switch to streaming TTS or optimize audio pipeline")
                elif metric == "gesture_fps":
                    recommendations.append("Reduce gesture detection resolution or optimize MediaPipe settings")
                elif metric == "command_dispatch":
                    recommendations.append("Optimize action registry or reduce command processing overhead")
        
        if not recommendations:
            recommendations.append("System is performing well! Consider fine-tuning for even better responsiveness")
        
        return recommendations
    
    def export_metrics(self, filepath: str = "jarvis_metrics.json"):
        """Export metrics to file"""
        export_data = {
            "session_start": self.session_start,
            "export_time": time.time(),
            "targets": self.targets,
            "quality_targets": self.quality_targets,
            "metrics": {}
        }
        
        for name, metric_list in self.metrics.items():
            export_data["metrics"][name] = [
                {
                    "value": m.value,
                    "timestamp": m.timestamp,
                    "context": m.context
                }
                for m in metric_list
            ]
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
            self.logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
    
    def get_performance_summary(self) -> str:
        """Get a human-readable performance summary"""
        readiness = self.get_jarvis_readiness_score()
        
        summary = [
            f"Jarvis Readiness Score: {readiness['overall_score']:.1f}/100 ({readiness['status'].upper()})",
            f"Status: {readiness['message']}",
            "",
            "Performance Breakdown:"
        ]
        
        for metric, data in readiness["metrics"].items():
            if metric in self.targets:
                unit = "fps" if metric == "gesture_fps" else "ms"
                summary.append(
                    f"  {metric.replace('_', ' ').title()}: "
                    f"{data['current']:.1f}{unit} (target: {data['target']}{unit}) - {data['status'].upper()}"
                )
            else:
                summary.append(
                    f"  {metric.replace('_', ' ').title()}: "
                    f"{data['current']:.2f} (target: {data['target']:.2f}) - {data['status'].upper()}"
                )
        
        if readiness["recommendations"]:
            summary.extend(["", "Recommendations:"])
            for rec in readiness["recommendations"]:
                summary.append(f"  • {rec}")
        
        return "\n".join(summary)


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_metric(name: str, value: float, context: Dict[str, Any] = None):
    """Convenience function to record a metric"""
    get_metrics_collector().record_metric(name, value, context)


def start_timer(name: str) -> str:
    """Convenience function to start a timer"""
    return get_metrics_collector().start_timer(name)


def end_timer(timer_id: str, context: Dict[str, Any] = None):
    """Convenience function to end a timer"""
    get_metrics_collector().end_timer(timer_id, context)


def get_jarvis_readiness_report() -> str:
    """Get a comprehensive Jarvis readiness report"""
    return get_metrics_collector().get_performance_summary()
