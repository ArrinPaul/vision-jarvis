"""
Security Monitoring System for JARVIS

Implements real-time security monitoring, threat detection,
and incident response capabilities.
"""

import os
import json
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from enum import Enum

class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SecurityEvent(Enum):
    """Types of security events."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    BRUTE_FORCE = "brute_force"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RESOURCE_ABUSE = "resource_abuse"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SYSTEM_ANOMALY = "system_anomaly"
    DATA_BREACH = "data_breach"
    MALWARE_DETECTION = "malware_detection"

class SecurityMonitor:
    """Real-time security monitoring system."""
    
    def __init__(self, config_file: str = "security_monitor_config.json"):
        self.config_file = config_file
        self.monitoring_active = False
        self.alerts = deque(maxlen=1000)
        self.threat_indicators = defaultdict(list)
        self.security_metrics = {}
        self.event_handlers = {}
        self.monitoring_thread = None
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize monitoring components
        self._initialize_monitoring()
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load monitoring configuration."""
        default_config = {
            "monitoring_interval_seconds": 5,
            "alert_thresholds": {
                "failed_login_attempts": 5,
                "cpu_usage_percent": 90,
                "memory_usage_percent": 85,
                "disk_usage_percent": 95,
                "network_connections": 100,
                "suspicious_process_count": 10
            },
            "threat_scoring": {
                "failed_login": 10,
                "resource_spike": 15,
                "unusual_process": 20,
                "unauthorized_access": 30,
                "privilege_escalation": 40
            },
            "auto_response": {
                "enabled": True,
                "block_suspicious_ips": True,
                "quarantine_threats": True,
                "notify_admin": True
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
                print(f"Failed to load monitoring config: {e}")
                return default_config
        else:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _initialize_monitoring(self):
        """Initialize monitoring components."""
        # Initialize baseline metrics
        self.security_metrics = {
            "system_start_time": datetime.now().isoformat(),
            "total_alerts": 0,
            "threat_level": ThreatLevel.LOW.name,
            "last_scan": None,
            "blocked_ips": set(),
            "quarantined_processes": set(),
            "active_connections": 0,
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0
        }
        
        # Register default event handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default security event handlers."""
        self.register_event_handler("unauthorized_access", self._handle_unauthorized_access)
        self.register_event_handler("brute_force", self._handle_brute_force)
        self.register_event_handler("resource_abuse", self._handle_resource_abuse)
        self.register_event_handler("system_anomaly", self._handle_system_anomaly)
    
    def start_monitoring(self) -> bool:
        """Start security monitoring."""
        if self.monitoring_active:
            return True
        
        try:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self._create_alert(
                ThreatLevel.LOW,
                SecurityEvent.SYSTEM_ANOMALY,
                "Security monitoring started",
                {"component": "SecurityMonitor"}
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to start monitoring: {e}")
            self.monitoring_active = False
            return False
    
    def stop_monitoring(self):
        """Stop security monitoring."""
        self.monitoring_active = False
        
        self._create_alert(
            ThreatLevel.LOW,
            SecurityEvent.SYSTEM_ANOMALY,
            "Security monitoring stopped",
            {"component": "SecurityMonitor"}
        )
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Perform security checks
                self._check_system_resources()
                self._check_network_activity()
                self._check_process_activity()
                self._check_login_attempts()
                self._update_threat_level()
                
                # Update last scan time
                self.security_metrics["last_scan"] = datetime.now().isoformat()
                
                # Sleep until next check
                time.sleep(self.config["monitoring_interval_seconds"])
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _check_system_resources(self):
        """Monitor system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.security_metrics["cpu_usage"] = cpu_percent
            
            if cpu_percent > self.config["alert_thresholds"]["cpu_usage_percent"]:
                self._create_alert(
                    ThreatLevel.MEDIUM,
                    SecurityEvent.RESOURCE_ABUSE,
                    f"High CPU usage detected: {cpu_percent:.1f}%",
                    {"cpu_usage": cpu_percent, "threshold": self.config["alert_thresholds"]["cpu_usage_percent"]}
                )
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.security_metrics["memory_usage"] = memory_percent
            
            if memory_percent > self.config["alert_thresholds"]["memory_usage_percent"]:
                self._create_alert(
                    ThreatLevel.MEDIUM,
                    SecurityEvent.RESOURCE_ABUSE,
                    f"High memory usage detected: {memory_percent:.1f}%",
                    {"memory_usage": memory_percent, "threshold": self.config["alert_thresholds"]["memory_usage_percent"]}
                )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.security_metrics["disk_usage"] = disk_percent
            
            if disk_percent > self.config["alert_thresholds"]["disk_usage_percent"]:
                self._create_alert(
                    ThreatLevel.HIGH,
                    SecurityEvent.RESOURCE_ABUSE,
                    f"High disk usage detected: {disk_percent:.1f}%",
                    {"disk_usage": disk_percent, "threshold": self.config["alert_thresholds"]["disk_usage_percent"]}
                )
                
        except Exception as e:
            print(f"Resource monitoring error: {e}")
    
    def _check_network_activity(self):
        """Monitor network activity for suspicious patterns."""
        try:
            connections = psutil.net_connections()
            active_connections = len([conn for conn in connections if conn.status == 'ESTABLISHED'])
            
            self.security_metrics["active_connections"] = active_connections
            
            if active_connections > self.config["alert_thresholds"]["network_connections"]:
                self._create_alert(
                    ThreatLevel.MEDIUM,
                    SecurityEvent.SUSPICIOUS_ACTIVITY,
                    f"High number of network connections: {active_connections}",
                    {"connection_count": active_connections, "threshold": self.config["alert_thresholds"]["network_connections"]}
                )
            
            # Check for suspicious IP addresses
            remote_ips = defaultdict(int)
            for conn in connections:
                if conn.raddr and conn.status == 'ESTABLISHED':
                    remote_ips[conn.raddr.ip] += 1
            
            # Flag IPs with too many connections
            for ip, count in remote_ips.items():
                if count > 10:  # Threshold for suspicious activity
                    self._create_alert(
                        ThreatLevel.HIGH,
                        SecurityEvent.SUSPICIOUS_ACTIVITY,
                        f"Suspicious activity from IP {ip}: {count} connections",
                        {"ip_address": ip, "connection_count": count}
                    )
                    
                    # Auto-block if enabled
                    if self.config["auto_response"]["block_suspicious_ips"]:
                        self._block_ip(ip)
                        
        except Exception as e:
            print(f"Network monitoring error: {e}")
    
    def _check_process_activity(self):
        """Monitor running processes for suspicious activity."""
        try:
            processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'create_time']))
            
            suspicious_processes = []
            high_resource_processes = []
            
            for proc in processes:
                try:
                    info = proc.info
                    
                    # Check for high resource usage
                    if info['cpu_percent'] > 50 or info['memory_percent'] > 20:
                        high_resource_processes.append(info)
                    
                    # Check for suspicious process names (basic heuristics)
                    suspicious_names = ['keylogger', 'trojan', 'backdoor', 'rootkit', 'malware']
                    if any(name in info['name'].lower() for name in suspicious_names):
                        suspicious_processes.append(info)
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Alert on suspicious processes
            if suspicious_processes:
                self._create_alert(
                    ThreatLevel.CRITICAL,
                    SecurityEvent.MALWARE_DETECTION,
                    f"Suspicious processes detected: {len(suspicious_processes)}",
                    {"processes": [p['name'] for p in suspicious_processes]}
                )
            
            # Alert on high resource usage
            if len(high_resource_processes) > self.config["alert_thresholds"]["suspicious_process_count"]:
                self._create_alert(
                    ThreatLevel.MEDIUM,
                    SecurityEvent.RESOURCE_ABUSE,
                    f"Multiple high-resource processes: {len(high_resource_processes)}",
                    {"process_count": len(high_resource_processes)}
                )
                
        except Exception as e:
            print(f"Process monitoring error: {e}")
    
    def _check_login_attempts(self):
        """Monitor for suspicious login patterns."""
        # This is a placeholder - in a real system, this would check actual login logs
        try:
            # Simulate checking failed login attempts from security logs
            failed_attempts_file = "failed_logins.log"
            
            if os.path.exists(failed_attempts_file):
                with open(failed_attempts_file, 'r') as f:
                    recent_attempts = []
                    current_time = datetime.now()
                    
                    for line in f.readlines()[-100:]:  # Check last 100 entries
                        try:
                            log_entry = json.loads(line.strip())
                            log_time = datetime.fromisoformat(log_entry['timestamp'])
                            
                            # Only consider attempts from last hour
                            if current_time - log_time < timedelta(hours=1):
                                recent_attempts.append(log_entry)
                        except:
                            continue
                    
                    # Group by IP address
                    ip_attempts = defaultdict(int)
                    for attempt in recent_attempts:
                        ip_attempts[attempt.get('ip_address', 'unknown')] += 1
                    
                    # Check for brute force patterns
                    for ip, count in ip_attempts.items():
                        if count > self.config["alert_thresholds"]["failed_login_attempts"]:
                            self._create_alert(
                                ThreatLevel.HIGH,
                                SecurityEvent.BRUTE_FORCE,
                                f"Brute force attack detected from {ip}: {count} failed attempts",
                                {"ip_address": ip, "attempt_count": count}
                            )
                            
                            if self.config["auto_response"]["block_suspicious_ips"]:
                                self._block_ip(ip)
                                
        except Exception as e:
            print(f"Login monitoring error: {e}")
    
    def _update_threat_level(self):
        """Update overall system threat level."""
        threat_score = 0
        
        # Calculate threat score based on recent alerts
        recent_time = datetime.now() - timedelta(hours=1)
        
        for alert in list(self.alerts):
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if alert_time > recent_time:
                threat_level = ThreatLevel[alert['threat_level']]
                threat_score += threat_level.value * 10
        
        # Determine threat level
        if threat_score < 50:
            current_threat = ThreatLevel.LOW
        elif threat_score < 100:
            current_threat = ThreatLevel.MEDIUM
        elif threat_score < 200:
            current_threat = ThreatLevel.HIGH
        else:
            current_threat = ThreatLevel.CRITICAL
        
        # Update if changed
        if self.security_metrics["threat_level"] != current_threat.name:
            old_level = self.security_metrics["threat_level"]
            self.security_metrics["threat_level"] = current_threat.name
            
            self._create_alert(
                current_threat,
                SecurityEvent.SYSTEM_ANOMALY,
                f"Threat level changed from {old_level} to {current_threat.name}",
                {"old_level": old_level, "new_level": current_threat.name, "threat_score": threat_score}
            )
    
    def _create_alert(self, threat_level: ThreatLevel, event_type: SecurityEvent, 
                     message: str, details: Dict[str, Any] = None):
        """Create security alert."""
        alert = {
            "id": f"alert_{int(time.time())}_{len(self.alerts)}",
            "timestamp": datetime.now().isoformat(),
            "threat_level": threat_level.name,
            "event_type": event_type.value,
            "message": message,
            "details": details or {},
            "status": "active",
            "acknowledged": False
        }
        
        self.alerts.append(alert)
        self.security_metrics["total_alerts"] += 1
        
        # Trigger event handlers
        self._trigger_event_handlers(event_type, alert)
        
        # Auto-response if critical
        if threat_level == ThreatLevel.CRITICAL and self.config["auto_response"]["enabled"]:
            self._execute_auto_response(alert)
        
        print(f"SECURITY ALERT [{threat_level.name}]: {message}")
    
    def _trigger_event_handlers(self, event_type: SecurityEvent, alert: Dict[str, Any]):
        """Trigger registered event handlers."""
        handler = self.event_handlers.get(event_type.value)
        if handler:
            try:
                handler(alert)
            except Exception as e:
                print(f"Event handler error for {event_type.value}: {e}")
    
    def _execute_auto_response(self, alert: Dict[str, Any]):
        """Execute automated response to critical threats."""
        event_type = alert["event_type"]
        details = alert["details"]
        
        if event_type == SecurityEvent.BRUTE_FORCE.value:
            ip_address = details.get("ip_address")
            if ip_address:
                self._block_ip(ip_address)
        
        elif event_type == SecurityEvent.MALWARE_DETECTION.value:
            processes = details.get("processes", [])
            for process_name in processes:
                self._quarantine_process(process_name)
        
        # Notify admin if configured
        if self.config["auto_response"]["notify_admin"]:
            self._notify_admin(alert)
    
    def _block_ip(self, ip_address: str):
        """Block suspicious IP address."""
        if ip_address not in self.security_metrics["blocked_ips"]:
            self.security_metrics["blocked_ips"].add(ip_address)
            
            # In a real implementation, this would add firewall rules
            print(f"BLOCKED IP: {ip_address}")
            
            self._create_alert(
                ThreatLevel.MEDIUM,
                SecurityEvent.SUSPICIOUS_ACTIVITY,
                f"IP address blocked: {ip_address}",
                {"blocked_ip": ip_address, "reason": "automated_response"}
            )
    
    def _quarantine_process(self, process_name: str):
        """Quarantine suspicious process."""
        if process_name not in self.security_metrics["quarantined_processes"]:
            self.security_metrics["quarantined_processes"].add(process_name)
            
            # In a real implementation, this would terminate/quarantine the process
            print(f"QUARANTINED PROCESS: {process_name}")
            
            self._create_alert(
                ThreatLevel.HIGH,
                SecurityEvent.MALWARE_DETECTION,
                f"Process quarantined: {process_name}",
                {"quarantined_process": process_name, "reason": "automated_response"}
            )
    
    def _notify_admin(self, alert: Dict[str, Any]):
        """Notify system administrator."""
        # In a real implementation, this would send email/SMS/push notification
        print(f"ADMIN NOTIFICATION: {alert['message']}")
    
    def register_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]):
        """Register custom event handler."""
        self.event_handlers[event_type] = handler
    
    def _handle_unauthorized_access(self, alert: Dict[str, Any]):
        """Handle unauthorized access attempts."""
        details = alert["details"]
        ip_address = details.get("ip_address")
        
        if ip_address:
            self._block_ip(ip_address)
    
    def _handle_brute_force(self, alert: Dict[str, Any]):
        """Handle brute force attacks."""
        details = alert["details"]
        ip_address = details.get("ip_address")
        
        if ip_address:
            self._block_ip(ip_address)
    
    def _handle_resource_abuse(self, alert: Dict[str, Any]):
        """Handle resource abuse."""
        # Could implement process throttling or termination
        pass
    
    def _handle_system_anomaly(self, alert: Dict[str, Any]):
        """Handle system anomalies."""
        # Could implement system health checks or recovery procedures
        pass
    
    def acknowledge_alert(self, alert_id: str, admin_user: str = "system") -> bool:
        """Acknowledge security alert."""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_by"] = admin_user
                alert["acknowledged_at"] = datetime.now().isoformat()
                return True
        return False
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        recent_alerts = [alert for alert in list(self.alerts)[-10:]]
        
        # Count alerts by threat level
        threat_counts = defaultdict(int)
        for alert in list(self.alerts):
            threat_counts[alert["threat_level"]] += 1
        
        # Count alerts by type
        event_counts = defaultdict(int)
        for alert in list(self.alerts):
            event_counts[alert["event_type"]] += 1
        
        return {
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "current_threat_level": self.security_metrics["threat_level"],
            "system_metrics": {
                "cpu_usage": self.security_metrics["cpu_usage"],
                "memory_usage": self.security_metrics["memory_usage"],
                "disk_usage": self.security_metrics["disk_usage"],
                "active_connections": self.security_metrics["active_connections"]
            },
            "security_stats": {
                "total_alerts": self.security_metrics["total_alerts"],
                "blocked_ips": len(self.security_metrics["blocked_ips"]),
                "quarantined_processes": len(self.security_metrics["quarantined_processes"]),
                "last_scan": self.security_metrics["last_scan"]
            },
            "recent_alerts": recent_alerts,
            "threat_distribution": dict(threat_counts),
            "event_distribution": dict(event_counts),
            "configuration": {
                "monitoring_interval": self.config["monitoring_interval_seconds"],
                "auto_response_enabled": self.config["auto_response"]["enabled"]
            }
        }
    
    def generate_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report for specified time period."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Filter alerts for time period
        period_alerts = []
        for alert in list(self.alerts):
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if start_time <= alert_time <= end_time:
                period_alerts.append(alert)
        
        # Analyze alerts
        threat_summary = defaultdict(int)
        event_summary = defaultdict(int)
        
        for alert in period_alerts:
            threat_summary[alert["threat_level"]] += 1
            event_summary[alert["event_type"]] += 1
        
        return {
            "report_period": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_hours": hours
            },
            "summary": {
                "total_alerts": len(period_alerts),
                "unique_threat_types": len(event_summary),
                "highest_threat_level": max(threat_summary.keys()) if threat_summary else "NONE"
            },
            "threat_analysis": dict(threat_summary),
            "event_analysis": dict(event_summary),
            "recommendations": self._generate_security_recommendations(period_alerts),
            "system_health": {
                "current_threat_level": self.security_metrics["threat_level"],
                "monitoring_uptime": self._calculate_uptime(),
                "blocked_threats": len(self.security_metrics["blocked_ips"]) + len(self.security_metrics["quarantined_processes"])
            }
        }
    
    def _generate_security_recommendations(self, alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on alerts."""
        recommendations = []
        
        # Analyze alert patterns
        event_counts = defaultdict(int)
        for alert in alerts:
            event_counts[alert["event_type"]] += 1
        
        if event_counts[SecurityEvent.BRUTE_FORCE.value] > 5:
            recommendations.append("Consider implementing rate limiting for login attempts")
        
        if event_counts[SecurityEvent.RESOURCE_ABUSE.value] > 10:
            recommendations.append("Review system resources and consider hardware upgrades")
        
        if event_counts[SecurityEvent.SUSPICIOUS_ACTIVITY.value] > 15:
            recommendations.append("Enhance network monitoring and consider implementing intrusion detection")
        
        if not recommendations:
            recommendations.append("System security appears stable - maintain current monitoring")
        
        return recommendations
    
    def _calculate_uptime(self) -> str:
        """Calculate monitoring system uptime."""
        start_time = datetime.fromisoformat(self.security_metrics["system_start_time"])
        uptime = datetime.now() - start_time
        
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        return f"{days}d {hours}h {minutes}m"


# Example usage and testing
if __name__ == "__main__":
    # Initialize security monitor
    monitor = SecurityMonitor()
    
    # Register custom event handler
    def custom_alert_handler(alert):
        print(f"CUSTOM HANDLER: {alert['message']}")
    
    monitor.register_event_handler("suspicious_activity", custom_alert_handler)
    
    # Start monitoring
    if monitor.start_monitoring():
        print("Security monitoring started successfully")
        
        # Simulate running for a short time
        time.sleep(10)
        
        # Create a test alert
        monitor._create_alert(
            ThreatLevel.HIGH,
            SecurityEvent.UNAUTHORIZED_ACCESS,
            "Test security alert",
            {"test": True, "ip_address": "192.168.1.999"}
        )
        
        # Get dashboard data
        dashboard = monitor.get_security_dashboard()
        print("Security Dashboard:")
        print(json.dumps(dashboard, indent=2))
        
        # Generate security report
        report = monitor.generate_security_report(1)  # Last 1 hour
        print("\nSecurity Report:")
        print(json.dumps(report, indent=2))
        
        # Stop monitoring
        monitor.stop_monitoring()
        
    else:
        print("Failed to start security monitoring")