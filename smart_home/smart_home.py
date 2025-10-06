import json
import threading
import time
from datetime import datetime
from .device_connectors import (PhilipsHueConnector, TPLinkKasaConnector, 
                               WiFiDeviceConnector, BluetoothDeviceConnector)

class SmartHomeController:
    """
    Advanced Smart Home/Device Integration supporting multiple protocols and device types.
    Features: Device discovery, automation, scheduling, and voice/gesture control.
    """
    def __init__(self, config_file="smart_home_config.json"):
        self.config_file = config_file
        self.devices = {}
        self.device_connectors = {}
        self.automation_rules = {}
        self.device_groups = {}
        self.is_monitoring = False
        
        # Load configuration
        self.load_configuration()
        
        # Start device monitoring
        self.start_monitoring()
        
    def load_configuration(self):
        """Load smart home configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.devices = config.get('devices', {})
                self.automation_rules = config.get('automation_rules', {})
                self.device_groups = config.get('device_groups', {})
        except FileNotFoundError:
            print("No smart home configuration found. Starting with empty config.")
            self.save_configuration()
        except Exception as e:
            print(f"Error loading configuration: {e}")
            
    def save_configuration(self):
        """Save smart home configuration to file."""
        try:
            config = {
                'devices': self.devices,
                'automation_rules': self.automation_rules,
                'device_groups': self.device_groups,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving configuration: {e}")
            
    def discover_devices(self):
        """Discover smart devices on the network."""
        print("Starting device discovery...")
        discovered_devices = []
        
        # Discover Philips Hue bridges
        hue_devices = self._discover_hue_bridges()
        discovered_devices.extend(hue_devices)
        
        # Discover TP-Link Kasa devices
        kasa_devices = self._discover_kasa_devices()
        discovered_devices.extend(kasa_devices)
        
        # Discover generic WiFi devices
        wifi_devices = self._discover_wifi_devices()
        discovered_devices.extend(wifi_devices)
        
        print(f"Discovered {len(discovered_devices)} devices")
        return discovered_devices
        
    def _discover_hue_bridges(self):
        """Discover Philips Hue bridges using UPnP."""
        # This would typically use UPnP discovery
        # For now, return simulated devices
        return [
            {
                'id': 'hue_bridge_1',
                'name': 'Philips Hue Bridge',
                'type': 'philips_hue',
                'ip': '192.168.1.100',
                'status': 'online'
            }
        ]
        
    def _discover_kasa_devices(self):
        """Discover TP-Link Kasa devices."""
        # This would scan the network for Kasa devices
        return [
            {
                'id': 'kasa_plug_1',
                'name': 'TP-Link Smart Plug',
                'type': 'tplink_kasa',
                'ip': '192.168.1.101',
                'status': 'online'
            }
        ]
        
    def _discover_wifi_devices(self):
        """Discover generic WiFi smart devices."""
        return []
        
    def connect_device(self, device_info):
        """Connect to a smart home device."""
        device_id = device_info.get('id')
        device_type = device_info.get('type')
        
        if not device_id or not device_type:
            print("Invalid device info: missing id or type")
            return False
            
        # Create appropriate connector
        connector = None
        
        if device_type == 'philips_hue':
            bridge_ip = device_info.get('ip')
            username = device_info.get('username', 'jarvis_user')
            connector = PhilipsHueConnector(bridge_ip, username)
            
        elif device_type == 'tplink_kasa':
            device_ip = device_info.get('ip')
            connector = TPLinkKasaConnector(device_ip)
            
        elif device_type == 'wifi_generic':
            device_ip = device_info.get('ip')
            api_endpoint = device_info.get('api_endpoint', '/api')
            auth_token = device_info.get('auth_token')
            connector = WiFiDeviceConnector(device_ip, api_endpoint, auth_token)
            
        elif device_type == 'bluetooth':
            device_address = device_info.get('address')
            device_name = device_info.get('name')
            connector = BluetoothDeviceConnector(device_address, device_name)
            
        if connector and connector.connect():
            self.device_connectors[device_id] = connector
            self.devices[device_id] = device_info
            self.save_configuration()
            print(f"Successfully connected to device: {device_id}")
            return True
        else:
            print(f"Failed to connect to device: {device_id}")
            return False
            
    def control_device(self, device_id, action, parameters=None):
        """Send control command to device."""
        connector = self.device_connectors.get(device_id)
        if not connector:
            print(f"Device {device_id} not connected")
            return False
            
        success = connector.send_command(action, parameters)
        if success:
            print(f"Command '{action}' sent to device {device_id}")
            # Log action
            self._log_device_action(device_id, action, parameters)
        else:
            print(f"Failed to send command '{action}' to device {device_id}")
            
        return success
        
    def control_group(self, group_name, action, parameters=None):
        """Control a group of devices."""
        if group_name not in self.device_groups:
            print(f"Device group '{group_name}' not found")
            return False
            
        device_ids = self.device_groups[group_name]
        results = []
        
        for device_id in device_ids:
            result = self.control_device(device_id, action, parameters)
            results.append(result)
            
        success_count = sum(results)
        print(f"Group '{group_name}': {success_count}/{len(device_ids)} devices controlled successfully")
        return success_count > 0
        
    def create_group(self, group_name, device_ids):
        """Create a device group."""
        self.device_groups[group_name] = device_ids
        self.save_configuration()
        print(f"Created device group '{group_name}' with {len(device_ids)} devices")
        
    def process_voice_command(self, command_text):
        """Process natural language voice commands."""
        command_lower = command_text.lower()
        
        # Parse common voice commands
        if "turn on" in command_lower:
            if "lights" in command_lower:
                return self.control_group("lights", "turn_on")
            elif "all" in command_lower:
                return self.control_group("all", "turn_on")
                
        elif "turn off" in command_lower:
            if "lights" in command_lower:
                return self.control_group("lights", "turn_off")
            elif "all" in command_lower:
                return self.control_group("all", "turn_off")
                
        elif "dim" in command_lower or "brightness" in command_lower:
            # Extract brightness level
            import re
            numbers = re.findall(r'\d+', command_text)
            if numbers:
                brightness = min(int(numbers[0]), 100)
                return self.control_group("lights", "set_brightness", {"brightness": brightness * 2.55})
                
        elif "color" in command_lower:
            # Extract color
            colors = {
                "red": 0, "green": 25500, "blue": 46920,
                "yellow": 12750, "purple": 56100, "orange": 8618
            }
            for color_name, hue_value in colors.items():
                if color_name in command_lower:
                    return self.control_group("lights", "turn_on", {"color": hue_value})
                    
        print(f"Could not understand command: {command_text}")
        return False
        
    def add_automation_rule(self, rule_name, trigger, action, conditions=None):
        """Add automation rule."""
        rule = {
            'trigger': trigger,
            'action': action,
            'conditions': conditions or {},
            'enabled': True,
            'created_at': datetime.now().isoformat()
        }
        
        self.automation_rules[rule_name] = rule
        self.save_configuration()
        print(f"Added automation rule: {rule_name}")
        
    def start_monitoring(self):
        """Start device monitoring thread."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        threading.Thread(target=self._monitoring_loop, daemon=True).start()
        
    def stop_monitoring(self):
        """Stop device monitoring."""
        self.is_monitoring = False
        
    def _monitoring_loop(self):
        """Monitor device status and execute automation rules."""
        while self.is_monitoring:
            try:
                # Check device status
                for device_id, connector in self.device_connectors.items():
                    status = connector.get_status()
                    if status:
                        self.devices[device_id]['last_status'] = status
                        self.devices[device_id]['last_check'] = datetime.now().isoformat()
                        
                # Execute automation rules
                self._execute_automation_rules()
                
                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                
    def _execute_automation_rules(self):
        """Execute automation rules based on triggers."""
        current_time = datetime.now()
        
        for rule_name, rule in self.automation_rules.items():
            if not rule.get('enabled', True):
                continue
                
            trigger = rule['trigger']
            
            # Time-based triggers
            if trigger['type'] == 'time':
                trigger_time = trigger.get('time')
                if trigger_time and current_time.strftime('%H:%M') == trigger_time:
                    self._execute_action(rule['action'])
                    
            # Device state triggers
            elif trigger['type'] == 'device_state':
                device_id = trigger.get('device_id')
                expected_state = trigger.get('state')
                
                if device_id in self.devices:
                    current_state = self.devices[device_id].get('last_status', {})
                    if self._check_state_condition(current_state, expected_state):
                        self._execute_action(rule['action'])
                        
    def _execute_action(self, action):
        """Execute automation action."""
        action_type = action.get('type')
        
        if action_type == 'device_control':
            device_id = action.get('device_id')
            command = action.get('command')
            parameters = action.get('parameters', {})
            self.control_device(device_id, command, parameters)
            
        elif action_type == 'group_control':
            group_name = action.get('group_name')
            command = action.get('command')
            parameters = action.get('parameters', {})
            self.control_group(group_name, command, parameters)
            
    def _check_state_condition(self, current_state, expected_state):
        """Check if current state matches expected state."""
        # Simple state comparison - can be enhanced
        return current_state.get('state') == expected_state
        
    def _log_device_action(self, device_id, action, parameters):
        """Log device action for history/analytics."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'device_id': device_id,
            'action': action,
            'parameters': parameters
        }
        # Could save to log file or database
        
    def get_device_status(self, device_id=None):
        """Get status of specific device or all devices."""
        if device_id:
            return self.devices.get(device_id, {})
        else:
            return self.devices
            
    def get_system_status(self):
        """Get overall smart home system status."""
        total_devices = len(self.devices)
        connected_devices = len(self.device_connectors)
        automation_rules = len(self.automation_rules)
        
        return {
            'total_devices': total_devices,
            'connected_devices': connected_devices,
            'automation_rules': automation_rules,
            'monitoring_active': self.is_monitoring,
            'last_updated': datetime.now().isoformat()
        }
