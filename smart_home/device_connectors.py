import requests
import json
import threading
import time
from datetime import datetime

class DeviceConnector:
    """
    Base class for smart device connectors.
    """
    def __init__(self, device_type, connection_info):
        self.device_type = device_type
        self.connection_info = connection_info
        self.is_connected = False
        self.last_status_check = None
        
    def connect(self):
        """Connect to the device."""
        raise NotImplementedError
        
    def disconnect(self):
        """Disconnect from the device."""
        raise NotImplementedError
        
    def send_command(self, command, parameters=None):
        """Send command to device."""
        raise NotImplementedError
        
    def get_status(self):
        """Get device status."""
        raise NotImplementedError

class PhilipsHueConnector(DeviceConnector):
    """
    Connector for Philips Hue smart lights.
    """
    def __init__(self, bridge_ip, username):
        super().__init__("philips_hue", {"bridge_ip": bridge_ip, "username": username})
        self.bridge_ip = bridge_ip
        self.username = username
        self.base_url = f"http://{bridge_ip}/api/{username}"
        
    def connect(self):
        """Connect to Hue bridge."""
        try:
            response = requests.get(f"{self.base_url}/lights", timeout=5)
            if response.status_code == 200:
                self.is_connected = True
                return True
        except Exception as e:
            print(f"Failed to connect to Hue bridge: {e}")
        return False
        
    def send_command(self, command, parameters=None):
        """Send command to Hue lights."""
        if not self.is_connected:
            return False
            
        if command == "turn_on":
            light_id = parameters.get("light_id", 1)
            data = {"on": True}
            if "brightness" in parameters:
                data["bri"] = parameters["brightness"]
            if "color" in parameters:
                data["hue"] = parameters["color"]
                
        elif command == "turn_off":
            light_id = parameters.get("light_id", 1)
            data = {"on": False}
            
        elif command == "set_brightness":
            light_id = parameters.get("light_id", 1)
            data = {"bri": parameters.get("brightness", 128)}
            
        else:
            return False
            
        try:
            url = f"{self.base_url}/lights/{light_id}/state"
            response = requests.put(url, json=data, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Hue command error: {e}")
            return False
            
    def get_status(self):
        """Get status of all Hue lights."""
        if not self.is_connected:
            return {}
            
        try:
            response = requests.get(f"{self.base_url}/lights", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Hue status error: {e}")
        return {}

class TPLinkKasaConnector(DeviceConnector):
    """
    Connector for TP-Link Kasa smart devices.
    """
    def __init__(self, device_ip):
        super().__init__("tplink_kasa", {"device_ip": device_ip})
        self.device_ip = device_ip
        
    def _send_request(self, command):
        """Send encrypted request to Kasa device."""
        # TP-Link uses a simple XOR encryption
        def encrypt(string):
            key = 171
            result = bytearray()
            for char in string:
                key = key ^ ord(char)
                result.append(key)
            return bytes(result)
            
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.device_ip, 9999))
            
            encrypted = encrypt(json.dumps(command))
            sock.send(len(encrypted).to_bytes(4, byteorder='big') + encrypted)
            
            # Read response
            response_length = int.from_bytes(sock.recv(4), byteorder='big')
            response_data = sock.recv(response_length)
            
            sock.close()
            return True
        except Exception as e:
            print(f"Kasa communication error: {e}")
            return False
            
    def connect(self):
        """Connect to Kasa device."""
        command = {"system": {"get_sysinfo": {}}}
        self.is_connected = self._send_request(command)
        return self.is_connected
        
    def send_command(self, command, parameters=None):
        """Send command to Kasa device."""
        if not self.is_connected:
            return False
            
        if command == "turn_on":
            kasa_command = {"system": {"set_relay_state": {"state": 1}}}
        elif command == "turn_off":
            kasa_command = {"system": {"set_relay_state": {"state": 0}}}
        else:
            return False
            
        return self._send_request(kasa_command)

class WiFiDeviceConnector(DeviceConnector):
    """
    Generic WiFi device connector for REST API based devices.
    """
    def __init__(self, device_ip, api_endpoint="/api", auth_token=None):
        super().__init__("wifi_generic", {"device_ip": device_ip, "api_endpoint": api_endpoint})
        self.device_ip = device_ip
        self.api_endpoint = api_endpoint
        self.auth_token = auth_token
        self.base_url = f"http://{device_ip}{api_endpoint}"
        
    def connect(self):
        """Connect to WiFi device."""
        try:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
                
            response = requests.get(f"{self.base_url}/status", headers=headers, timeout=5)
            self.is_connected = response.status_code == 200
            return self.is_connected
        except Exception as e:
            print(f"WiFi device connection error: {e}")
            return False
            
    def send_command(self, command, parameters=None):
        """Send command to WiFi device."""
        if not self.is_connected:
            return False
            
        try:
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
                
            payload = {"command": command}
            if parameters:
                payload.update(parameters)
                
            response = requests.post(f"{self.base_url}/command", 
                                   json=payload, headers=headers, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"WiFi device command error: {e}")
            return False

class BluetoothDeviceConnector(DeviceConnector):
    """
    Bluetooth device connector (requires bluetooth libraries).
    """
    def __init__(self, device_address, device_name=None):
        super().__init__("bluetooth", {"device_address": device_address, "device_name": device_name})
        self.device_address = device_address
        self.device_name = device_name
        
    def connect(self):
        """Connect to Bluetooth device."""
        try:
            # This would require bluetooth libraries like pybluez
            # For now, simulate connection
            print(f"Simulating Bluetooth connection to {self.device_address}")
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Bluetooth connection error: {e}")
            return False
            
    def send_command(self, command, parameters=None):
        """Send command to Bluetooth device."""
        if not self.is_connected:
            return False
            
        # Simulate command sending
        print(f"Sending Bluetooth command: {command} with parameters: {parameters}")
        return True