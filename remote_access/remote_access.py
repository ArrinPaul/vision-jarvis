from flask import Flask, request, jsonify
import threading

class RemoteAccess:
    """
    Remote Access via mobile/web dashboard for control and monitoring using Flask web server.
    """
    def __init__(self, host="0.0.0.0", port=5000):
        self.app = Flask(__name__)
        self.status = "Idle"
        self.host = host
        self.port = port
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/status", methods=["GET"])
        def get_status():
            return jsonify({"status": self.status})

        @self.app.route("/control", methods=["POST"])
        def control():
            data = request.json
            action = data.get("action")
            self.status = f"Action received: {action}"
            return jsonify({"result": "success", "action": action})

    def connect_dashboard(self):
        """Start the Flask web dashboard in a separate thread."""
        threading.Thread(target=self.app.run, kwargs={"host": self.host, "port": self.port}, daemon=True).start()
        print(f"Dashboard running at http://{self.host}:{self.port}")

    def send_status_update(self, status):
        """Update the status shown on the dashboard."""
        self.status = status
