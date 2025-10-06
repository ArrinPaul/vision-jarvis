import json
import os

class AutomationEngine:
    """
    Task Automation and Routine Engine for custom routines triggered by voice or gesture. Routines stored in JSON file.
    """
    def __init__(self, routine_path="routines.json"):
        self.routine_path = routine_path
        self.routines = self._load_routines()

    def _load_routines(self):
        if os.path.exists(self.routine_path):
            with open(self.routine_path, "r") as f:
                return json.load(f)
        return {}

    def _save_routines(self):
        with open(self.routine_path, "w") as f:
            json.dump(self.routines, f, indent=2)

    def create_routine(self, routine_info):
        """Create a new automation routine."""
        routine_id = routine_info.get("id")
        if not routine_id:
            print("Routine info must include 'id'.")
            return
        self.routines[routine_id] = routine_info
        self._save_routines()
        print(f"Routine created: {routine_id}")

    def run_routine(self, routine_id):
        """Execute a routine by ID (prints actions for now)."""
        routine = self.routines.get(routine_id)
        if not routine:
            print(f"Routine {routine_id} not found.")
            return
        actions = routine.get("actions", [])
        print(f"Running routine {routine_id}:")
        for action in actions:
            print(f"- {action}")
