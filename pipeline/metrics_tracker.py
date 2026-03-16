import os
import json
from datetime import datetime

class MetricsTracker:
    """
    Tracks metrics for each pipeline stage in a structured way.
    Saves metrics as JSON in the experiment run folder.
    """

    def __init__(self, run_dir: str):
        """
        Args:
            run_dir (str): Path to the timestamped experiment run folder.
        """
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.run_dir, "metrics.json")

        # Initialize metrics dictionary
        self.metrics = {
            "experiment_start": self._current_time(),
            "stages": {}
        }

    def _current_time(self) -> str:
        return datetime.now().isoformat()

    def log_event(self, stage_name: str, count: int = None, error: str = None):
        """
        Log an event for a pipeline stage.

        Args:
            stage_name (str): Name of the stage (e.g., 'notice_collection_start').
            count (int, optional): Number of items processed at this stage.
            error (str, optional): Error message if stage failed.
        """
        now = self._current_time()

        # Initialize stage entry if not exists
        if stage_name not in self.metrics["stages"]:
            self.metrics["stages"][stage_name] = {}

        stage_entry = self.metrics["stages"][stage_name]
        stage_entry["timestamp"] = now
        if count is not None:
            stage_entry["count"] = count
        if error:
            stage_entry["error"] = error

        # Save metrics after each log
        self._save_metrics()

    def save_metrics(self):
        """
        Finalize experiment metrics by adding experiment_end timestamp.
        """
        self.metrics["experiment_end"] = self._current_time()
        self._save_metrics()

    def _save_metrics(self):
        """
        Save metrics dictionary to JSON file in run folder.
        """
        try:
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(self.metrics, f, indent=4)
        except Exception as e:
            print(f"[MetricsTracker] Error saving metrics: {e}")