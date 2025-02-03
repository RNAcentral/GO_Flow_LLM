import json
import logging
import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import uuid


class EventLogger:
    """Logger for event sourcing pattern that writes to NDJSON files"""

    def __init__(
        self,
        output_dir: str = "curation_traces",
        filename_prefix: str = "flowchart_events",
        encoding: str = "utf-8",
    ):
        self.output_dir = Path(output_dir)
        self.filename_prefix = filename_prefix
        self.encoding = encoding

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ## Use a run instance uid
        self.run_id = str(uuid.uuid4())
        self.paper_id = None

    def _get_current_filename(self) -> str:
        """Generate filename for current day's events"""
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        return str(self.output_dir / f"{self.filename_prefix}_{date_str}.ndjson")

    def set_paper_id(self, paper_id: str) -> None:
        """
        Set the paper ID, which will be included alongside the run ID in all events.
        """
        self.paper_id = paper_id

    def set_model_name(self, model_name: str) -> None:
        """
        Set the model name used in this run
        """
        self.model_id = model_name

    def log_event(self, event_type: str, **event_data: Any) -> bool:
        """
        Log an event with the given type and data.
        Returns True if logging was successful, False otherwise.
        """
        """
        Log an event with the given type and data.
        
        Args:
            event_type: Type of the event (e.g., "curation_started")
            **event_data: Additional event data as keyword arguments
        """
        event_dict = {
            "type": event_type,
            "run_id": self.run_id,
            "paper_id": self.paper_id,
            "model_id": self.model_id,
            "date" : datetime.datetime.now().strftime("%Y-%m-%d"),
            **event_data,
        }
        with open(self._get_current_filename(), "a", encoding="utf-8") as f:
            json.dump(event_dict, f)
            f.write("\n")

# Create singleton object when this module is imported
curation_tracer = EventLogger()