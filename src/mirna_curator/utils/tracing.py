import json
import logging
import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

class EventFormatter(logging.Formatter):
    """Custom formatter that outputs JSON formatted events"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Extract the event dictionary from the extra fields
        event_dict = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": record.levelname,
        }
        
        # Add any extra attributes that were passed
        if hasattr(record, "event_data"):
            event_dict.update(record.event_data)
            
        return json.dumps(event_dict)

class EventLogger:
    """Logger for event sourcing pattern that writes to NDJSON files"""
    
    def __init__(
        self,
        output_dir: str = "curation_traces",
        filename_prefix: str = "flowchart_events",
        encoding: str = "utf-8"
    ):
        self.output_dir = Path(output_dir)
        self.filename_prefix = filename_prefix
        self.encoding = encoding
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger("event_sourcing")
        self.logger.setLevel(logging.INFO)
        
        ## Use a run instance uid
        self.run_id = str(uuid.uuid4())
        self.paper_id = None

        # Create handler that writes to today's file
        handler = logging.FileHandler(
            self._get_current_filename(),
            mode='a',
            encoding=self.encoding,
            delay=False  # Open the file immediately
        )
        # Ensure immediate flush after each write
        handler.flush = True
        handler.setFormatter(EventFormatter())
        self.logger.addHandler(handler)
    
    def _get_current_filename(self) -> str:
        """Generate filename for current day's events"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return str(self.output_dir / f"{self.filename_prefix}_{date_str}.ndjson")
    
    def set_paper_id(self, paper_id: str) -> None:
        """
        Set the paper ID, which will be included alongside the run ID in all events.
        """
        self.paper_id = paper_id
    
    def log_event(
        self,
        event_type: str,
        **event_data: Any
    ) -> bool:
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
            **event_data
        }
        print(event_dict)
        try:
            # Log using standard logging with extra data
            self.logger.info("", extra={"event_data": event_dict})
            return True
        except Exception as e:
            # If primary logging fails, attempt emergency backup
            try:
                emergency_file = Path(f"emergency_events_{self.run_id}.ndjson")
                with emergency_file.open('a', encoding=self.encoding) as f:
                    json.dump(event_dict, f)
                    f.write('\n')
                return True
            except:
                return False
