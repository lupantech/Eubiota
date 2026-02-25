"""
Agent Logger - Unified logging system for agent execution.
Imported from Scientist framework.
"""

import json
import logging
from typing import Dict, Any, Optional


def setup_logger(logger_name: str, logging_path: str = None):
    """
    Setup structured logger for agents with file and console handlers.

    Args:
        logger_name: Name for the logger (e.g., f"Agent_{id(self)}")
        logging_path: Optional path for log file

    Returns:
        Logger instance or None if setup failed
    """
    raw_logger = logging.getLogger(logger_name)
    raw_logger.setLevel(logging.INFO)

    # Clear existing handlers (close them first to release file handles)
    for handler in raw_logger.handlers[:]:
        handler.close()
        raw_logger.removeHandler(handler)
    raw_logger.handlers.clear()
    handlers_added = False

    # Add file handler if path provided
    if logging_path:
        import os
        log_dir = os.path.dirname(logging_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Use unbuffered file handler for real-time logging
        class FlushFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()  # Force flush after every log

        file_handler = FlushFileHandler(logging_path, mode='w')
        file_handler.setLevel(logging.INFO)

        # Create formatter with timestamp for file
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        raw_logger.addHandler(file_handler)
        handlers_added = True

    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create console formatter (without timestamp for cleaner output)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    raw_logger.addHandler(console_handler)
    handlers_added = True

    # Only set up logger if at least one handler was added
    if handlers_added:
        raw_logger.propagate = False
        # Return Logger instance
        return Logger(raw_logger)
    else:
        return None


class Logger:
    """Unified logging system for agent execution."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger

    def safe_json_dumps(self, obj, max_length: int = 1000000, **kwargs):
        """Safely serialize an object to JSON, handling non-serializable objects."""
        try:
            # Simple serialization without truncation
            return json.dumps(obj, **kwargs, default=str)
        except Exception as e:
            return f"<Unable to serialize: {str(e)}>"

    def log(self, message: str, data: Dict[str, Any] = None, level: str = "info"):
        """Unified logging method."""
        if not self.logger:
            return

        if data:
            full_message = f"{message} | {self.safe_json_dumps(data, ensure_ascii=False)}"
        else:
            full_message = message

        if level == "error":
            self.logger.error(full_message)
        elif level == "warning":
            self.logger.warning(full_message)
        else:
            self.logger.info(full_message)

    def info(self, message: str):
        """Standard info logging method."""
        if self.logger:
            self.logger.info(message)

    def error(self, message: str):
        """Standard error logging method."""
        if self.logger:
            self.logger.error(message)

    def warning(self, message: str):
        """Standard warning logging method."""
        if self.logger:
            self.logger.warning(message)

    def debug(self, message: str):
        """Standard debug logging method."""
        if self.logger:
            self.logger.debug(message)

    def critical(self, message: str):
        """Standard critical logging method."""
        if self.logger:
            self.logger.critical(message)

    def log_header(self, header: str, separator: str = "=", width: int = 80):
        """Log a title."""
        if not self.logger:
            return
        self.logger.info(separator * width)
        self.logger.info(f"## {header}")
        self.logger.info(separator * width)

    def log_section(self, title: str = None, content: Any = None, separator: str = "-", width: int = 60):
        """Log a section with content."""
        if not self.logger:
            return

        # Log title
        self.logger.info("")
        self.logger.info(separator * width)
        self.logger.info(f"==> {title}")
        self.logger.info(separator * width)

        # Log content
        if content:
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, str):
                        if len(value) > 100000:
                            self.logger.info(f"- {key}: {value[:100000]} ... [TRUNCATED]")
                        else:
                            self.logger.info(f"- {key}: {value}")
                    elif isinstance(value, list):
                        self.logger.info(f"- {key}: List with {len(value)} items")
                        if len(value) > 100:
                            self.logger.info(f"  - {value[:100]} ... [TRUNCATED]")
                        else:
                            self.logger.info(f"  - {value}")
                    elif isinstance(value, dict):
                        output_str = self.safe_json_dumps(value, indent=4, ensure_ascii=False)
                        if len(output_str) > 100000:
                            self.logger.info(f"- {key}: {output_str[:100000]} ... [TRUNCATED]")
                        else:
                            self.logger.info(f"- {key}: {output_str}")
                    else:
                        self.logger.info(f"- {key}: {value}")
            else:
                content_str = str(content)
                if len(content_str) > 100000:
                    self.logger.info(content_str[:100000] + "... [TRUNCATED]")
                else:
                    self.logger.info(content_str)

        self.logger.info(separator * width)
        self.logger.info("")
