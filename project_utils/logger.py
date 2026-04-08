# utils/logger.py
import logging
import os
from typing import Optional

class PipelineLogger:
    """
    A structured logger for pipeline applications.
    
    Provides a wrapper around Python's logging module with validation,
    handler deduplication, and a simplified interface.
    
    Args:
        name: Logger name, used to identify the source of log messages.
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to "INFO".
        log_file: Optional path to log file. If provided, logs will be written
            to this file in addition to console.
        formatter: Optional logging.Formatter instance. If not provided,
            a default formatter is used.
    
    Raises:
        ValueError: If an invalid log level is provided.
    """
    
    # Valid logging levels for validation
    _VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    
    def __init__(self, name: str, level: str = "INFO", 
                 log_file: Optional[str] = None,
                 formatter: Optional[logging.Formatter] = None):
        # Validate log level
        level_upper = level.upper()
        if level_upper not in self._VALID_LEVELS:
            raise ValueError(
                f"Invalid log level '{level}'. "
                f"Valid levels are: {', '.join(sorted(self._VALID_LEVELS))}"
            )
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level_upper))
        
        # Use provided formatter or create default
        if formatter is None:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Check for existing handlers to avoid duplication
        has_console_handler = any(
            isinstance(h, logging.StreamHandler) and 
            not isinstance(h, logging.FileHandler)
            for h in self.logger.handlers
        )
        
        if not has_console_handler:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            # Check for existing file handler for this specific file
            has_file_handler = any(
                isinstance(h, logging.FileHandler) and 
                h.baseFilename == os.path.abspath(log_file)
                for h in self.logger.handlers
            )
            
            if not has_file_handler:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
    
    def get_level(self) -> str:
        """Get the current logging level as a string."""
        return logging.getLevelName(self.logger.level)
    
    def debug(self, message: str):
        """Log a message with DEBUG level."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log a message with INFO level."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log a message with WARNING level."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log a message with ERROR level."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log a message with CRITICAL level."""
        self.logger.critical(message)
    
    def exception(self, message: str):
        """
        Log a message with ERROR level and include exception traceback.
        
        This method should be called from within an exception handler.
        """
        self.logger.exception(message)
