import os
import logging
import sys
import traceback
import inspect
import coloredlogs
from datetime import datetime
from src.constants import *


class CustomLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CustomLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Set up and configure the logger with both file and console handlers."""
        os.makedirs(LOGS_DIRECTORY_PATH, exist_ok=True)

        # Create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers if any (avoids duplicate handlers on re-initialization)
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)

        # Create formatters with detailed information including line and module
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
        )

        # File handler - append mode
        file_handler = logging.FileHandler(
            os.path.join(LOGS_DIRECTORY_PATH, LOGS_FILE_NAME),
            mode="a",  # Append mode instead of overwrite
        )
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        # Console handler with colored logs
        console_format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
        coloredlogs.install(
            level=logging.INFO,
            logger=logger,
            fmt=console_format
        )

        return logger

    def get_logger(self):
        """Get the configured logger instance."""
        return self.logger
    
    def error(self, message, exit_code=1):
        """
        Log an error message and exit the application.
        
        Args:
            message: The error message to log
            exit_code: The exit code to use when terminating the program
        """
        # Get call frame information
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        
        # Format error message with stack trace
        error_details = f"{message}\nOccurred at: {filename}:{lineno} in {func_name}()"
        
        # Log the error with detailed information
        self.logger.error(error_details)
        
        # Print stack trace to help with debugging
        traceback.print_stack()
        
        # Exit the program with the specified exit code
        print(f"\nExecution terminated due to error. Check logs for details.")
        sys.exit(exit_code)
    
    def critical(self, message, exit_code=1):
        """Alias for error method with critical level"""
        self.error(message, exit_code)


# Create a singleton instance
logger_instance = CustomLogger()

# Convenience functions that provide the enhanced logging capabilities
def get_logger():
    """Get the configured logger instance."""
    return logger_instance.get_logger()

def error(message, exit_code=1):
    """Log an error and terminate execution."""
    logger_instance.error(message, exit_code)

def critical(message, exit_code=1):
    """Log a critical error and terminate execution."""
    logger_instance.error(message, exit_code)

# For backward compatibility
logger = get_logger()