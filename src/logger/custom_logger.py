import os
import logging
from src.constants import *
import coloredlogs


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

        coloredlogs.install(level=logging.INFO)
        # Create logger
        logger = logging.getLogger(__name__)
        # logger.setLevel(logging.INFO)

        # Create formatters
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # File handler - append mode
        file_handler = logging.FileHandler(
            os.path.join(LOGS_DIRECTORY_PATH, LOGS_FILE_NAME),
            mode="a",  # Append mode instead of overwrite
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def get_logger(self):
        """Get the configured logger instance."""
        return self.logger



logger = CustomLogger().get_logger()