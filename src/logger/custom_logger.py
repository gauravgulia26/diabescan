import os
import logging
from src.config import LOGS_DIRECTORY_PATH, LOGS_FILE_NAME


class CreateLogger:
    def __init__(self):
        pass

    def initialise(self):

        os.makedirs(LOGS_DIRECTORY_PATH, exist_ok=True)

        # Configure the logger
        logging.basicConfig(
            filemode="w",  # Overwrites the file each time
            filename=os.path.join(LOGS_DIRECTORY_PATH, LOGS_FILE_NAME),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        logger = logging.getLogger(__name__)

        return logger
