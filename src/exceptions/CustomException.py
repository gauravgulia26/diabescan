from src.logger.custom_logger import CreateLogger
import sys
import traceback


class CustomException(Exception):
    def __init__(self, error_message):
        self.logger = CreateLogger().initialise()
        super().__init__(error_message)
        self.error_message = error_message

    def log_exception(self):
        # Extract the current exception information
        exc_type, exc_value, exc_traceback = traceback.sys.exc_info()
        if exc_traceback:
            #* This will work only when try except block
            # Format the traceback to get detailed information
            tb_details = traceback.extract_tb(exc_traceback)
            for frame in tb_details:
                file_name = frame.filename
                line_number = frame.lineno
                function_name = frame.name
                self.logger.error(
                    f"Exception in {function_name} at {file_name}, line {line_number}: {self.error_message}"
                )
        else:
            # * this will work simply
            self.logger.error(self.error_message)