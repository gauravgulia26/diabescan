from src.logger.custom_logger import CreateLogger
from src.exceptions.CustomException import CustomException

logger = CreateLogger().initialise()

try:
    3 / 0
except ZeroDivisionError as e:
    err = CustomException(error_message=e).log_exception()
else:
    print('OKAY')