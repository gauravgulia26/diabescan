import os, sys
from pathlib import Path
from datetime import datetime

def GetCurrentDateTime():
    return f"{datetime.now().strftime('%d:%m:%Y-%H:%M:%S')}"

TIMESTAMP = GetCurrentDateTime()

CURRENT_PATH = os.getcwd()
LOGS_DIRECTORY_NAME = 'logs'
LOGS_FILE_NAME = f'{TIMESTAMP}.log'
