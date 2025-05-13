import os, sys
from pathlib import Path
from datetime import datetime


def GetCurrentDateTime():
    return f"{datetime.now().strftime('%d:%m:%Y-%H:%M:%S')}"


TIMESTAMP = GetCurrentDateTime()

CURRENT_PATH = os.getcwd()
LOGS_DIRECTORY_NAME = "logs"
LOGS_FILE_NAME = f"{TIMESTAMP}.log"

LOGS_DIRECTORY_PATH = Path(f"{CURRENT_PATH}/{LOGS_DIRECTORY_NAME}")

CFG_FILE_NAME = "config.yaml"
CFG_FILE_PATH = Path(os.path.join(CURRENT_PATH, "config", CFG_FILE_NAME))

RAW_DATA_FILE_NAME = "raw.csv"
RAW_DATA_FILE_DIR = Path(os.path.join(CURRENT_PATH, "data/raw", RAW_DATA_FILE_NAME))

PROCESSED_FILE_NAME = "processed.csv"
PROCESSED_FILE_DIR = Path(os.path.join(CURRENT_PATH, "data/processed", PROCESSED_FILE_NAME))

TARGET_FEATURE = 'class'

CACHE_DIR_PATH = Path(os.path.join(CURRENT_PATH, "cachedir"))
