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

TRAIN_PROCESSED_FILE_NAME = "train.csv"
TEST_PROCESSED_FILE_NAME = "test.csv"
TRAIN_PROCESSED_FILE_DIR = Path(
    os.path.join(CURRENT_PATH, "data/processed/train")
)
TEST_PROCESSED_FILE_DIR = Path(
    os.path.join(CURRENT_PATH, "data/processed/test")
)

TARGET_FEATURE = "class"

RESULT_DIR = 'reports'
RESULT_DIR_PATH = Path(os.path.join(CURRENT_PATH,RESULT_DIR))

MODEL_NAME = 'BestModel.pkl'
MODEL_DIR = 'models'
MODEL_DIR_PATH = Path(os.path.join(CURRENT_PATH,MODEL_DIR,MODEL_NAME)) 

PREPROCESSOR_NAME = 'Preprocessor.joblib'
PREPROCESSOR_DIR_PATH = Path(os.path.join(CURRENT_PATH,MODEL_DIR,PREPROCESSOR_NAME))

ENCODER_NAME = 'OHE_Encoder.joblib'
ENCODER_DIR_PATH = Path(os.path.join(CURRENT_PATH,MODEL_DIR,ENCODER_NAME))

TRF_DF_NAME = 'data/external/trf_df.csv'
TRF_DF_DIR = Path(os.path.join(CURRENT_PATH,TRF_DF_NAME))