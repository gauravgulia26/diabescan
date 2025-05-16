from src.components.DataIngestion import DataIngestion
from src.components.DataTransformation import TransformData
from src.components.ModelTrainer import ModelTrainer
from src.components.ModelTuner import GetBestModel, optimize_rf, save_best_model
from src.logger.custom_logger import logger
from src.constants import (
    MODEL_DIR_PATH,
    PREPROCESSOR_DIR_PATH,
    TRAIN_PROCESSED_FILE_DIR,
    TEST_PROCESSED_FILE_DIR,
)
from src.components.Prediction import SVMPredictor
from src.utils.common import DataUtils
import pandas as pd
from src.exceptions.CustomException import CustomException

if __name__ == "__main__":
    try:
        ingest = DataIngestion().ingest_data()
        X_train, X_test, y_train, y_test = TransformData().InitiateTransformation()
        trainer = ModelTrainer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        trainer.train_and_evaluate()
        logger.info("Finding the best model...")
        best_model = GetBestModel()
        logger.info(f"Best model found: {best_model.__class__.__name__}")
        logger.info("Starting hyperparameter optimization...")
        study = optimize_rf(best_model, n_trials=100, storage_path=None)
        save_best_model(study, best_model, model_path=MODEL_DIR_PATH)
        predictor = SVMPredictor(model_path=MODEL_DIR_PATH)
        loader = DataUtils().load_preprocessor_pipeline(load_path=PREPROCESSOR_DIR_PATH)

    except Exception as e:
        CustomException(error_message=e).log_exception()
    else:
        logger.info("All Process Completed Successfully")
