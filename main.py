from src.components.DataIngestion import DataIngestion
from src.components.DataTransformation import TransformData
from src.components.ModelTrainer import ModelTrainer
from src.components.ModelTuner import GetBestModel, optimize_svm, save_best_model
from src.logger.custom_logger import logger
from src.constants import MODEL_DIR_PATH, PREPROCESSOR_DIR_PATH
from src.components.Prediction import SVMPredictor
from src.utils.common import DataUtils

if __name__ == "__main__":
    try:
        ingest = DataIngestion().ingest_data()
        train_df, test_df = TransformData().InitiateTransformation()
        X_train = train_df.drop(columns=train_df.columns[-1])
        y_train = train_df[train_df.columns[-1]]
        X_test = test_df.drop(columns=train_df.columns[-1])
        y_test = test_df[train_df.columns[-1]]
        trainer = ModelTrainer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        trainer.train_and_evaluate()
        logger.info("Finding the best model...")
        best_model = GetBestModel()
        logger.info(f"Best model found: {best_model.__class__.__name__}")
        logger.info("Starting hyperparameter optimization...")
        study = optimize_svm(best_model, n_trials=100, storage_path=None)
        save_best_model(study, best_model, model_path=MODEL_DIR_PATH)
        predictor = SVMPredictor(model_path=MODEL_DIR_PATH)
        loader = DataUtils().load_preprocessor_pipeline(load_path=PREPROCESSOR_DIR_PATH)

        sample_input = [
            [
                0.5,
                1.2,
                3.3,
                4.4,
                5.5,
                6.6,
                7.7,
                8.8,
                9.9,
                10.1,
                11.2,
                12.3,
                13.4,
                14.5,
                15.6,
                16.7,
                0,
            ]
        ]

        try:
            prediction = predictor.predict(sample_input)
            logger.info(f"Predicted class: {prediction}")
            if int(prediction) == 1:
                logger.error("You have Diabetes.")
            else:
                logger.info("You are Healthy.")

        except Exception as e:
            logger.error(f"Prediction failed: {e}")

    except Exception as e:
        logger.error(e)
    else:
        logger.info("All Process Completed Successfully")
