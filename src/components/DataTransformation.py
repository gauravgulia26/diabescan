from src.logger.custom_logger import logger
from src.exceptions.CustomException import CustomException
from src.components.DataIngestion import DataIngestion
from src.utils.common import DataUtils
from src.constants import (
    TRAIN_PROCESSED_FILE_DIR,
    TEST_PROCESSED_FILE_DIR,
    RAW_DATA_FILE_DIR,
    TRAIN_PROCESSED_FILE_NAME,
    TEST_PROCESSED_FILE_NAME,
    PREPROCESSOR_DIR_PATH,
    ENCODER_DIR_PATH,
    TRF_DF_DIR,
)
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, validate_arguments, ValidationError
from typing import List, Any
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn


class TransformData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    loader: DataUtils = Field(description="Loader to load Utilities Function", default=DataUtils())

    def __init__(self, **data):
        super().__init__(**data)
        # Set up MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("diabetes_data_transformation")

    def read_data(self):
        """Function to load the df and split into x and y

        Returns:
            tuple: tuple[DataFrame,x,y]
        """
        logger.info("Reading and Splitting the data")
        df, x, y = self.loader.load_and_split_df(RAW_DATA_FILE_DIR)
        return df, x, y

    def CreatePipeline(self):
        df, x, y = self.read_data()
        logger.info("Pipeline Creation Started")
        ordinal_categories = (
            df.select_dtypes(exclude="int64").drop(columns=["class", "gender"]).columns.tolist()
        )
        ord_cat = [["Yes", "No"]] * len(ordinal_categories)
        cat_ordinal = Pipeline(steps=[("Ordinal Pipeline", OrdinalEncoder(categories=ord_cat))])
        cat_ohe = Pipeline(
            steps=[("One Hot Encoder", OneHotEncoder(drop="first", sparse_output=False))]
        )

        return cat_ordinal, cat_ohe, ordinal_categories

    def CreateColumnTransformer(self):
        cat_ord, cat_ohe, ordinal_categories = self.CreatePipeline()
        logger.info("Column Transformer Creation Started")
        trf = ColumnTransformer(
            transformers=[
                ("Categorical Ordinal Transformation", cat_ord, ordinal_categories),
                ("One Hot Transformation", cat_ohe, ["gender"]),
            ],
            remainder="passthrough",
        )
        return trf

    def ApplyTransformation(self):
        trf = self.CreateColumnTransformer()
        return trf

    def CreateDataframe(self):
        trf = self.ApplyTransformation()
        df, x, y = self.read_data()
        data = trf.fit_transform(df)
        Encoder = trf.named_transformers_["One Hot Transformation"]
        joblib.dump(Encoder, ENCODER_DIR_PATH)
        logger.info(f"Encoder Saved Successfully at {ENCODER_DIR_PATH}")
        labels = trf.get_feature_names_out()
        logger.info("Transformation Applied Successfully !!")
        trf_df = pd.DataFrame(data=data, columns=labels)
        le = LabelEncoder()
        trf_df[trf_df.columns[-1]] = le.fit_transform(trf_df[trf_df.columns[-1]])
        # trf_df["remainder__class"] = le.fit_transform(trf_df["remainder__class"])
        logger.info("Processed Dataframe Created Successfully")
        trf_df = trf_df.drop(columns=trf_df.columns[-3])
        trf_df.to_csv(TRF_DF_DIR, index=False)
        return trf_df

    def SplitData(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        trf_df = self.CreateDataframe()
        X = trf_df.drop(columns=trf_df.columns[-1])
        y = trf_df[trf_df.columns[-1]]

        # Validate that the target variable is discrete
        if not pd.api.types.is_integer_dtype(y):
            raise ValueError("Target variable must be discrete. Ensure proper encoding.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        logger.info("Data Successfully Splitted into Training and Testing Set")
        return X_train, X_test, y_train, y_test

    def ApplyScaling(self, preprocessor_path: Path = PREPROCESSOR_DIR_PATH):
        X_train, X_test, y_train, y_test = self.SplitData()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("Scaling Applied Successfully")
        joblib.dump(scaler, preprocessor_path)
        logger.info(f"Standard Scaler saved successfully at {preprocessor_path}")

        return X_train, X_test, y_train, y_test

    def CreateTrainTestSet(self):
        X_train, X_test, y_train, y_test = self.ApplyScaling()
        # training_set = pd.DataFrame(X_train)  # Exclude dependent feature
        # test_set = pd.DataFrame(X_test)  # Exclude dependent feature
        logger.info("Train Test df created Successfully !!")

        return X_train, X_test, y_train, y_test

    def SaveProcessedData(self):
        X_train, X_test, y_train, y_test = self.CreateTrainTestSet()
        os.makedirs(TRAIN_PROCESSED_FILE_DIR, exist_ok=True)
        os.makedirs(TEST_PROCESSED_FILE_DIR, exist_ok=True)
        # try:
        #     train_df.to_csv(
        #         os.path.join(TRAIN_PROCESSED_FILE_DIR, TRAIN_PROCESSED_FILE_NAME), index=False
        #     )
        #     test_df.to_csv(
        #         os.path.join(TEST_PROCESSED_FILE_DIR, TEST_PROCESSED_FILE_NAME), index=False
        #     )
        # except Exception as e:
        #     logger.error(e)
        # logger.info("Training Set and Testing set has been saved Successfully")

        return X_train, X_test, y_train, y_test

    def InitiateTransformation(self):
        with mlflow.start_run(run_name="data_transformation"):
            try:
                X_train, X_test, y_train, y_test = self.SaveProcessedData()

                # Log dataset shapes
                mlflow.log_params(
                    {
                        "train_samples": X_train.shape[0],
                        "test_samples": X_test.shape[0],
                        "n_features": X_train.shape[1],
                        "test_size": 0.33,
                        "random_state": 42,
                    }
                )

                # Log feature names
                mlflow.log_dict(
                    {
                        "feature_names": (
                            list(X_train.columns)
                            if hasattr(X_train, "columns")
                            else [f"feature_{i}" for i in range(X_train.shape[1])]
                        )
                    },
                    "feature_names.json",
                )

                # Log data statistics
                train_stats = {
                    "train_mean": (
                        np.mean(X_train).tolist()
                        if isinstance(X_train, np.ndarray)
                        else X_train.mean().tolist()
                    ),
                    "train_std": (
                        np.std(X_train).tolist()
                        if isinstance(X_train, np.ndarray)
                        else X_train.std().tolist()
                    ),
                    "train_class_distribution": np.bincount(y_train).tolist(),
                }
                mlflow.log_dict(train_stats, "train_statistics.json")

                # Log transformation artifacts
                mlflow.log_artifact(ENCODER_DIR_PATH, "encoders")
                mlflow.log_artifact(PREPROCESSOR_DIR_PATH, "preprocessor")

            except Exception as e:
                logger.error(e)
                mlflow.log_param("error", str(e))
                mlflow.log_param("status", "failed")
                raise e
            else:
                logger.info("Transformation Finished Successfully !!")
                mlflow.log_param("status", "success")
                return X_train, X_test, y_train, y_test


obj = TransformData().InitiateTransformation()
