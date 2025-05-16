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
    PREPROCESSOR_DIR_PATH
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


class TransformData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    loader: DataUtils = Field(description="Loader to load Utilities Function", default=DataUtils())

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
        labels = trf.get_feature_names_out()
        logger.info("Transformation Applied Successfully !!")
        trf_df = pd.DataFrame(data=data, columns=labels)
        le = LabelEncoder()
        trf_df["remainder__class"] = le.fit_transform(trf_df["remainder__class"])
        logger.info("Processed Dataframe Created Successfully")
        return trf_df

    def SplitData(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        trf_df = self.CreateDataframe()
        X = trf_df.drop(columns=trf_df.columns[-1])
        y = trf_df[trf_df.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        logger.info("Data Successfully Splitted into Training and Testing Set")
        return X_train, X_test, y_train, y_test

    def ApplyScaling(self):
        X_train, X_test, y_train, y_test = self.SplitData()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("Scaling Applied Successfully")

        return X_train, X_test, y_train, y_test

    def CreateTrainTestSet(self):
        X_train, X_test, y_train, y_test = self.ApplyScaling()
        training_set = pd.DataFrame(np.hstack([X_train, y_train.values.reshape(-1, 1)]))
        test_set = pd.DataFrame(np.hstack([X_test, y_test.values.reshape(-1, 1)]))
        logger.info("Train Test df created Successfully !!")

        return training_set, test_set

    def SaveProcessedData(self):
        train_df, test_df = self.CreateTrainTestSet()
        os.makedirs(TRAIN_PROCESSED_FILE_DIR, exist_ok=True)
        os.makedirs(TEST_PROCESSED_FILE_DIR, exist_ok=True)
        try:
            train_df.to_csv(
                os.path.join(TRAIN_PROCESSED_FILE_DIR, TRAIN_PROCESSED_FILE_NAME), index=False
            )
            test_df.to_csv(
                os.path.join(TEST_PROCESSED_FILE_DIR, TEST_PROCESSED_FILE_NAME), index=False
            )
        except Exception as e:
            logger.error(e)
        logger.info("Training Set and Testing set has been saved Successfully")

        return train_df, test_df

    def InitiateTransformation(self):
        try:
            train_df, test_df = self.SaveProcessedData()
            self.get_preprocessor_pipeline()
        except Exception as e:
            logger.error(e)
        else:
            logger.info("Transformation Finished Successfully !!")
            return train_df, test_df

    def get_preprocessor_pipeline(self, save_path: Path = PREPROCESSOR_DIR_PATH):
        """Returns and optionally saves the fitted preprocessor pipeline that can be used for future predictions
        
        Args:
            save_path (str, optional): Path where to save the pipeline. Defaults to None.
            If not provided, pipeline will be saved in models/preprocessor.joblib
        
        Returns:
            Pipeline: A scikit-learn Pipeline containing all preprocessing steps
        """
        try:
            # Get the transformers
            trf = self.CreateColumnTransformer()
            df, _, _ = self.read_data()
            
            # Create a pipeline with column transformer and label encoder
            le = LabelEncoder()
            preprocessor = Pipeline([
                ('column_transformer', trf),
                ('scaler', StandardScaler())
            ])
            
            # Fit the pipeline on the data (excluding the target column)
            X = df.drop(columns=['class'])
            preprocessor.fit(X)
            
            # Save the pipeline
            from joblib import dump
            dump(preprocessor, save_path)
            logger.info(f"Preprocessor pipeline saved successfully at {save_path}")
            
            return preprocessor
            
        except Exception as e:
            logger.error(f"Error in creating/saving preprocessor pipeline: {str(e)}")
            raise CustomException(error_message=e)