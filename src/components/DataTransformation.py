from src.logger.custom_logger import logger
from src.exceptions.CustomException import CustomException
from src.components.DataIngestion import DataIngestion
from src.utils.common import DataUtils
from src.constants import PROCESSED_FILE_DIR, RAW_DATA_FILE_DIR
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Any
import os


class TransformData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    file_url: Path = Field(
        description="Path of the raw file after ingestion, must be a POSIX Path",
        default=RAW_DATA_FILE_DIR,
    )
    loader: DataUtils = Field(description="Loader to load Utilities Function", default=DataUtils())

    def read_data(self):
        """Function to load the df and split into x and y

        Returns:
            tuple: tuple[DataFrame,x,y]
        """
        logger.info("Reading and Splitting the data")
        df, x, y = self.loader.load_and_split_df(self.file_url)
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

    def InitiateTransformation(self):
        trf = self.CreateColumnTransformer()
        df,x,y = self.read_data()
        data = trf.fit_transform(df)
        labels = trf.get_feature_names_out()
        logger.info('Transformation Applied Successfully !!')
        trf_df = pd.DataFrame(data, columns=labels)
        logger.info('Processed Data Saved Successfully !!')
        le = LabelEncoder()
        trf_df["remainder__class"] = le.fit_transform(trf_df["remainder__class"])
        try:
            trf_df.to_csv(PROCESSED_FILE_DIR)
        except Exception as e:
            logger.error(e)
        else:
            X = trf_df.drop(columns=trf_df.columns[-1])
            y = trf_df["remainder__class"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            scaler = StandardScaler()
            logger.info('Applying Feature Scaling')
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logger.info('Transformation Finished Successfully !!')
