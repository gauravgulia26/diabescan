# This file is used to make some common Utility Functions similar to some extra Features that can be reused across different modules.

import os
import yaml
from src.logger.custom_logger import CustomLogger
import json
import joblib
from box import ConfigBox
from pathlib import Path
from typing import Any, Tuple, List
from box.exceptions import BoxValueError
from ucimlrepo import fetch_ucirepo
import pandas as pd
from pydantic import BaseModel, validate_arguments

logger = CustomLogger().get_logger()


class DataUtils(BaseModel):
    """A utility class for common data handling tasks with Pydantic type validation."""

    class Config:
        """Pydantic configuration to allow arbitrary types (e.g., Path, ConfigBox)."""
        arbitrary_types_allowed = True

    @staticmethod
    @validate_arguments
    def read_yaml(path_to_yaml: Path) -> ConfigBox:
        """Reads a YAML file and returns its content as a ConfigBox.

        Args:
            path_to_yaml: Path to the YAML file.

        Raises:
            ValueError: If the YAML file is empty.
            Exception: For other file reading errors.

        Returns:
            ConfigBox: Configuration object with YAML content.
        """
        try:
            with open(path_to_yaml) as yaml_file:
                content = yaml.safe_load(yaml_file)
                if content is None:
                    raise ValueError("yaml file is empty")
                logger.info(f"yaml file: {path_to_yaml} loaded successfully")
                return ConfigBox(content)
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise e

    @staticmethod
    @validate_arguments
    def create_directories(path_to_directories: List[Path], verbose: bool = True) -> None:
        """Creates a list of directories.

        Args:
            path_to_directories: List of directory paths to create.
            verbose: If True, logs directory creation. Defaults to True.
        """
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"created directory at: {path}")

    @staticmethod
    @validate_arguments
    def save_json(path: Path, data: dict) -> None:
        """Saves data to a JSON file.

        Args:
            path: Path to the JSON file.
            data: Dictionary to save as JSON.
        """
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"json file saved at: {path}")

    @staticmethod
    @validate_arguments
    def load_json(path: Path) -> ConfigBox:
        """Loads data from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            ConfigBox: Configuration object with JSON content.
        """
        with open(path) as f:
            content = json.load(f)
        logger.info(f"json file loaded successfully from: {path}")
        return ConfigBox(content)

    @staticmethod
    @validate_arguments
    def save_bin(data: Any, path: Path) -> None:
        """Saves data as a binary file.

        Args:
            data: Data to save.
            path: Path to the binary file.
        """
        joblib.dump(value=data, filename=path)
        logger.info(f"binary file saved at: {path}")

    @staticmethod
    @validate_arguments
    def load_bin(path: Path) -> Any:
        """Loads data from a binary file.

        Args:
            path: Path to the binary file.

        Returns:
            Any: Object stored in the file.
        """
        data = joblib.load(path)
        logger.info(f"binary file loaded from: {path}")
        return data

    @staticmethod
    @validate_arguments
    def load_data_uciml(repo_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetches data from a UCI ML repository.

        Args:
            repo_id: Integer ID of the UCI repository.

        Raises:
            ValueError: If repo_id is invalid.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features (X) and targets (y) as DataFrames.
        """
        try:
            early_stage_diabetes_risk_prediction = fetch_ucirepo(id=repo_id)
            X = early_stage_diabetes_risk_prediction.data.features
            y = early_stage_diabetes_risk_prediction.data.targets
            return X, y
        except Exception as e:
            if "not a valid dataset id" in str(e).lower():
                raise ValueError(f"Invalid repo_id: {repo_id}")
            raise e