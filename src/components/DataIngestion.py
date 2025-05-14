from src.logger.custom_logger import CustomLogger
from src.exceptions.CustomException import CustomException
from src.constants import CURRENT_PATH, CFG_FILE_PATH, RAW_DATA_FILE_DIR
from src.utils.common import DataUtils
import pandas as pd
import os
from pathlib import Path
from joblib import Memory


class DataIngestion:

    def __init__(
        self,
        cfg_file_path: str = CFG_FILE_PATH,
        current_path: str = CURRENT_PATH,
        raw_data_file_dir: str = RAW_DATA_FILE_DIR,
    ):
        """Initialize the DataIngestion class with necessary paths and logger."""
        self.__logger = CustomLogger().get_logger()
        self.loader = DataUtils()
        self.CFG_FILE_PATH = cfg_file_path
        self.CURRENT_PATH = current_path
        self.RAW_DATA_FILE_DIR = raw_data_file_dir

    def fetch_data(self, repo_id):
        """Fetch data from UCI ML repository by ID."""
        self.__logger.info(f"Fetching data from UCI ML repository with ID: {repo_id}")
        try:
            x, y = self.loader.load_data_uciml(repo_id=repo_id)
            return x, y
        except Exception as e:
            self.__logger.error(f"Failed to fetch data: {str(e)}")
            raise CustomException(error_message=e)

    def create_dataframe(self, x, y):
        """Convert data to DataFrame."""
        self.__logger.info("Creating DataFrame from fetched data")
        try:
            df = pd.concat([x, y], axis=1)
            return df
        except Exception as e:
            self.__logger.error(f"Failed to create DataFrame: {str(e)}")
            raise CustomException(error_message=e)

    def load_config(self):
        """Load configuration from YAML file."""
        self.__logger.info(f"Loading configuration from {self.CFG_FILE_PATH}")
        try:
            yaml_file = self.loader.read_yaml(path_to_yaml=self.CFG_FILE_PATH)
            return yaml_file
        except Exception as e:
            self.__logger.error(f"Failed to load configuration: {str(e)}")
            raise CustomException(error_message=e)

    def create_directory(self, yaml_file):
        """Create directory based on configuration."""
        self.__logger.info("Creating directory structure")
        try:
            root_dir = yaml_file.data_ingestion.root_dir
            full_path = Path(os.path.join(self.CURRENT_PATH, root_dir))
            os.makedirs(full_path, exist_ok=True)
            return full_path
        except Exception as e:
            self.__logger.error(f"Failed to create directory: {str(e)}")
            raise CustomException(error_message=e)

    def save_data(self, df):
        """Save DataFrame to specified path."""
        self.__logger.info(f"Saving data to {self.RAW_DATA_FILE_DIR}")
        try:
            df.to_csv(self.RAW_DATA_FILE_DIR)
        except Exception as e:
            self.__logger.error(f"Failed to save data: {str(e)}")
            raise CustomException(error_message=e)

    def ingest_data(self, repo_id=529):
        """Main method to orchestrate the data ingestion process."""
        try:
            # Step 1: Fetch data
            x, y = self.fetch_data(repo_id)
            self.__logger.info("Data fetched successfully")

            # Step 2: Create DataFrame
            df = self.create_dataframe(x, y)
            self.__logger.info("DataFrame created successfully")

            # Step 3: Load configuration
            yaml_file = self.load_config()
            self.__logger.info("Configuration loaded successfully")

            # Step 4: Create directory
            full_path = self.create_directory(yaml_file)
            self.__logger.info(f"Directory created at {full_path}")

            # Step 5: Save data
            self.save_data(df)
            self.__logger.info(f"Data saved successfully at {RAW_DATA_FILE_DIR}")

        except CustomException as ce:
            self.__logger.error("Data ingestion failed")
            return ce
        except Exception as e:
            self.__logger.error(f"Unexpected error: {str(e)}")
            return CustomException(error_message=e)
        else:
            self.__logger.info("Data Ingestion Completed Successfully!!")
