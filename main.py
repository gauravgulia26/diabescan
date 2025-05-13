from src.components.DataIngestion import DataIngestion
from src.components.DataTransformation import TransformData
from src.logger.custom_logger import logger

if __name__ == '__main__':
    try:
        ingest = DataIngestion().ingest_data()
        transform = TransformData().InitiateTransformation()
    except Exception as e:
        logger.error(e)
    else:
        logger.info('Ingested and Transformation Pipeline built Successfully !!')