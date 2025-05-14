from src.components.DataIngestion import DataIngestion
from src.components.DataTransformation import TransformData
from src.components.ModelTrainer import ModelTrainer
from src.logger.custom_logger import logger

if __name__ == '__main__':
    try:
        ingest = DataIngestion().ingest_data()
        train_df,test_df = TransformData().InitiateTransformation()
        X_train = train_df.drop(columns=train_df.columns[-1])
        y_train = train_df[train_df.columns[-1]]
        X_test = test_df.drop(columns=train_df.columns[-1])
        y_test = test_df[train_df.columns[-1]]
        trainer = ModelTrainer(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
        trainer.train_and_evaluate()
    except Exception as e:
        logger.error(e)
    else:
        logger.info('All Process Completed Successfully')