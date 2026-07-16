import sys
from src.spam_detection.components.data_ingestion import DataIngestion
from src.spam_detection.components.data_transformation import DataTransformation
from src.spam_detection.components.model_trainer import ModelTrainer
from src.spam_detection.logger import logging
from src.spam_detection.exception import CustomException

def start_training_pipeline():
    try:
        logging.info("Training Pipeline Started")

        logging.info("Starting Data Ingestion")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        logging.info("Starting Data Transformation")
        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test, preprocessor_path = data_transformation.initialize_data_transformation(train_data_path, test_data_path)
        logging.info("Data Transformation Completed")

        logging.info("Starting Model Training")
        model_trainer = ModelTrainer()
        best_model_name = model_trainer.initiate_model_training(X_train, X_test, y_train, y_test)
        logging.info(f"Training completed. Best Model: {best_model_name}")

        return best_model_name

    except Exception as e:
        logging.info("Exception occurred in training pipeline")
        raise CustomException(e, sys)


if __name__ == "__main__":
    start_training_pipeline()