import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from src.spam_detection.logger import logging
from src.spam_detection.exception import CustomException
from src.spam_detection.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts",
        "Preprocessor.joblib"
    )

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Creating TF-IDF Vectorizer")
            vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
            logging.info("TF-IDF Vectorizer created successfully")
            return vectorizer

        except Exception as e:
            logging.info("Exception occurred while creating TF-IDF Vectorizer")
            raise CustomException(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation Started")
            train_df = pd.read_csv(
                train_path
            )

            test_df = pd.read_csv(
                test_path
            )

            logging.info(f"Train Shape : {train_df.shape}")
            logging.info(f"Test Shape : {test_df.shape}")

            X_train = train_df["transformed_text"]

            y_train = train_df["label"]

            X_test = test_df["transformed_text"]

            y_test = test_df["label"]

            logging.info("Input and target features separated")

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Fitting TF-IDF on training data")
            X_train_transformed = preprocessing_obj.fit_transform(X_train)

            logging.info("Transforming testing data")
            X_test_transformed = preprocessing_obj.transform(X_test)


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("TF-IDF Vectorizer saved successfully")
            logging.info("Data Transformation Completed Successfully")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occurred during Data Transformation")
            raise CustomException(e, sys)