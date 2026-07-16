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


    def get_data_transformation(self):
        try:
            logging.info("Creating TF-IDF transformation object")
            tfidf = TfidfVectorizer(
                stop_words="english",
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2
            )
            logging.info("TF-IDF object created successfully")

            return tfidf

        except Exception as e:
            logging.info("Exception occurred while creating transformation object")
            raise CustomException(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation Started")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Train dataset shape: {train_df.shape}")
            logging.info(f"Test dataset shape: {test_df.shape}")

            target_column_name = "label"

            X_train = train_df["text"]
            y_train = train_df[target_column_name]


            X_test = test_df["text"]
            y_test = test_df[target_column_name]

            logging.info("Input and target features separated")

            preprocessing_obj = self.get_data_transformation()

            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            logging.info("TF-IDF fitted on training data")

            X_test_transformed = preprocessing_obj.transform( X_test)
            logging.info("Training and testing data transformed")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("TF-IDF preprocessing object saved successfully")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occurred during data transformation")
            raise CustomException(e, sys)