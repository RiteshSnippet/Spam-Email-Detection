import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.spam_detection.logger import logging
from src.spam_detection.exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")

        try:
            data = pd.read_csv("../notebooks/data/enron_spam_data.csv")
            logging.info("Dataset loaded successfully.")

            data["Message"] = data["Message"].fillna("")

            data["text"] = data["Subject"] + " " + data["Message"]

            data["label"] = data["Spam/Ham"].map({
                "ham": 0,
                "spam": 1
            })

            data = data[["text", "label"]]
            logging.info("Text preprocessing completed.")

            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path),
                exist_ok=True
            )

            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved.")

            train_data, test_data = train_test_split(data, test_size=0.20, random_state=42, stratify=data["label"])
            logging.info("Train-Test split completed.")

            train_data.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )
            test_data.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )
            logging.info("Train and Test datasets saved successfully.")
            logging.info("Data Ingestion Completed Successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occurred during Data Ingestion.")
            raise CustomException(e, sys)