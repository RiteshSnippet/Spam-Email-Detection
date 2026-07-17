import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.spam_detection.logger import logging
from src.spam_detection.exception import CustomException
from src.spam_detection.utils.utils import transform_text


@dataclass
class DataIngestionConfig:

    raw_data_path: str = os.path.join(
        "artifacts",
        "raw_data.csv"
    )

    train_data_path: str = os.path.join(
        "artifacts",
        "train_data.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts",
        "test_data.csv"
    )


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion Started")

            data = pd.read_csv(r"C:\Users\LENOVO\Desktop\E2E-Spam-Email-Detection\notebooks\data\spam.csv", encoding="latin1")
            logging.info(f"Dataset loaded successfully. Shape: {data.shape}")

            data.drop(
                columns=[
                    "Unnamed: 2",
                    "Unnamed: 3",
                    "Unnamed: 4"
                ], inplace=True, errors="ignore"
            )

            data.rename(
                columns={
                    "v1": "label",
                    "v2": "text"
                }, inplace=True
            )

            data.drop_duplicates(
                keep="first",
                inplace=True
            )

            data.reset_index(
                drop=True,
                inplace=True
            )

            logging.info("Duplicate values removed")

            data["label"] = data["label"].map(
                {
                    "ham":0,
                    "spam":1
                }
            )

            logging.info("Applying text preprocessing")

            data["transformed_text"] = data["text"].apply(transform_text)
            logging.info("Text preprocessing completed")

            data = data[
                [
                    "text",
                    "transformed_text",
                    "label"
                ]
            ]

            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.raw_data_path
                ), exist_ok=True
            )


            data.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            logging.info(f"Raw dataset saved at {self.ingestion_config.raw_data_path}")

            train_data, test_data = train_test_split(
                data,
                test_size=0.20,
                random_state=42,
                stratify=data["label"]
            )

            logging.info(f"Training samples: {train_data.shape[0]}")
            logging.info(f"Testing samples: {test_data.shape[0]}")

            train_data.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            test_data.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            logging.info("Train and Test datasets saved successfully")
            logging.info("Data Ingestion Completed Successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occurred during Data Ingestion")
            raise CustomException(e, sys)