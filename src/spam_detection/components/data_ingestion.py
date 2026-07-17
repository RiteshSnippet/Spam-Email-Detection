import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.spam_detection.logger import logging
from src.spam_detection.exception import CustomException


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
        logging.info("Data Ingestion Started")

        try:
            logging.info("Loading dataset")

            df = pd.read_csv(r"C:\Users\LENOVO\Desktop\E2E-Spam-Email-Detection\notebooks\data\spam.csv", encoding="latin1")
            logging.info(f"Dataset loaded successfully with shape: {df.shape}")

            df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, errors="ignore")
            logging.info("Dropped unnecessary columns")

            df.rename(columns={"v1": "label", "v2": "text"}, inplace=True)
            logging.info("Columns renamed successfully")

            initial_rows = df.shape[0]

            df.drop_duplicates(keep="first", inplace=True)
            df.reset_index(drop=True, inplace=True)
            logging.info(f"Removed {initial_rows - df.shape[0]} duplicate rows")

            df["label"] = df["label"].map({
                "ham": 0,
                "spam": 1
            })
            logging.info("Labels converted to numerical values")
            logging.info(f"Final dataset shape: {df.shape}")
            logging.info(f"{df.head(1)}")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
        
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            logging.info(f"Raw dataset saved at: {self.ingestion_config.raw_data_path}")
            
            train_df, test_df = train_test_split(
                df,
                test_size=0.20,
                random_state=42,
                stratify=df["label"]
            )

            logging.info(f"Training samples: {train_df.shape[0]}")
            logging.info(f"Testing samples: {test_df.shape[0]}")

            train_df.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            test_df.to_csv(
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