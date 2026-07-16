import os
import sys
from dataclasses import dataclass
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from src.spam_detection.logger import logging
from src.spam_detection.exception import CustomException
from src.spam_detection.utils.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "Model.joblib"
    )

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Model Training Started")
            models = {
                "Multinomial Naive Bayes":
                    MultinomialNB(),
                "Logistic Regression":
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42
                    )
            }

            logging.info("Models initialized successfully")

            model_report, trained_models = evaluate_model(X_train, y_train, X_test, y_test, models)
            logging.info(f"Model Evaluation Report: {model_report}")

            best_model_name = max(model_report, key=lambda x:model_report[x]["f1_score"])

            best_model = trained_models[best_model_name]

            best_metrics = model_report[best_model_name]

            logging.info(
                f"""
                Best Model Selected:

                Model Name : {best_model_name}
                Accuracy   : {best_metrics['accuracy']}
                Precision  : {best_metrics['precision']}
                Recall     : {best_metrics['recall']}
                F1 Score   : {best_metrics['f1_score']}
                ROC-AUC    : {best_metrics['roc_auc']}
                PR-AUC     : {best_metrics['pr_auc']}
                """
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved successfully")

            return best_model_name

        except Exception as e:
            logging.info("Exception occurred during model training")
            raise CustomException(e, sys)