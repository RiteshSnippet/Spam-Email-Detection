import os
import sys
from dataclasses import dataclass
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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

            # Models initialized using the best hyperparameters obtained from GridSearchCV
            models = {
                "MultinomialNB":
                    MultinomialNB(
                        alpha=0.3,
                        fit_prior=True
                    ),
                "LogisticRegression":
                    LogisticRegression(
                        C=5,
                        penalty="l2",
                        solver="liblinear",
                        max_iter=1000,
                        random_state=42
                    ),
                "LinearSVC":
                    LinearSVC(
                        C=0.5,
                        loss="squared_hinge",
                        random_state=42
                    )
            }

            logging.info("Models initialized successfully")

            model_report, trained_models = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            logging.info("Model evaluation completed")

            best_model_name = sorted(
                model_report,
                key=lambda model: (
                    model_report[model]["f1_score"],
                    model_report[model]["roc_auc"]
                ),reverse=True)[0]
            
            best_model = trained_models[best_model_name]

            best_metrics = model_report[best_model_name]
            logging.info(
                f"""
            ======================== BEST MODEL ========================
            Model Name : {best_model_name}
            Accuracy   : {best_metrics['accuracy']:.4f}
            Precision  : {best_metrics['precision']:.4f}
            Recall     : {best_metrics['recall']:.4f}
            F1 Score   : {best_metrics['f1_score']:.4f}
            ROC AUC    : {best_metrics['roc_auc']:.4f}
            PR AUC     : {best_metrics['pr_auc']:.4f}
            ============================================================
            """
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved successfully")
            logging.info("Model Training Completed")

            return best_model_name
        
        except Exception as e:
            logging.info("Exception occurred during Model Training")
            raise CustomException(e, sys)