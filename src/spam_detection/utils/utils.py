import os
import sys
import joblib
from src.spam_detection.logger import logging
from src.spam_detection.exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, classification_report

def save_object(file_path, obj):
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.info("Exception occurred while saving object")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        obj = joblib.load(file_path)
        logging.info(f"Object loaded from {file_path}")
        return obj

    except Exception as e:
        logging.info("Exception occurred while loading object")
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            y_probability = None
            if hasattr(model,"predict_proba"):
                y_probability = model.predict_proba(X_test)[:,1]

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            roc_auc = None
            pr_auc = None

            if y_probability is not None:
                roc_auc = roc_auc_score(
                    y_test,
                    y_probability
                )

                pr_auc = average_precision_score(
                    y_test,
                    y_probability
                )

            print("\n==============================")
            print(model_name)

            print(classification_report(y_test, y_pred, target_names=["Ham","Spam"]))

            print("==============================\n")
            report[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc
            }

            trained_models[model_name] = model

        return (
            report,
            trained_models
        )

    except Exception as e:
        logging.info("Exception occurred during model evaluation")
        raise CustomException(e, sys)