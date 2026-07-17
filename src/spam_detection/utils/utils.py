import os
import sys
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, classification_report
from src.spam_detection.logger import logging
from src.spam_detection.exception import CustomException
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
for package in [
    "punkt",
    "stopwords",
    "wordnet",
    "omw-1.4"
]:
    nltk.download(package, quiet=True)


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
            logging.info(f"Training {model_name}")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                y_score = y_pred


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

            roc_auc = roc_auc_score(
                y_test,
                y_score
            )

            pr_auc = average_precision_score(
                y_test,
                y_score
            )

            print("\n" + "=" * 60)
            print(model_name)
            print("=" * 60)
            print(
                classification_report(
                    y_test,
                    y_pred,
                    target_names=["Ham", "Spam"]
                )
            )
            print("=" * 60 + "\n")

            report[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc
            }
            trained_models[model_name] = model

            logging.info(

                f"""
                Model : {model_name}
                Accuracy : {accuracy:.4f}
                Precision : {precision:.4f}
                Recall : {recall:.4f}
                F1 Score : {f1:.4f}
                ROC AUC : {roc_auc:.4f}
                PR AUC : {pr_auc:.4f}
                """
            )

        return report, trained_models
    
    except Exception as e:
        logging.info("Exception occurred during model evaluation")
        raise CustomException(e, sys)
    

LEMMATIZER = WordNetLemmatizer()
def transform_text(text: str) -> str:
    # lowercase
    text = text.lower()

    # tokenize
    text = nltk.word_tokenize(text)

    # remove special characters/punctuation
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # remove stopwords
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # lemmatization 
    for i in text:
        y.append(LEMMATIZER.lemmatize(i))

    return " ".join(y)