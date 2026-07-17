import os
import sys
import pandas as pd
from src.spam_detection.logger import logging
from src.spam_detection.exception import CustomException
from src.spam_detection.utils.utils import load_object, transform_text

class PredictPipeline:
    def __init__(self):
        pass


    def predict(self, subject, message):
        try:
            logging.info("Prediction pipeline started")
            preprocessor_path = os.path.join(
                "artifacts",
                "Preprocessor.joblib"
            )


            model_path = os.path.join(
                "artifacts",
                "Model.joblib"
            )


            preprocessor = load_object(
                preprocessor_path
            )


            model = load_object(
                model_path
            )


            logging.info("TF-IDF Vectorizer and Model loaded successfully")

            custom_data = CustomData(
                subject,
                message
            )

            input_df = custom_data.get_data_as_dataframe()
            
            logging.info("Applying text preprocessing")
            input_df["text"] = input_df["text"].apply(transform_text)

            logging.info("Text preprocessing completed")

            transformed_data = (
                preprocessor.transform(
                    input_df["text"]
                )
            )

            logging.info("Text transformed using TF-IDF")

            prediction = model.predict(
                transformed_data
            )

            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = model.predict_proba(transformed_data)[0][1]

            elif hasattr(model, "decision_function"):
                confidence = model.decision_function(transformed_data)[0]


            result = {
                "prediction": int(prediction[0]),
                "label":
                    "Spam"
                    if prediction[0] == 1
                    else "Ham",
                "confidence": confidence
            }

            logging.info(f"Prediction result: {result}")

            return result



        except Exception as e:
            logging.info("Exception occurred in prediction pipeline")
            raise CustomException(e, sys)




class CustomData:
    def __init__(self, subject: str, message: str):
        self.subject = subject
        self.message = message


    def get_data_as_dataframe(self):
        try:

            text = self.subject + " " + self.message

            data = {
                "text": [text]
            }


            df = pd.DataFrame(
                data
            )
            logging.info("Prediction dataframe created successfully")
            return df



        except Exception as e:
            logging.info("Exception occurred while creating prediction dataframe")
            raise CustomException(e, sys)