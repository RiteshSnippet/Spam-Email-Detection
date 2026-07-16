from src.spam_detection.pipeline.prediction_pipeline import PredictPipeline

pipeline = PredictPipeline()

result = pipeline.predict(
    "Congratulations",
    "You have won a free prize. Click here now."
)

print(result)