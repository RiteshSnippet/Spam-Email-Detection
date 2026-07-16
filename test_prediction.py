from src.spam_detection.pipeline.prediction_pipeline import PredictPipeline

pipeline = PredictPipeline()

# result = pipeline.predict(
#     "Meeting tomorrow",
#     """
#     Hi team,

#     The project meeting is scheduled for tomorrow at 10 AM.
#     Please bring your updates.

#     Thanks.
#     """
# )

result = pipeline.predict(
    "Cheap mortgage rates available",
    """
    Get the lowest mortgage rates today.
    Special offer available.
    Contact us for more information.
    """
)

print(result)