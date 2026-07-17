from src.spam_detection.pipeline.prediction_pipeline import PredictPipeline

def test_predictions():
    pipeline = PredictPipeline()

    examples = [
        (
            "Congratulations!",
            "You have won a FREE vacation to Dubai. Call now!"
        ),

        (
            "URGENT!",
            "Your bank account has been suspended. Verify immediately."
        ),

        (
            "Amazon Gift Card",
            "You have been selected to receive a $500 Amazon gift card."
        ),

        (
            "Lottery Prize",
            "Claim your lottery prize by clicking the link below."
        ),

        (
            "XXXMobileMovieClub",
            """
            To use your credit, click the WAP link in the next txt message
            or click here>>
            http://wap.xxxmobilemovieclub.com?n=QJKGIGHJJGCBL
            """
        ),

        (
            "Cheap mortgage rates available",
            """
            Get the lowest mortgage rates today.
            Special offer available.
            Contact us for more information.
            """
        )
    ]

    print("\n========== Spam Detection Testing ==========\n")
    for subject, message in examples:
        print("-" * 70)
        print("Subject:")
        print(subject)
        print("\nMessage:")
        print(message)

        result = pipeline.predict(
            subject,
            message
        )

        print("\nPrediction:")
        print(result["label"])

        if result["confidence"] is not None:
            print(
                "Confidence:",
                result["confidence"]
            )

        print("-" * 70)
        print()

if __name__ == "__main__":
    test_predictions()