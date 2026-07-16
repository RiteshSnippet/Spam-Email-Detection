import streamlit as st
from src.spam_detection.pipeline.prediction_pipeline import PredictPipeline

st.title("E-mail Spam Detection")

subject = st.text_input("Email Subject")

message = st.text_area("Email Message")

if st.button("Predict"):
    pipeline = PredictPipeline()
    result = pipeline.predict(subject, message)

    if result["label"] == "Spam":
        st.error(
            "🚨 This email is Spam"
        )
    else:
        st.success(
            "✅ This email is Ham"
        )

    if result["probability"]:
        st.write(
            f"Spam Probability: {result['probability']:.2%}"
        )