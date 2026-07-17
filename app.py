import streamlit as st
from src.spam_detection.pipeline.prediction_pipeline import PredictPipeline

# Page configuration
st.set_page_config(
    page_title="Spam Detection",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Detection")
st.write(
    "Enter an email subject and message to check whether it is Spam or Ham."
)

subject = st.text_input(
    "Email Subject"
)

message = st.text_area(
    "Email Message",
    height=200
)

if st.button("Predict"):
    if subject.strip() == "" and message.strip() == "":
        st.warning(
            "Please enter an email subject or message."
        )
    else:
        try:
            pipeline = PredictPipeline()

            result = pipeline.predict(
                subject,
                message
            )

            st.subheader("Prediction Result")

            if result["label"] == "Spam":
                st.error(
                    "🚨 This email is predicted as Spam"
                )
            else:
                st.success(
                    "✅ This email is predicted as Ham"
                )

            st.write(
                f"Prediction Class: **{result['label']}**"
            )

            if result["confidence"] is not None:
                confidence = result["confidence"]

                # Logistic Regression / Naive Bayes probability
                if 0 <= confidence <= 1:
                    st.write(
                        f"Spam Probability: **{confidence:.2%}**"
                    )

                # LinearSVC decision score
                else:
                    st.write(
                        f"Decision Score: **{confidence:.4f}**"
                    )

        except Exception as e:
            st.error(
                "An error occurred while making prediction."
            )
            st.exception(e)