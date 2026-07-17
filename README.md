# Spam Email Detection

A machine learning web application that classifies emails as **Spam** or **Ham** based on the subject and message content, packaged for both local use and containerized deployment.

## Overview

This project treats spam classification as a small end-to-end system rather than a single script: text preprocessing, model training, a prediction pipeline, and a served web interface. The trained model reads an email's subject and body and returns a label plus a confidence score.

## Key Features

- **Reproducible pipeline** for text preprocessing, model training, and evaluation
- **Multiple model backends** (Logistic Regression, Naive Bayes, LinearSVC) for benchmarking
- **Streamlit web application** for interactive, form-based predictions
- **Containerized deployment** via a prebuilt Docker image
- **Test script** covering the prediction pipeline

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Data & ML | pandas, NumPy, scikit-learn, nltk |
| Visualization (EDA) | Matplotlib, Seaborn, WordCloud |
| Serving | Streamlit |
| Containerization | Docker |

## Project Structure

```
Spam-Email-Detection/
├── notebooks/                  # EDA and model training notebooks
├── src/
│   └── spam_detection/         # Core package: pipeline, prediction logic
├── app.py                      # Streamlit application entry point
├── test_prediction.py          # Test script for prediction pipeline
├── setup.py                    # Package metadata and installation
├── template.py                 # Project scaffolding script
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build definition
└── LICENSE
```

## Model Input Features

The prediction pipeline accepts the following inputs:

| Feature | Description |
|---|---|
| `subject` | Email subject line |
| `message` | Email body text |

The service returns a classification (Spam or Ham) along with a confidence score or decision score, depending on the underlying model.

## Getting Started

### Prerequisites

- Python 3.9 or later
- Git
- Docker (optional, for containerized deployment)

### Installation

```bash
git clone https://github.com/RiteshSnippet/Spam-Email-Detection.git
cd Spam-Email-Detection

python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Running the Web Application

```bash
streamlit run app.py
```

The app runs at `http://localhost:8501`. Enter an email subject and message, then click **Predict** to view the result.

### Running Tests

```bash
python test_prediction.py
```

### Docker Deployment

A prebuilt image is available on Docker Hub — no local build required:

```bash
docker pull riteshimage/spam-detection-app:latest
docker run -p 8501:8501 riteshimage/spam-detection-app:latest
```

Then open `http://localhost:8501` in your browser.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Ritesh Kumar Behera**
GitHub: [@RiteshSnippet](https://github.com/RiteshSnippet)

## Contributing

Issues and pull requests are welcome. For significant changes, open an issue first to discuss what you would like to change.