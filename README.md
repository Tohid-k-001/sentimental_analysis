# Sentiment Analysis Project

A Streamlit web application that classifies IMDB movie reviews as positive or negative using a machine learning model trained on cleaned review text. The project contains both the training workflow (via a Jupyter notebook) and an interactive front end for end users.

## Features

- Streamlit UI with real-time sentiment prediction and confidence score visualisation.
- Optional word cloud generation for submitted text.
- Reusable preprocessing pipeline that strips URLs, HTML, punctuation, digits, and stop words.
- Notebook-driven experimentation for training, evaluation, and model export.

## Project Structure

- `app.py` – Streamlit application that loads the trained model (`sentiment_model.pkl`) and TF-IDF vectorizer (`vectorizer.pkl`).
- `analyzer.ipynb` – End-to-end workflow for data exploration, text cleaning, model training, and evaluation.
- `IMDB Dataset.csv` – Source dataset of IMDB reviews and sentiment labels.
- `requirements.txt` – Python dependencies required for both the notebook and app.
- `sentiment_model.pkl`, `vectorizer.pkl` – Saved artifacts produced by the notebook and consumed by the app.

## Prerequisites

- Python 3.10+ recommended (the bundled `requirements.txt` expects a modern Python build).
- Node-based tools are **not** required; everything runs on Python.

## Setup

1. **Clone and navigate**
   ```powershell
   git clone https://github.com/dukebismaya/sentiment_analysis_project.git
   cd sentiment_analysis_project
   ```
2. **Create a virtual environment (optional but recommended)**
   ```powershell
   python -m venv myvenv
   .\myvenv\Scripts\activate
   ```
3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Download NLTK stopwords** (first run only)
   ```python
   import nltk
   nltk.download('stopwords')
   ```
5. **Ensure model artifacts exist**
   - If you already have `sentiment_model.pkl` and `vectorizer.pkl`, keep them in the project root.
   - Otherwise, open `analyzer.ipynb`, run through the cells to train a model, and save the artifacts using `joblib.dump`.

## Running the Streamlit App

```powershell
streamlit run app.py
```

Open the provided local URL to interact with the sentiment analyzer. Enter any movie review-style text, click **Analyze Sentiment**, and the app will display the predicted label, confidence, and (optionally) a word cloud of key terms.

## Retraining the Model

1. Launch Jupyter Notebook:
   ```powershell
   jupyter notebook analyzer.ipynb
   ```
2. Step through the cells to:
   - Explore the dataset.
   - Clean and preprocess text with the `clean_text` helper.
   - Train candidate models (e.g., Logistic Regression, Linear SVM).
   - Evaluate performance metrics and select the best model.
3. Export the chosen estimator and vectorizer:
   ```python
   import joblib
   joblib.dump(best_model, "sentiment_model.pkl")
   joblib.dump(tfidf_vectorizer, "vectorizer.pkl")
   ```
4. Restart the Streamlit app to pick up the new artifacts.

## Notes

- The dataset is large (~50k reviews). Training steps can take a few minutes depending on hardware.
- Adjust the `clean_text` function in the notebook/app if you need different tokenisation rules or additional preprocessing.
- Keep any API keys or secrets out of the repository; `.env` files are ignored by default.

## Credit
[IMDB Dataset kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## License

© 2025 Bismaya Jyoti Dalei. All rights reserved.
