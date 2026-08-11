# Marathi Movie Review Sentiment Analysis

A complete sentiment analysis web application for **Marathi movie reviews**, built with Python, Flask, and Multinomial Naive Bayes.

## Project Structure

```
project/
│
├── dataset/
│   └── marathi_movie_reviews.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── preprocess.py
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## Dataset

- **File:** `dataset/marathi_movie_reviews.csv`
- **Rows:** 1000
- **Columns:** `Review`, `Sentiment`
- **Classes:** Positive, Negative

## Setup

1. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model:**

   ```bash
   python train_model.py
   ```

   This will:
   - Preprocess Marathi reviews (remove punctuation, numbers, stopwords)
   - Vectorize text using TF-IDF
   - Split data 80/20 (random_state=42)
   - Train Multinomial Naive Bayes
   - Print accuracy, precision, recall, F1, confusion matrix, and classification report
   - Save `models/sentiment_model.pkl` and `models/tfidf_vectorizer.pkl`

4. **Run the Flask web app:**

   ```bash
   python app.py
   ```

5. **Open in browser:**

   ```
   http://127.0.0.1:5000
   ```

## Usage

1. Enter a Marathi movie review in the text box.
2. Click **Predict**.
3. View the result:
   - **Positive** 😊 or **Negative** ☹️
   - Prediction confidence percentage

## Modules

| File | Description |
|------|-------------|
| `preprocess.py` | Text cleaning and Marathi stopword removal |
| `train_model.py` | Dataset loading, training, evaluation, model saving |
| `predict.py` | Load saved model and predict sentiment |
| `app.py` | Flask web server (loads model from disk, no retraining) |

## Model Pipeline

1. **Preprocessing** — Keep Devanagari Unicode, strip punctuation/numbers/special chars, remove stopwords
2. **Feature extraction** — TF-IDF Vectorizer
3. **Classifier** — Multinomial Naive Bayes
4. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report

## Notes

- The Flask app loads the pre-trained model at startup and does **not** retrain.
- Run `train_model.py` before starting the web app for the first time.
- Marathi text is preserved using the Devanagari Unicode range (U+0900–U+097F).

## License

MIT
