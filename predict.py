"""
Prediction utilities for Marathi movie review sentiment analysis.

Loads pre-trained model and TF-IDF vectorizer from disk (no retraining).
Cosine similarity on dataset TF-IDF vectors assists the final confidence score
(blended with Multinomial Naive Bayes — NB remains the primary classifier).
"""

import os
import pickle

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_corpus, preprocess_text

MODEL_PATH = os.path.join("models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
DATASET_PATH = os.path.join("dataset", "marathi_movie_reviews.csv")

# NB is primary; cosine similarity assists the blended prediction score
NB_WEIGHT = 0.75
SIMILARITY_WEIGHT = 0.25
TOP_K_NEIGHBORS = 15

# Emoji labels for web display
SENTIMENT_EMOJI = {
    "Positive": "😊",
    "Negative": "☹️",
}


class SentimentPredictor:
    """Wrapper for loading saved artifacts and predicting sentiment."""

    def __init__(self, model_path: str = MODEL_PATH, vectorizer_path: str = VECTORIZER_PATH):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.dataset_sentiments = None
        self.dataset_tfidf = None

    def load(self) -> None:
        """Load model, vectorizer, and dataset TF-IDF matrix for similarity scoring."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at '{self.model_path}'. Run train_model.py first."
            )
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(
                f"Vectorizer not found at '{self.vectorizer_path}'. Run train_model.py first."
            )

        with open(self.model_path, "rb") as file:
            self.model = pickle.load(file)

        with open(self.vectorizer_path, "rb") as file:
            self.vectorizer = pickle.load(file)

        self._load_dataset_vectors()

    def _load_dataset_vectors(self) -> None:
        """Build TF-IDF matrix for the dataset (used for cosine similarity scoring)."""
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found at '{DATASET_PATH}'.")

        df = pd.read_csv(DATASET_PATH)
        df = df.dropna(subset=["Review", "Sentiment"])

        reviews = df["Review"].astype(str).tolist()
        self.dataset_sentiments = df["Sentiment"].astype(str).str.strip().tolist()
        cleaned_corpus = preprocess_corpus(reviews)
        self.dataset_tfidf = self.vectorizer.transform(cleaned_corpus)

    def _similarity_sentiment_scores(self, features) -> tuple[float, float]:
        """
        Derive Positive/Negative scores from top similar dataset reviews.

        Uses cosine similarity between TF-IDF vectors — assists the final score,
        does not replace Naive Bayes classification.
        """
        similarities = cosine_similarity(features, self.dataset_tfidf)[0]
        top_indices = similarities.argsort()[-TOP_K_NEIGHBORS:][::-1]
        top_sims = similarities[top_indices]

        # Ignore zero-similarity neighbors
        positive_weight = 0.0
        negative_weight = 0.0
        for idx, sim in zip(top_indices, top_sims):
            if sim <= 0:
                continue
            if self.dataset_sentiments[idx] == "Positive":
                positive_weight += sim
            else:
                negative_weight += sim

        total = positive_weight + negative_weight
        if total == 0:
            return 0.5, 0.5

        return positive_weight / total, negative_weight / total

    def predict(self, review: str) -> dict:
        """
        Predict sentiment and blended confidence for a Marathi review.

        Pipeline:
            1. Preprocess → TF-IDF
            2. Multinomial Naive Bayes probabilities (primary)
            3. Cosine similarity scores from similar dataset reviews (assist)
            4. Blend into final sentiment and confidence

        Returns:
            Dictionary with sentiment, emoji, confidence, and cleaned text.
        """
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not loaded. Call load() before predict().")

        cleaned = preprocess_text(review)
        features = self.vectorizer.transform([cleaned])

        # Primary classifier — Multinomial Naive Bayes
        classes = list(self.model.classes_)
        nb_probs = self.model.predict_proba(features)[0]
        nb_scores = {label: prob for label, prob in zip(classes, nb_probs)}

        nb_positive = nb_scores.get("Positive", 0.0)
        nb_negative = nb_scores.get("Negative", 0.0)

        # Assist — cosine similarity weighted sentiment from similar reviews
        sim_positive, sim_negative = self._similarity_sentiment_scores(features)

        # Blended prediction score
        blended_positive = NB_WEIGHT * nb_positive + SIMILARITY_WEIGHT * sim_positive
        blended_negative = NB_WEIGHT * nb_negative + SIMILARITY_WEIGHT * sim_negative

        total = blended_positive + blended_negative
        if total > 0:
            blended_positive /= total
            blended_negative /= total

        if blended_positive >= blended_negative:
            sentiment = "Positive"
            confidence = blended_positive * 100
        else:
            sentiment = "Negative"
            confidence = blended_negative * 100

        return {
            "sentiment": sentiment,
            "emoji": SENTIMENT_EMOJI.get(sentiment, ""),
            "confidence": round(confidence, 2),
            "cleaned_text": cleaned,
        }


# Module-level singleton for Flask app
_predictor = SentimentPredictor()


def get_predictor() -> SentimentPredictor:
    """Return a loaded predictor instance (lazy load on first call)."""
    if _predictor.model is None:
        _predictor.load()
    return _predictor


def predict_sentiment(review: str) -> dict:
    """Convenience function to predict sentiment for a single review."""
    return get_predictor().predict(review)
