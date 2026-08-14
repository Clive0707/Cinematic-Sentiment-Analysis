"""
Prediction utilities for Marathi movie review sentiment analysis.

Loads pre-trained Multinomial Naive Bayes and Logistic Regression models
alongside the TF-IDF vectorizer from disk.

Provides independent predictions and confidence scores for both models,
analyzes model agreement, and extracts top 3 similar dataset reviews
using cosine similarity on TF-IDF representations.
"""

import json
import os
import pickle

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_corpus, preprocess_text

NB_MODEL_PATH = os.path.join("models", "sentiment_model.pkl")
LR_MODEL_PATH = os.path.join("models", "logistic_regression_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join("models", "metrics.json")
DATASET_PATH = os.path.join("dataset", "marathi_movie_reviews.csv")

# Emoji labels for visual UI rendering
SENTIMENT_EMOJI = {
    "Positive": "😊",
    "Negative": "☹️",
}


class SentimentPredictor:
    """Wrapper for loading dual ML models and executing sentiment prediction & NLP analysis."""

    def __init__(
        self,
        nb_path: str = NB_MODEL_PATH,
        lr_path: str = LR_MODEL_PATH,
        vectorizer_path: str = VECTORIZER_PATH,
        metrics_path: str = METRICS_PATH,
    ):
        self.nb_path = nb_path
        self.lr_path = lr_path
        self.vectorizer_path = vectorizer_path
        self.metrics_path = metrics_path

        self.nb_model = None
        self.lr_model = None
        self.vectorizer = None
        self.metrics = None
        self.dataset_reviews = []
        self.dataset_sentiments = []
        self.dataset_tfidf = None

    def load(self) -> None:
        """Load NB model, LR model, vectorizer, metrics, and build dataset TF-IDF matrix."""
        if not os.path.exists(self.nb_path):
            raise FileNotFoundError(
                f"Naive Bayes model not found at '{self.nb_path}'. Run train_model.py first."
            )
        if not os.path.exists(self.lr_path):
            raise FileNotFoundError(
                f"Logistic Regression model not found at '{self.lr_path}'. Run train_model.py first."
            )
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(
                f"Vectorizer not found at '{self.vectorizer_path}'. Run train_model.py first."
            )

        with open(self.nb_path, "rb") as file:
            self.nb_model = pickle.load(file)

        with open(self.lr_path, "rb") as file:
            self.lr_model = pickle.load(file)

        with open(self.vectorizer_path, "rb") as file:
            self.vectorizer = pickle.load(file)

        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r", encoding="utf-8") as file:
                self.metrics = json.load(file)

        self._load_dataset_vectors()

    def _load_dataset_vectors(self) -> None:
        """Build TF-IDF matrix for the dataset for cosine similarity search."""
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found at '{DATASET_PATH}'.")

        df = pd.read_csv(DATASET_PATH)
        df = df.dropna(subset=["Review", "Sentiment"])

        self.dataset_reviews = df["Review"].astype(str).tolist()
        self.dataset_sentiments = df["Sentiment"].astype(str).str.strip().tolist()

        cleaned_corpus = preprocess_corpus(self.dataset_reviews)
        self.dataset_tfidf = self.vectorizer.transform(cleaned_corpus)

    def _predict_single_model(self, model, features) -> dict:
        """Predict sentiment and extract confidence score from model probabilities."""
        classes = list(model.classes_)
        probs = model.predict_proba(features)[0]

        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}

        pos_prob = prob_dict.get("Positive", 0.0)
        neg_prob = prob_dict.get("Negative", 0.0)

        if pos_prob >= neg_prob:
            sentiment = "Positive"
            confidence = pos_prob * 100
        else:
            sentiment = "Negative"
            confidence = neg_prob * 100

        return {
            "sentiment": sentiment,
            "emoji": SENTIMENT_EMOJI.get(sentiment, ""),
            "confidence": round(confidence, 2),
            "pos_prob": round(pos_prob * 100, 2),
            "neg_prob": round(neg_prob * 100, 2),
            "probabilities": prob_dict,
        }

    def _get_top_similar_reviews(self, features, top_k: int = 3) -> list:
        """Find top-K most similar reviews from dataset using TF-IDF cosine similarity."""
        similarities = cosine_similarity(features, self.dataset_tfidf)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        similar_reviews = []
        for idx in top_indices:
            sim_score = float(similarities[idx])
            similar_reviews.append(
                {
                    "review": self.dataset_reviews[idx],
                    "sentiment": self.dataset_sentiments[idx],
                    "similarity": round(sim_score * 100, 1),
                    "emoji": SENTIMENT_EMOJI.get(self.dataset_sentiments[idx], ""),
                }
            )

        return similar_reviews

    def predict(self, review: str) -> dict:
        """
        Run independent predictions for Multinomial Naive Bayes and Logistic Regression.

        Pipeline:
            1. Preprocess review text -> TF-IDF vector
            2. Predict Naive Bayes independently via predict_proba()
            3. Predict Logistic Regression independently via predict_proba()
            4. Evaluate model agreement
            5. Retrieve top 3 dataset review matches via Cosine Similarity

        Returns:
            Dictionary containing independent model outputs, agreement status,
            similar reviews, and model metrics.
        """
        if self.nb_model is None or self.lr_model is None or self.vectorizer is None:
            raise RuntimeError("Models not loaded. Call load() before predict().")

        cleaned = preprocess_text(review)
        features = self.vectorizer.transform([cleaned])

        # 1. Naive Bayes Independent Prediction
        nb_result = self._predict_single_model(self.nb_model, features)
        nb_result["name"] = "Multinomial Naive Bayes"
        nb_result["role"] = "Primary Classifier"

        # 2. Logistic Regression Independent Prediction
        lr_result = self._predict_single_model(self.lr_model, features)
        lr_result["name"] = "Logistic Regression"
        lr_result["role"] = "Secondary Classifier"

        # 3. Model Agreement Analysis
        is_agree = nb_result["sentiment"] == lr_result["sentiment"]
        agreement = {
            "is_agree": is_agree,
            "status_label": "✓ Both models agree" if is_agree else "⚠ Models disagree",
            "message": (
                "Both models agree on the sentiment."
                if is_agree
                else "The models produced different predictions. Review the confidence scores and model comparison."
            ),
        }

        # 4. Cosine Similarity Top 3 Similar Reviews
        similar_reviews = self._get_top_similar_reviews(features, top_k=3)

        return {
            "review_text": review,
            "cleaned_text": cleaned,
            "naive_bayes": nb_result,
            "logistic_regression": lr_result,
            "agreement": agreement,
            "similar_reviews": similar_reviews,
            "metrics": self.metrics,
        }


# Module-level singleton for Flask app
_predictor = SentimentPredictor()


def get_predictor() -> SentimentPredictor:
    """Return loaded predictor instance (lazy load on first call)."""
    if _predictor.nb_model is None:
        _predictor.load()
    return _predictor


def predict_sentiment(review: str) -> dict:
    """Convenience function to run dual prediction for a single review."""
    return get_predictor().predict(review)

