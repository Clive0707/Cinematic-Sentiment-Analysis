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
    "Positive": "",
    "Negative": "",
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

    def _extract_feature_contributions(self, model, features, model_type: str) -> dict:
        """
        Extract positive and negative feature contributions present in the review text.
        Derived from actual model parameters (feature_log_prob_ for MNB, coef_ for LR).
        """
        if self.vectorizer is None:
            return {"positive_features": [], "negative_features": []}

        feature_names = self.vectorizer.get_feature_names_out()
        nonzero_indices = features.nonzero()[1]

        if len(nonzero_indices) == 0:
            return {"positive_features": [], "negative_features": []}

        classes = list(model.classes_)
        pos_idx = classes.index("Positive") if "Positive" in classes else 1
        neg_idx = classes.index("Negative") if "Negative" in classes else 0

        pos_features = []
        neg_features = []

        for idx in nonzero_indices:
            word = feature_names[idx]
            tfidf_weight = float(features[0, idx])

            if model_type == "naive_bayes":
                # Difference in log probability between Positive and Negative classes
                log_prob_pos = model.feature_log_prob_[pos_idx, idx]
                log_prob_neg = model.feature_log_prob_[neg_idx, idx]
                score = tfidf_weight * (log_prob_pos - log_prob_neg)
            else:  # logistic_regression
                # Coefficient weight for Positive class
                coef = model.coef_[0, idx] if pos_idx == 1 else -model.coef_[0, idx]
                score = tfidf_weight * float(coef)

            entry = {"word": word, "score": round(score, 4), "weight": round(abs(score), 4)}
            if score > 0:
                pos_features.append(entry)
            elif score < 0:
                neg_features.append(entry)

        # Sort positive features descending by score, negative features ascending by score
        pos_features.sort(key=lambda x: x["score"], reverse=True)
        neg_features.sort(key=lambda x: x["score"])

        return {
            "positive_features": pos_features,
            "negative_features": neg_features,
        }

    def _predict_single_model(self, model, features, model_type: str) -> dict:
        """Predict sentiment and extract confidence score and explainability from model probabilities."""
        classes = list(model.classes_)
        probs = model.predict_proba(features)[0]

        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}

        pos_prob = prob_dict.get("Positive", 0.0) * 100
        neg_prob = prob_dict.get("Negative", 0.0) * 100

        if pos_prob >= neg_prob:
            sentiment = "Positive"
            confidence = pos_prob
            winning_prob = pos_prob
            alt_class = "Negative"
            alt_prob = neg_prob
        else:
            sentiment = "Negative"
            confidence = neg_prob
            winning_prob = neg_prob
            alt_class = "Positive"
            alt_prob = pos_prob

        evidence = self._extract_feature_contributions(model, features, model_type)

        explanation = {
            "pos_prob": round(pos_prob, 1),
            "neg_prob": round(neg_prob, 1),
            "winning_class": sentiment,
            "winning_prob": round(winning_prob, 1),
            "alt_class": alt_class,
            "alt_prob": round(alt_prob, 1),
            "summary_text": (
                f"The model assigns a {winning_prob:.1f}% probability to {sentiment} and a "
                f"{alt_prob:.1f}% probability to {alt_class} for this review."
            ),
            "remaining_explanation": (
                f"The remaining {alt_prob:.1f}% represents the model's estimated probability for the "
                f"alternative sentiment class ({alt_class}), ensuring the probability distribution sums to 100% "
                f"({winning_prob:.1f}% + {alt_prob:.1f}% = 100%)."
            ),
            "statistical_basis": (
                f"Statistical Basis: The model assigned a higher probability to {sentiment} based on the learned "
                f"statistical weights of the TF-IDF features present in your review, evaluated against the training dataset."
            ),
        }

        return {
            "sentiment": sentiment,
            "emoji": SENTIMENT_EMOJI.get(sentiment, ""),
            "confidence": round(confidence, 2),
            "pos_prob": round(pos_prob, 2),
            "neg_prob": round(neg_prob, 2),
            "probabilities": prob_dict,
            "explanation": explanation,
            "positive_features": evidence["positive_features"],
            "negative_features": evidence["negative_features"],
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
            4. Evaluate model agreement & confidence difference
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
        nb_result = self._predict_single_model(self.nb_model, features, "naive_bayes")
        nb_result["name"] = "Multinomial Naive Bayes"
        nb_result["role"] = "Primary Classifier"

        # 2. Logistic Regression Independent Prediction
        lr_result = self._predict_single_model(self.lr_model, features, "logistic_regression")
        lr_result["name"] = "Logistic Regression"
        lr_result["role"] = "Secondary Classifier"

        # 3. Model Agreement & Confidence Difference Analysis
        is_agree = nb_result["sentiment"] == lr_result["sentiment"]
        nb_conf = nb_result["confidence"]
        lr_conf = lr_result["confidence"]
        conf_diff = round(abs(lr_conf - nb_conf), 2)

        if lr_conf > nb_conf:
            diff_text = f"Logistic Regression is more confident by {conf_diff:.1f} percentage points."
        elif nb_conf > lr_conf:
            diff_text = f"Multinomial Naive Bayes is more confident by {conf_diff:.1f} percentage points."
        else:
            diff_text = "Both models have equal confidence scores."

        agreement = {
            "is_agree": is_agree,
            "status_label": "✓ Both models agree" if is_agree else "⚠ Models disagree",
            "message": (
                f"Both models agree on the sentiment ({nb_result['sentiment']})."
                if is_agree
                else "The models produced different predictions. Review the confidence scores and model comparison."
            ),
            "conf_diff": conf_diff,
            "diff_text": diff_text,
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

