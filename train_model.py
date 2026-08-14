"""
Train Multinomial Naive Bayes and Logistic Regression sentiment classifiers on Marathi movie reviews.

Reads the dataset, preprocesses text, vectorizes with TF-IDF, trains both models,
evaluates performance, and saves model artifacts and metrics to the models/ directory.
"""

import json
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import preprocess_corpus

# Paths relative to project root
DATASET_PATH = os.path.join("dataset", "marathi_movie_reviews.csv")
NB_MODEL_PATH = os.path.join("models", "sentiment_model.pkl")
LR_MODEL_PATH = os.path.join("models", "logistic_regression_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join("models", "metrics.json")

# Reproducible split
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the Marathi movie reviews CSV."""
    df = pd.read_csv(path)

    if "Review" not in df.columns or "Sentiment" not in df.columns:
        raise ValueError("Dataset must contain 'Review' and 'Sentiment' columns.")

    df = df.dropna(subset=["Review", "Sentiment"])
    df["Review"] = df["Review"].astype(str)
    df["Sentiment"] = df["Sentiment"].astype(str).str.strip()

    return df


def save_pickle(obj, path: str) -> None:
    """Serialize an object to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """Evaluate a classifier model on test set and return metric dictionary."""
    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, pos_label="Positive", zero_division=0))
    recall = float(recall_score(y_test, y_pred, pos_label="Positive", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label="Positive", zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Positive"]).tolist()
    report = classification_report(y_test, y_pred)

    print("\n" + "=" * 50)
    print(f"EVALUATION: {name.upper()}")
    print("=" * 50)
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print("Labels: [Negative, Positive]")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return {
        "name": name,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "raw_accuracy": accuracy,
        "raw_precision": precision,
        "raw_recall": recall,
        "raw_f1_score": f1,
    }


def train_and_evaluate() -> None:
    """Full training pipeline: preprocess, vectorize, train NB & LR, evaluate, save."""
    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)
    print(f"Loaded {len(df)} reviews.")

    print("Preprocessing reviews...")
    df["Cleaned_Review"] = preprocess_corpus(df["Review"])

    X = df["Cleaned_Review"]
    y = df["Sentiment"]

    print("Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("1. Training Multinomial Naive Bayes classifier (Primary)...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_metrics = evaluate_model(nb_model, X_test_tfidf, y_test, "Multinomial Naive Bayes")

    print("2. Training Logistic Regression classifier (Secondary)...")
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train_tfidf, y_train)
    lr_metrics = evaluate_model(lr_model, X_test_tfidf, y_test, "Logistic Regression")

    metrics_payload = {
        "naive_bayes": nb_metrics,
        "logistic_regression": lr_metrics,
        "test_samples": len(y_test),
        "train_samples": len(y_train),
        "vocab_size": len(vectorizer.vocabulary_),
    }

    print("\nSaving models, vectorizer, and metrics JSON...")
    save_pickle(nb_model, NB_MODEL_PATH)
    save_pickle(lr_model, LR_MODEL_PATH)
    save_pickle(vectorizer, VECTORIZER_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"NB Model saved to: {NB_MODEL_PATH}")
    print(f"LR Model saved to: {LR_MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("\nTraining complete!")


if __name__ == "__main__":
    train_and_evaluate()

