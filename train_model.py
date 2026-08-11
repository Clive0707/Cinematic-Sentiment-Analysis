"""
Train Multinomial Naive Bayes sentiment classifier on Marathi movie reviews.

Reads the dataset, preprocesses text, vectorizes with TF-IDF, trains the model,
evaluates performance, and saves artifacts to the models/ directory.
"""

import os
import pickle

import pandas as pd
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
MODEL_PATH = os.path.join("models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

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


def train_and_evaluate() -> None:
    """Full training pipeline: preprocess, vectorize, train, evaluate, save."""
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

    print("Training Multinomial Naive Bayes classifier...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    print("Evaluating on test set...")
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="Positive", zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label="Positive", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="Positive", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Positive"])
    report = classification_report(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print("Labels: [Negative, Positive]")
    print(cm)
    print("\nClassification Report:")
    print(report)

    print("Saving model and vectorizer...")
    save_pickle(model, MODEL_PATH)
    save_pickle(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print("\nTraining complete!")


if __name__ == "__main__":
    train_and_evaluate()
