"""
Train Multinomial Naive Bayes and Logistic Regression sentiment classifiers on Marathi movie reviews.

Reads the dataset, preprocesses text, vectorizes with TF-IDF, trains both models,
evaluates performance, and saves model artifacts and metrics to the models/ directory.
"""

import json
import os
import pickle
import sys

# Ensure UTF-8 output encoding for Marathi text in console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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

from scipy.sparse import hstack
from preprocess import (
    contains_contrast,
    contains_negation,
    is_mixed_sentiment,
    preprocess_corpus,
    preprocess_text,
)

# Paths relative to project root
DATASET_PATH = os.path.join("dataset", "marathi_movie_reviews_6000_negation_hardcases.csv")
NB_MODEL_PATH = os.path.join("models", "sentiment_model.pkl")
LR_MODEL_PATH = os.path.join("models", "logistic_regression_model.pkl")
WORD_VECTORIZER_PATH = os.path.join("models", "word_vectorizer.pkl")
CHAR_VECTORIZER_PATH = os.path.join("models", "char_vectorizer.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join("models", "metrics.json")

# Reproducible split (80% train / 20% test)
RANDOM_STATE = 42
TEST_SIZE = 0.20


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
        "confusion_matrix": cm,
        "raw_accuracy": accuracy,
        "raw_precision": precision,
        "raw_recall": recall,
        "raw_f1_score": f1,
    }


def run_diagnostic_tests(nb_model, lr_model, word_vec, char_vec) -> list:
    """Run diagnostic negation, contrast, and mixed-sentiment test cases to observe model behavior."""
    test_cases = [
        {"text": "हा चित्रपट वाईट नाही.", "expected": "Positive", "type": "Negation (Positive)"},
        {"text": "कथा अजिबात कंटाळवाणी नाही.", "expected": "Positive", "type": "Negation (Positive)"},
        {"text": "चित्रपटाने निराश केले नाही.", "expected": "Positive", "type": "Negation (Positive)"},
        {"text": "हा चित्रपट चांगला नाही.", "expected": "Negative", "type": "Negation (Negative)"},
        {"text": "अभिनय उत्कृष्ट वाटला नाही.", "expected": "Negative", "type": "Negation (Negative)"},
        {"text": "शेवट समाधानकारक नव्हता.", "expected": "Negative", "type": "Negation (Negative)"},
        {"text": "कथा सुंदर आहे पण पटकथा थोडी संथ आहे.", "expected": "Positive", "type": "Mixed Sentiment"},
        {
            "text": "सुरुवात थोडी कंटाळवाणी आहे, पण पुढे कथा इतकी सुंदर उलगडते आणि क्लायमॅक्स थक्क करून सोडतो.",
            "expected": "Positive",
            "type": "Contrast (Negative -> Positive)",
        },
    ]

    results = []
    classes = list(nb_model.classes_)
    pos_idx = classes.index("Positive")

    print("\n" + "=" * 50)
    print("DIAGNOSTIC NEGATION & HARD-CASE TESTS")
    print("=" * 50)

    for item in test_cases:
        cleaned = preprocess_text(item["text"])
        w_feat = word_vec.transform([cleaned])
        c_feat = char_vec.transform([cleaned])
        comb_feat = hstack([w_feat, c_feat]).tocsr()

        nb_p = nb_model.predict(comb_feat)[0]
        nb_prob = nb_model.predict_proba(comb_feat)[0]
        nb_pos_prob = round(float(nb_prob[pos_idx]) * 100, 1)

        lr_p = lr_model.predict(comb_feat)[0]
        lr_prob = lr_model.predict_proba(comb_feat)[0]
        lr_pos_prob = round(float(lr_prob[pos_idx]) * 100, 1)

        results.append({
            "text": item["text"],
            "expected": item["expected"],
            "type": item["type"],
            "nb_prediction": nb_p,
            "nb_pos_prob": nb_pos_prob,
            "lr_prediction": lr_p,
            "lr_pos_prob": lr_pos_prob,
        })

        print(f"Case: \"{item['text']}\" (Expected: {item['expected']})")
        print(f"  -> NB: {nb_p} (Pos: {nb_pos_prob}%, Neg: {100-nb_pos_prob:.1f}%)")
        print(f"  -> LR: {lr_p} (Pos: {lr_pos_prob}%, Neg: {100-lr_pos_prob:.1f}%)")

    return results


def train_and_evaluate() -> None:
    """Full training pipeline: preprocess, word+char TF-IDF, train NB & LR, evaluate, error analysis, save."""
    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)
    total_samples = len(df)
    pos_samples = int((df["Sentiment"] == "Positive").sum())
    neg_samples = int((df["Sentiment"] == "Negative").sum())

    print(f"Dataset size: {total_samples}")
    print(f"Positive samples: {pos_samples}")
    print(f"Negative samples: {neg_samples}")

    print("\nPreprocessing reviews with negation and contrast awareness...")
    df["Cleaned_Review"] = preprocess_corpus(df["Review"])

    X = df["Cleaned_Review"]
    y = df["Sentiment"]

    print(f"\nSplitting data ({int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    print("\nTF-IDF Configuration:")
    print("Word n-grams: 1–2")
    print("Character n-grams: 3–5")

    # 1. Word-level TF-IDF (1-2 grams)
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=1,
    )
    X_train_word = word_vectorizer.fit_transform(X_train)
    X_test_word = word_vectorizer.transform(X_test)
    word_vocab_size = len(word_vectorizer.vocabulary_)
    print(f"Word TF-IDF vocabulary size (1–2 grams): {word_vocab_size:,}")

    # 2. Character-level TF-IDF (3-5 char_wb)
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=15000,
        sublinear_tf=True,
    )
    X_train_char = char_vectorizer.fit_transform(X_train)
    X_test_char = char_vectorizer.transform(X_test)
    char_vocab_size = len(char_vectorizer.vocabulary_)
    print(f"Character TF-IDF vocabulary size (3–5 char_wb): {char_vocab_size:,}")

    # 3. Combine sparse feature matrices
    X_train_combined = hstack([X_train_word, X_train_char]).tocsr()
    X_test_combined = hstack([X_test_word, X_test_char]).tocsr()
    total_features = X_train_combined.shape[1]
    print(f"Combined sparse feature matrix shape: {X_train_combined.shape}")

    print("\nModels:")
    print("Multinomial Naive Bayes")
    print("Logistic Regression")

    print("\n1. Training Multinomial Naive Bayes classifier (Primary)...")
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_combined, y_train)
    nb_metrics = evaluate_model(nb_model, X_test_combined, y_test, "Multinomial Naive Bayes")

    print("\n2. Training Logistic Regression classifier (Secondary)...")
    lr_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr_model.fit(X_train_combined, y_train)
    lr_metrics = evaluate_model(lr_model, X_test_combined, y_test, "Logistic Regression")

    # Error Analysis on 1,200 Unseen Test Set
    print("\nAnalyzing test set errors...")
    test_indices = y_test.index.tolist()
    nb_preds = nb_model.predict(X_test_combined)
    lr_preds = lr_model.predict(X_test_combined)

    errors = []
    neg_errors = 0
    contrast_errors = 0
    mixed_errors = 0

    for orig_idx, nb_p, lr_p in zip(test_indices, nb_preds, lr_preds):
        actual = df.loc[orig_idx, "Sentiment"]
        if nb_p != actual or lr_p != actual:
            raw_rev = df.loc[orig_idx, "Review"]
            has_neg = contains_negation(raw_rev)
            has_contrast = contains_contrast(raw_rev)
            has_mixed = is_mixed_sentiment(raw_rev)

            if has_neg:
                neg_errors += 1
            if has_contrast:
                contrast_errors += 1
            if has_mixed:
                mixed_errors += 1

            errors.append({
                "review": raw_rev,
                "actual": actual,
                "nb_pred": nb_p,
                "lr_pred": lr_p,
                "contains_negation": has_neg,
                "contains_contrast": has_contrast,
                "is_mixed": has_mixed,
            })

    total_errors = len(errors)
    correct_nb = int((nb_preds == y_test).sum())
    correct_lr = int((lr_preds == y_test).sum())

    print(f"Total test samples: {len(y_test)}")
    print(f"NB Correct predictions: {correct_nb} / {len(y_test)}")
    print(f"LR Correct predictions: {correct_lr} / {len(y_test)}")
    print(f"Total misclassifications in test set: {total_errors} / {len(y_test)}")
    print(f"Negation-related errors: {neg_errors} / {total_errors}")
    print(f"Contrast-related errors: {contrast_errors} / {total_errors}")
    print(f"Mixed-sentiment errors : {mixed_errors} / {total_errors}")

    # Run diagnostic tests on specific negation & contrast sentences
    diagnostic_results = run_diagnostic_tests(nb_model, lr_model, word_vectorizer, char_vectorizer)

    metrics_payload = {
        "naive_bayes": nb_metrics,
        "logistic_regression": lr_metrics,
        "test_samples": len(y_test),
        "train_samples": len(y_train),
        "word_vocab_size": word_vocab_size,
        "char_vocab_size": char_vocab_size,
        "vocab_size": total_features,
        "ngram_range": "Word (1-2), Char (3-5)",
        "dataset_name": os.path.basename(DATASET_PATH),
        "total_dataset_rows": total_samples,
        "error_analysis": {
            "total_test_samples": len(y_test),
            "total_errors": total_errors,
            "negation_errors": neg_errors,
            "contrast_errors": contrast_errors,
            "mixed_errors": mixed_errors,
            "error_samples": errors[:25],
            "diagnostic_cases": diagnostic_results,
        },
    }

    print("\nSaving models, vectorizers, and metrics JSON...")
    save_pickle(nb_model, NB_MODEL_PATH)
    save_pickle(lr_model, LR_MODEL_PATH)
    save_pickle(word_vectorizer, WORD_VECTORIZER_PATH)
    save_pickle(char_vectorizer, CHAR_VECTORIZER_PATH)

    # Save a combined bundle for vectorizer loading convenience
    save_pickle({"word_vec": word_vectorizer, "char_vec": char_vectorizer}, VECTORIZER_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print(f"NB Model saved to: {NB_MODEL_PATH}")
    print(f"LR Model saved to: {LR_MODEL_PATH}")
    print(f"Word Vectorizer saved to: {WORD_VECTORIZER_PATH}")
    print(f"Char Vectorizer saved to: {CHAR_VECTORIZER_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("\nTraining and evaluation pipeline complete!")


if __name__ == "__main__":
    train_and_evaluate()



