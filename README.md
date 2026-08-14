# Marathi Movie Review Sentiment Analysis

A comprehensive dual-supervised machine learning web application for **Marathi movie review sentiment analysis**, built using Python, Flask, TF-IDF vectorization, **Multinomial Naive Bayes**, and **Logistic Regression**.

---

## 📷 Screenshots & Interface Showcase

### 1. Review Input & Main Dashboard
Enter Marathi movie reviews or select sample review presets to initiate analysis.

![Review Input Dashboard](docs/images/review_input.png)

---

### 2. Dual Independent Predictions & Model Agreement
Independent predictions from **Multinomial Naive Bayes** (Primary) and **Logistic Regression** (Secondary) alongside model confidence percentages and agreement status.

![Dual Model Predictions](docs/images/dual_model_predictions.png)

---

### 3. Model Performance Benchmark Comparison
Interactive comparison view displaying test set evaluation metrics (Accuracy, Precision, Recall, F1 Score) and visual metric charts.

![Model Performance Comparison](docs/images/model_performance_comparison.png)

---

### 4. Downloadable PDF Analysis Report
Export complete sentiment analysis reports in PDF format with native Devanagari font rendering (`Nirmala.ttc`).

![PDF Analysis Report](docs/images/pdf_analysis_report.png)

---

## 📁 Project Structure

```
project/
│
├── dataset/
│   └── marathi_movie_reviews.csv       # 1,000 Marathi movie reviews dataset
│
├── docs/
│   └── images/                         # Dashboard & PDF screenshots
│       ├── review_input.png
│       ├── dual_model_predictions.png
│       ├── model_performance_comparison.png
│       └── pdf_analysis_report.png
│
├── models/
│   ├── sentiment_model.pkl             # Multinomial Naive Bayes model
│   ├── logistic_regression_model.pkl   # Logistic Regression model
│   ├── tfidf_vectorizer.pkl            # TF-IDF vectorizer
│   └── metrics.json                    # Saved test set evaluation metrics
│
├── templates/
│   └── index.html                      # Cinematic dark mode dashboard
│
├── static/
│   ├── style.css                       # Responsive CSS styles & glassmorphism
│   └── script.js                       # Tab navigation & frontend interaction
│
├── preprocess.py                       # Devanagari text cleaning & Marathi stopwords
├── train_model.py                      # Model training & evaluation pipeline
├── predict.py                         # Independent dual-model prediction engine
├── pdf_generator.py                    # PDF report builder using ReportLab
├── app.py                             # Flask web app & API endpoints
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

---

## 📊 Dataset Overview

- **File:** `dataset/marathi_movie_reviews.csv`
- **Rows:** 1,000
- **Columns:** `Review`, `Sentiment`
- **Classes:** Positive, Negative

---

## ⚙️ Setup & Installation

1. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models (creates pickles in `models/`):**

   ```bash
   python train_model.py
   ```

   This pipeline will:
   - Clean Marathi text (preserve Devanagari Unicode, remove noise & stopwords)
   - Vectorize text using TF-IDF
   - Perform an 80/20 stratified split (`random_state=42`)
   - Train **Multinomial Naive Bayes** (`sentiment_model.pkl`)
   - Train **Logistic Regression** (`logistic_regression_model.pkl`)
   - Evaluate Accuracy, Precision, Recall, F1 Score on test set
   - Save metrics to `models/metrics.json`

4. **Run the Flask web server:**

   ```bash
   python app.py
   ```

5. **Open in browser:**

   ```
   http://127.0.0.1:5000
   ```

---

## 🚀 Key Features

1. **Dual Independent Classification**:
   - **Multinomial Naive Bayes** (Primary Classifier)
   - **Logistic Regression** (Secondary Classifier)
   - Predictions and confidence scores derived strictly from `predict_proba()`.

2. **Model Agreement Indicator**:
   - Automatically determines whether both classifiers agree (`✓ Both models agree`) or produce conflicting predictions (`⚠ Models disagree`).

3. **TF-IDF Cosine Similarity Matches**:
   - NLP feature extracting top 3 dataset review matches with similarity percentages.

4. **Professional PDF Reports**:
   - One-click PDF generation featuring native Devanagari font rendering.

5. **Cinematic Dark Mode Dashboard**:
   - Responsive tabbed navigation (**Analyze**, **Model Comparison**, **About the Model**).

---

## 📦 Modules Description

| File | Description |
|------|-------------|
| `preprocess.py` | Marathi Devanagari text normalization and stopword filtering |
| `train_model.py` | Training & benchmark evaluation for Naive Bayes and Logistic Regression |
| `predict.py` | Dual model prediction engine and TF-IDF Cosine Similarity retrieval |
| `pdf_generator.py` | PDF document generation using ReportLab |
| `app.py` | Flask web server, API routes, and PDF download endpoint |

---

## 📜 License

MIT License
