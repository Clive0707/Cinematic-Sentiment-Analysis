"""
Flask web application for Marathi movie review sentiment analysis.

Loads pre-trained Multinomial Naive Bayes and Logistic Regression models.
"""

import os
import socket

from flask import Flask, Response, jsonify, render_template, request

from pdf_generator import generate_sentiment_pdf
from predict import get_predictor

app = Flask(__name__)

# Pre-load models, vectorizer, and metrics at application startup
predictor = get_predictor()


@app.route("/", methods=["GET", "POST"])
def index():
    """Render main dashboard with dual model sentiment prediction."""
    result = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review", "").strip()

        if review_text:
            result = predictor.predict(review_text)
        else:
            result = {"error": "Please enter a Marathi movie review before analyzing."}

    # Pass model metrics for model comparison tab
    metrics = predictor.metrics or {}

    return render_template("index.html", result=result, review_text=review_text, metrics=metrics)


@app.route("/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for asynchronous review analysis."""
    data = request.get_json(silent=True) or request.form
    review_text = data.get("review", "").strip()

    if not review_text:
        return jsonify({"error": "Please enter a Marathi movie review before analyzing."}), 400

    result = predictor.predict(review_text)
    return jsonify(result)


@app.route("/download-pdf", methods=["POST", "GET"])
def download_pdf():
    """Generate and download professional PDF analysis report."""
    if request.method == "POST":
        review_text = request.form.get("review", "").strip()
    else:
        review_text = request.args.get("review", "").strip()

    if not review_text:
        # Fallback sample review if none provided
        review_text = "चित्रपटाची कथा खूप छान होती पण शेवट थोडा कमजोर वाटला."

    result = predictor.predict(review_text)
    pdf_bytes = generate_sentiment_pdf(result)

    filename = "Marathi_Movie_Sentiment_Report.pdf"
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": "application/pdf",
        },
    )


if __name__ == "__main__":
    import socket
    port = int(os.environ.get("PORT", 5000))
    # If default port 5000 is already in use by another local process, use 5001
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0 and "PORT" not in os.environ:
            port = 5001

    print(f"Starting Marathi Sentiment Analysis server on http://127.0.0.1:{port}...")
    app.run(debug=True, host="0.0.0.0", port=port)

