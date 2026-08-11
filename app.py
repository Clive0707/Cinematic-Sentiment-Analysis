"""
Flask web application for Marathi movie review sentiment analysis.

Loads pre-trained model from disk at startup — does NOT retrain.
"""

from flask import Flask, render_template, request

from predict import get_predictor

app = Flask(__name__)

# Load model and vectorizer once when the app starts
get_predictor()


@app.route("/", methods=["GET", "POST"])
def index():
    """Render home page and handle review prediction."""
    result = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review", "").strip()

        if review_text:
            result = get_predictor().predict(review_text)
        else:
            result = {"error": "Please enter a Marathi movie review before analyzing."}

    return render_template("index.html", result=result, review_text=review_text)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
