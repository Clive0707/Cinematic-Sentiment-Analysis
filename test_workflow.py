"""
End-to-end verification script for MarathiSense AI.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from app import app
from predict import get_predictor, predict_sentiment
from pdf_generator import generate_sentiment_pdf

def test_all():
    print("==================================================")
    print("1. TESTING PREDICTOR UTILITY (predict.py)")
    print("==================================================")
    sample_review = "चित्रपटाची कथा खूप छान होती पण शेवट थोडा कमजोर वाटला."
    result = predict_sentiment(sample_review)
    
    print("Review:", result["review_text"])
    print("\n--- Naive Bayes Prediction ---")
    print("Sentiment :", result["naive_bayes"]["sentiment"])
    print("Confidence:", result["naive_bayes"]["confidence"], "%")
    print("Role      :", result["naive_bayes"]["role"])
    
    print("\n--- Logistic Regression Prediction ---")
    print("Sentiment :", result["logistic_regression"]["sentiment"])
    print("Confidence:", result["logistic_regression"]["confidence"], "%")
    print("Role      :", result["logistic_regression"]["role"])
    
    print("\n--- Model Agreement ---")
    print("Status  :", result["agreement"]["status_label"])
    print("Message :", result["agreement"]["message"])
    
    print("\n--- Top 3 Similar Reviews (Cosine Similarity) ---")
    for i, sim in enumerate(result["similar_reviews"], 1):
        print(f"#{i} [{sim['similarity']}% match | {sim['sentiment']}]: {sim['review'][:60]}...")
        
    print("==================================================")
    print("2. TESTING PDF GENERATOR (pdf_generator.py)")
    print("==================================================")
    pdf_bytes = generate_sentiment_pdf(result)
    print(f"Successfully generated PDF report ({len(pdf_bytes)} bytes)")
    
    print("==================================================")
    print("3. TESTING FLASK APP ENDPOINTS (app.py)")
    print("==================================================")
    client = app.test_client()
    
    # Test GET /
    resp_get = client.get("/")
    assert resp_get.status_code == 200, f"GET / failed: {resp_get.status_code}"
    print("GET / -> 200 OK")
    
    # Test POST /
    resp_post = client.post("/", data={"review": sample_review})
    assert resp_post.status_code == 200, f"POST / failed: {resp_post.status_code}"
    assert b"Multinomial Naive Bayes" in resp_post.data, "MNB not found in HTML"
    assert b"Logistic Regression" in resp_post.data, "LR not found in HTML"
    print("POST / -> 200 OK (HTML contains both MNB & LR predictions)")
    
    # Test POST /predict API
    resp_api = client.post("/predict", json={"review": sample_review})
    assert resp_api.status_code == 200, f"POST /predict failed: {resp_api.status_code}"
    json_data = resp_api.get_json()
    assert "naive_bayes" in json_data and "logistic_regression" in json_data
    print("POST /predict -> 200 OK (JSON API returning dual predictions)")
    
    # Test POST /download-pdf
    resp_pdf = client.post("/download-pdf", data={"review": sample_review})
    assert resp_pdf.status_code == 200, f"POST /download-pdf failed: {resp_pdf.status_code}"
    assert resp_pdf.mimetype == "application/pdf"
    print(f"POST /download-pdf -> 200 OK (Served PDF file of size {len(resp_pdf.data)} bytes)")
    
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_all()
