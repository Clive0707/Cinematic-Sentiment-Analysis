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
    print("Pos Prob  :", result["naive_bayes"]["pos_prob"], "%")
    print("Neg Prob  :", result["naive_bayes"]["neg_prob"], "%")
    print("Pos Features:", [f['word'] for f in result["naive_bayes"]["positive_features"]])
    print("Neg Features:", [f['word'] for f in result["naive_bayes"]["negative_features"]])

    print("\n--- Logistic Regression Prediction ---")
    print("Sentiment :", result["logistic_regression"]["sentiment"])
    print("Confidence:", result["logistic_regression"]["confidence"], "%")
    print("Pos Prob  :", result["logistic_regression"]["pos_prob"], "%")
    print("Neg Prob  :", result["logistic_regression"]["neg_prob"], "%")
    print("Pos Features:", [f['word'] for f in result["logistic_regression"]["positive_features"]])
    print("Neg Features:", [f['word'] for f in result["logistic_regression"]["negative_features"]])

    # Assert probability sum equals 100%
    nb_sum = round(result["naive_bayes"]["pos_prob"] + result["naive_bayes"]["neg_prob"], 1)
    lr_sum = round(result["logistic_regression"]["pos_prob"] + result["logistic_regression"]["neg_prob"], 1)
    assert abs(nb_sum - 100.0) <= 0.1, f"NB probabilities do not sum to 100%: {nb_sum}"
    assert abs(lr_sum - 100.0) <= 0.1, f"LR probabilities do not sum to 100%: {lr_sum}"
    print("\n✓ Probability Distribution Sum Verification: Passed (Pos % + Neg % = 100.0%)")

    print("\n--- Model Agreement & Difference ---")
    print("Status  :", result["agreement"]["status_label"])
    print("Message :", result["agreement"]["message"])
    print("Diff    :", result["agreement"]["diff_text"])

    print("\n--- Testing Critical Negation Case: 'हा चित्रपट वाईट नाही.' ---")
    neg_case = "हा चित्रपट वाईट नाही."
    res_neg = predict_sentiment(neg_case)
    print("Review:", res_neg["review_text"])
    print("MNB Prediction:", res_neg["naive_bayes"]["sentiment"], f"(Pos {res_neg['naive_bayes']['pos_prob']}%, Neg {res_neg['naive_bayes']['neg_prob']}%)")
    print("LR  Prediction:", res_neg["logistic_regression"]["sentiment"], f"(Pos {res_neg['logistic_regression']['pos_prob']}%, Neg {res_neg['logistic_regression']['neg_prob']}%)")
    print("Model Agreement:", res_neg["agreement"]["status_label"])
    assert res_neg["naive_bayes"]["sentiment"] == "Positive", f"MNB failed on '{neg_case}'"
    assert res_neg["logistic_regression"]["sentiment"] == "Positive", f"LR failed on '{neg_case}'"
    print("✓ Critical Negation Case Passed: Both models naturally predict Positive!")

    print("\n--- Testing Multiple Review Types ---")
    test_reviews = [
        ("Strong Positive", "चित्रपटाची कथा खूप छान होती, दिग्दर्शन आणि अभिनय उत्तम झाला आहे. सर्वांनी नक्की पाहावा असा चित्रपट!"),
        ("Strong Negative", "इतका रटाळ चित्रपट खूप दिवसांनी पाहिला. वेळेचा आणि पैशांचा पूर्णपणे अपव्यय झाला. अजिबात आवडला नाही."),
        ("Mixed Review", "चित्रपटाची कथा खूप छान होती पण शेवट थोडा कमजोर वाटला."),
        ("Short Review", "उत्कृष्ट चित्रपट"),
        ("Negation Positive 2", "कथा अजिबात कंटाळवाणी नाही."),
        ("Negation Positive 3", "चित्रपटाने निराश केले नाही."),
        ("Negation Negative 1", "हा चित्रपट चांगला नाही."),
        ("Negation Negative 2", "अभिनय उत्कृष्ट वाटला नाही."),
        ("Contrast Hard Case", "सुरुवात थोडी कंटाळवाणी आहे, पण पुढे कथा इतकी सुंदर उलगडते आणि क्लायमॅक्स थक्क करून सोडतो."),
    ]

    for label, rev in test_reviews:
        r = predict_sentiment(rev)
        nb_p = round(r["naive_bayes"]["pos_prob"] + r["naive_bayes"]["neg_prob"], 1)
        lr_p = round(r["logistic_regression"]["pos_prob"] + r["logistic_regression"]["neg_prob"], 1)
        assert abs(nb_p - 100.0) <= 0.1
        assert abs(lr_p - 100.0) <= 0.1
        print(f"  - [{label}] MNB: {r['naive_bayes']['sentiment']} ({r['naive_bayes']['confidence']}%), LR: {r['logistic_regression']['sentiment']} ({r['logistic_regression']['confidence']}%) -> Prob Sum 100% OK")

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
    assert b"Probability Distribution" in resp_post.data, "Probability Distribution not found in HTML"
    assert b"Why this confidence score?" in resp_post.data, "Why this score not found in HTML"
    print("POST / -> 200 OK (HTML contains MNB, LR, probability distributions, & explainability)")

    # Test POST /predict API
    resp_api = client.post("/predict", json={"review": sample_review})
    assert resp_api.status_code == 200, f"POST /predict failed: {resp_api.status_code}"
    json_data = resp_api.get_json()
    assert "naive_bayes" in json_data and "logistic_regression" in json_data
    assert "explanation" in json_data["naive_bayes"] and "positive_features" in json_data["naive_bayes"]
    print("POST /predict -> 200 OK (JSON API returning explainability & feature evidence)")

    # Test POST /download-pdf
    resp_pdf = client.post("/download-pdf", data={"review": sample_review})
    assert resp_pdf.status_code == 200, f"POST /download-pdf failed: {resp_pdf.status_code}"
    assert resp_pdf.mimetype == "application/pdf"
    print(f"POST /download-pdf -> 200 OK (Served PDF file of size {len(resp_pdf.data)} bytes)")

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_all()
