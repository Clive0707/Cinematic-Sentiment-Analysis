"""
Text preprocessing utilities for Marathi movie review sentiment analysis.

Handles cleaning of Marathi (Devanagari) text while preserving Unicode characters,
removing noise (punctuation, numbers, special characters), and filtering stopwords.
"""

import re

# Devanagari Unicode block (U+0900–U+097F) covers Marathi script
DEVANAGARI_PATTERN = re.compile(r"[^\u0900-\u097F\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Common Marathi negation words and structures
MARATHI_NEGATION_WORDS = {
    "नाही", "नाहीत", "नव्हता", "नव्हती", "नव्हते", "नव्हतं",
    "नसतो", "नसते", "नसलेला", "नसलेली", "नसलेले", "नसून", "नको"
}

# Auxiliary / connector verbs that frequently link sentiment predicates with negation
MARATHI_AUX_VERBS = {
    "वाटला", "वाटले", "वाटली", "वाटतो", "वाटते", "वाटतात",
    "केले", "केला", "केली", "झाला", "झाली", "झाले", "ठरला", "ठरली", "ठरले"
}

# Contrastive conjunctions and transition words that reverse or modulate sentiment
MARATHI_CONTRAST_WORDS = {
    "पण", "मात्र", "तरी", "उलट", "परंतु", "तरीही"
}

# Common Marathi stopwords (excluding negation particles, auxiliary connectors, and contrast markers)
RAW_STOPWORDS = {
    "आणि", "किंवा", "तर", "म्हणून", "कारण", "जर", "म्हणजे",
    "ही", "हे", "हा", "हो", "होता", "होती", "आहे", "आहेत",
    "असे", "असा", "अशी", "अस", "तो", "ती", "ते", "त्या", "त्याचा",
    "त्याची", "त्याचे", "त्यां", "त्यांना", "त्याने", "त्यांनी", "मी", "माझा",
    "माझी", "माझे", "तू", "तुम्ही", "तुमचा", "तुमची", "तुमचे", "आपण", "आपला",
    "आपली", "आपले", "आम्ही", "आमचा", "आमची", "आमचे", "त्याला",
    "येथे", "तेथे", "कुठे", "कधी", "कसे", "कशी", "काय", "कोण", "कोणता", "कोणती",
    "कोणते", "किती", "खूप", "फार", "अजून", "आता", "मग", "नंतर", "आधी", "पुढे",
    "मागे", "वर", "खाली", "मध्ये", "बाहेर", "आत", "साठी", "पासून", "पर्यंत",
    "विषयी", "बद्दल", "शिवाय", "सुद्धा", "देखील", "फक्त", "केवळ",
    "अर्थात", "म्हणाल", "म्हणाला", "म्हणाली", "म्हणाले", "असले",
    "असल्या", "असत", "होणार", "होणारा", "होणारी", "होणारे",
    "येणार", "येणारा", "येणारी", "येणारे", "करणार", "करणारा", "करणारी", "करणारे",
    "गेला", "गेली", "गेले", "आला", "आली", "आले", "दिला", "दिली", "दिले",
    "हवा", "हवी", "हवे", "पाहिजे", "शकत", "शकते", "शकतो", "शकती", "शकतात",
    "चा", "ची", "चे", "च्या", "नी", "ने", "ला", "एक", "दोन", "तीन", "चार", "पाच",
}

MARATHI_STOPWORDS = RAW_STOPWORDS - MARATHI_NEGATION_WORDS - MARATHI_AUX_VERBS - MARATHI_CONTRAST_WORDS

# Phrase-aware normalized features mapped from Marathi negation expressions
PHRASE_NORMALIZATION_PATTERNS = [
    (re.compile(r"(?:वाईट|खराब|कमकुवत|फिका)\s+(?:नाही|नव्हता|नव्हती|नव्हते|नव्हतं|नसतो|नसते|नसलेला|नसलेली|नसलेले|नको)", re.IGNORECASE), "NOT_BAD"),
    (re.compile(r"(?:निराश|निराशाजनक)\s+(?:केले\s+)?(?:नाही|नव्हता|नव्हती|नव्हते|नव्हतं|नसतो|नसते|नको)", re.IGNORECASE), "NOT_DISAPPOINTING"),
    (re.compile(r"(?:कंटाळवाणा|कंटाळवाणी|कंटाळवाणे|कंटाळवाण|बोअर)\s+(?:नाही|नव्हता|नव्हती|नव्हते|नव्हतं|नसतो|नसते|नको)", re.IGNORECASE), "NOT_BORING"),
    (re.compile(r"(?:प्रभावी|सुरेख)\s+(?:नाही|नव्हता|नव्हती|नव्हते|नव्हतं|नसतो|नसते|नको)", re.IGNORECASE), "NOT_EFFECTIVE"),
    (re.compile(r"(?:चांगला|चांगली|चांगले|छान)\s+(?:नाही|नव्हता|नव्हती|नव्हते|नव्हतं|नसतो|नसते|नको)", re.IGNORECASE), "NOT_GOOD"),
    (re.compile(r"(?:उत्कृष्ट|अप्रतिम|सुंदर)\s+(?:वाटला\s+|वाटले\s+|वाटली\s+)?(?:नाही|नव्हता|नव्हती|नव्हते|नव्हतं|नसतो|नसते|नको)", re.IGNORECASE), "NOT_EXCELLENT"),
    (re.compile(r"(?:आवडले|आवडला|आवडली)\s+(?:नाही|नव्हता|नव्हती|नव्हते|नव्हतं|नसतो|नसते|नको)", re.IGNORECASE), "NOT_LIKED"),
    (re.compile(r"(?:समाधानकारक)\s+(?:नाही|नव्हता|नव्हती|नव्हते|नव्हतं|नसतो|नसते|नको)", re.IGNORECASE), "NOT_SATISFACTORY"),
]


def contains_negation(text: str) -> bool:
    """Return True if text contains any recognized Marathi negation words."""
    if not isinstance(text, str):
        return False
    words = set(text.split())
    return any(w in MARATHI_NEGATION_WORDS for w in words) or any(
        neg in text for neg in ["नाही", "नव्हता", "नव्हती", "नव्हते", "नव्हतं", "नको", "नसून"]
    )


def contains_contrast(text: str) -> bool:
    """Return True if text contains any recognized Marathi contrastive conjunctions."""
    if not isinstance(text, str):
        return False
    words = set(text.split())
    return any(w in MARATHI_CONTRAST_WORDS for w in words) or any(
        cw in text for cw in ["पण", "मात्र", "तरी", "उलट", "परंतु", "तरीही"]
    )


def is_mixed_sentiment(text: str) -> bool:
    """Check if text contains indicators of mixed sentiment or contrast structures."""
    return contains_contrast(text) or (contains_negation(text) and len(text.split()) > 6)


def preprocess_text(text: str) -> str:
    """
    Clean, normalize, and tag Marathi review string for negation-aware ML feature extraction.

    Steps:
        1. Convert to string and lowercase.
        2. Detect phrase-aware negation expressions and map them to normalized feature tokens
           (e.g., 'वाईट नाही' -> 'NOT_BAD', 'चांगला नाही' -> 'NOT_GOOD').
        3. Remove punctuation, numbers, and non-Devanagari characters.
        4. Normalize whitespace and filter general stopwords (preserving negation, aux verbs & contrast words).
        5. Apply smart predicate tagging: tag preceding content words or auxiliary-connected predicates
           with '_NEG' to preserve the sentiment inversion relationship.
        6. Combine the original Marathi words/bigrams with the phrase-normalized tokens.

    Args:
        text: Raw review text.

    Returns:
        Preprocessed, negation-aware text ready for word and character TF-IDF vectorization.
    """
    if not isinstance(text, str):
        text = str(text)

    raw_lower = text.lower()

    # Step 1: Detect phrase-level normalized feature tokens
    phrase_tokens = []
    for pattern, feature_token in PHRASE_NORMALIZATION_PATTERNS:
        if pattern.search(raw_lower):
            phrase_tokens.append(feature_token)

    # Step 2: Strip non-Devanagari letters and normalize whitespace
    cleaned = DEVANAGARI_PATTERN.sub(" ", raw_lower)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()

    # Step 3: Tokenize and filter general stopwords
    tokens = [word for word in cleaned.split() if word not in MARATHI_STOPWORDS and len(word) > 1]

    # Step 4: Negation-aware predicate tagging
    processed_tokens = []
    for i, word in enumerate(tokens):
        if word in MARATHI_NEGATION_WORDS:
            if i > 0:
                prev = tokens[i - 1]
                if prev in MARATHI_AUX_VERBS and i > 1:
                    prev2 = tokens[i - 2]
                    processed_tokens.append(prev2 + "_NEG")
                    processed_tokens.append(prev + "_NEG")
                else:
                    processed_tokens.append(prev + "_NEG")
            processed_tokens.append(word)
        else:
            processed_tokens.append(word)

    # Step 5: Append normalized phrase tokens alongside original Marathi text
    all_features = processed_tokens + phrase_tokens
    return " ".join(all_features)


def preprocess_corpus(texts) -> list:
    """
    Apply preprocessing to a collection of review texts.

    Args:
        texts: Iterable of raw review strings.

    Returns:
        List of preprocessed strings.
    """
    return [preprocess_text(text) for text in texts]


