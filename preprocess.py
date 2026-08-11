"""
Text preprocessing utilities for Marathi movie review sentiment analysis.

Handles cleaning of Marathi (Devanagari) text while preserving Unicode characters,
removing noise (punctuation, numbers, special characters), and filtering stopwords.
"""

import re

# Devanagari Unicode block (U+0900–U+097F) covers Marathi script
DEVANAGARI_PATTERN = re.compile(r"[^\u0900-\u097F\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Common Marathi stopwords (function words with little sentiment signal)
MARATHI_STOPWORDS = {
    "आणि", "किंवा", "पण", "तर", "म्हणून", "कारण", "जर", "तरी", "म्हणजे",
    "ही", "हे", "हा", "हो", "होते", "होता", "होती", "आहे", "आहेत", "नाही",
    "नाहीत", "असे", "असा", "अशी", "अस", "तो", "ती", "ते", "त्या", "त्याचा",
    "त्याची", "त्याचे", "त्यां", "त्यांना", "त्याने", "त्यांनी", "मी", "माझा",
    "माझी", "माझे", "तू", "तुम्ही", "तुमचा", "तुमची", "तुमचे", "आपण", "आपला",
    "आपली", "आपले", "आम्ही", "आमचा", "आमची", "आमचे", "त्याला", "त्याला",
    "येथे", "तेथे", "कुठे", "कधी", "कसे", "कशी", "काय", "कोण", "कोणता", "कोणती",
    "कोणते", "किती", "खूप", "फार", "अजून", "आता", "मग", "नंतर", "आधी", "पुढे",
    "मागे", "वर", "खाली", "मध्ये", "बाहेर", "आत", "साठी", "पासून", "पर्यंत",
    "विषयी", "बद्दल", "शिवाय", "सुद्धा", "देखील", "फक्त", "केवळ",
    "अर्थात", "म्हणाल", "म्हणाला", "म्हणाली", "म्हणाले", "असले", "असलेला",
    "असलेली", "असलेले", "असल्या", "असून", "असत", "असतो", "असते", "असती",
    "असतात", "होणार", "होणारा", "होणारी", "होणारे", "येणार", "येणारा", "येणारी",
    "येणारे", "करणार", "करणारा", "करणारी", "करणारे", "गेला", "गेली", "गेले",
    "आला", "आली", "आले", "केला", "केली", "केले", "दिला", "दिली", "दिले",
    "झाला", "झाली", "झाले", "वाटते", "वाटत", "वाटतो", "वाटती", "वाटतात",
    "हवा", "हवी", "हवे", "पाहिजे", "शकत", "शकते", "शकतो", "शकती", "शकतात",
    "चा", "ची", "चे", "च्या", "नी", "ने", "ला", "एक", "दोन", "तीन", "चार", "पाच",
}


def preprocess_text(text: str) -> str:
    """
    Clean and normalize a Marathi review string for ML feature extraction.

    Steps:
        1. Convert to string and lowercase (handles mixed English tokens).
        2. Remove punctuation, numbers, and non-Devanagari special characters.
        3. Collapse extra whitespace.
        4. Remove Marathi stopwords and very short tokens.

    Args:
        text: Raw review text.

    Returns:
        Preprocessed text ready for TF-IDF vectorization.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # Strip everything except Devanagari letters and whitespace
    text = DEVANAGARI_PATTERN.sub(" ", text)

    # Normalize whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    # Remove stopwords and single-character tokens
    words = [word for word in text.split() if word not in MARATHI_STOPWORDS and len(word) > 1]

    return " ".join(words)


def preprocess_corpus(texts) -> list:
    """
    Apply preprocessing to a collection of review texts.

    Args:
        texts: Iterable of raw review strings.

    Returns:
        List of preprocessed strings.
    """
    return [preprocess_text(text) for text in texts]
