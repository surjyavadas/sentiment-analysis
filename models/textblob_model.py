"""
TextBlob Sentiment Analyzer with Word-Level Scoring.

Uses TextBlob's pattern-based sentiment analysis for polarity
and subjectivity. Returns the same unified JSON schema as all models.
"""
from textblob import TextBlob
from datetime import datetime, timezone

from utils.preprocessor import clean_text, extract_words


def analyze(text):
    """
    Analyze sentiment of text using TextBlob.

    Args:
        text: Raw input text string.

    Returns:
        Dict matching the unified schema with word_scores array.
    """
    cleaned = clean_text(text)
    blob = TextBlob(cleaned)
    polarity = blob.sentiment.polarity       # -1 to +1
    subjectivity = blob.sentiment.subjectivity  # 0 to 1

    # Classify
    if polarity > 0.1:
        label = "positive"
    elif polarity < -0.1:
        label = "negative"
    else:
        label = "neutral"

    # Map polarity to pos/neg/neu percentages
    if polarity > 0:
        pos_pct = round(polarity, 4)
        neg_pct = 0.0
        neu_pct = round(1.0 - polarity, 4)
    elif polarity < 0:
        pos_pct = 0.0
        neg_pct = round(abs(polarity), 4)
        neu_pct = round(1.0 - abs(polarity), 4)
    else:
        pos_pct = 0.0
        neg_pct = 0.0
        neu_pct = 1.0

    confidence = round(abs(polarity) * 100, 1)

    # Word-level scoring
    word_scores = _get_word_scores(cleaned)

    return {
        "text": text,
        "clean_text": cleaned,
        "label": label,
        "compound": round(polarity, 4),
        "pos": pos_pct,
        "neg": neg_pct,
        "neu": neu_pct,
        "confidence": confidence,
        "model": "textblob",
        "word_scores": word_scores,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _get_word_scores(text):
    """
    Run TextBlob on each individual word.

    Args:
        text: Cleaned text string.

    Returns:
        List of dicts: [{"word": str, "score": float, "sentiment": str}, ...]
    """
    words = extract_words(text)
    result = []

    for word in words:
        blob = TextBlob(word)
        pol = blob.sentiment.polarity

        if pol > 0.1:
            sentiment = "positive"
        elif pol < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        result.append({
            "word": word,
            "score": round(pol, 4),
            "sentiment": sentiment,
        })

    return result
