"""
VADER Sentiment Analyzer with Word-Level Scoring.

Uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner),
specifically tuned for social media text. Provides per-word sentiment
scores for the inline text highlighting feature.
"""
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timezone

from utils.preprocessor import clean_text, extract_words

# Auto-download VADER lexicon
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# Global singleton
_sia = SentimentIntensityAnalyzer()


def analyze(text):
    """
    Analyze sentiment of text using VADER.

    Args:
        text: Raw input text string.

    Returns:
        Dict matching the unified schema with word_scores array.
    """
    cleaned = clean_text(text)
    scores = _sia.polarity_scores(cleaned)
    compound = scores["compound"]

    # Classify
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    # Confidence: scale compound to 0-100
    confidence = round(abs(compound) * 100, 1)

    # Word-level scoring
    word_scores = _get_word_scores(cleaned)

    return {
        "text": text,
        "clean_text": cleaned,
        "label": label,
        "compound": round(compound, 4),
        "pos": round(scores["pos"], 4),
        "neg": round(scores["neg"], 4),
        "neu": round(scores["neu"], 4),
        "confidence": confidence,
        "model": "vader",
        "word_scores": word_scores,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _get_word_scores(text):
    """
    Run VADER on each individual word to get per-word sentiment.

    Args:
        text: Cleaned text string.

    Returns:
        List of dicts: [{"word": str, "score": float, "sentiment": str}, ...]
    """
    words = extract_words(text)
    result = []

    for word in words:
        s = _sia.polarity_scores(word)
        comp = s["compound"]

        if comp >= 0.05:
            sentiment = "positive"
        elif comp <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        result.append({
            "word": word,
            "score": round(comp, 4),
            "sentiment": sentiment,
        })

    return result
