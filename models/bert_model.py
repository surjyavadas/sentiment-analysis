"""
BERT / RoBERTa Sentiment Analyzer with Graceful Fallback.

Uses HuggingFace cardiffnlp/twitter-roberta-base-sentiment for
Twitter-optimized deep learning sentiment analysis. Falls back
to VADER if the model is not available (no GPU, not downloaded).
"""
import logging
from datetime import datetime, timezone

from utils.preprocessor import clean_text, extract_words

logger = logging.getLogger(__name__)

_pipeline = None
_load_error = None
_label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


def _load_model():
    """Lazy-load the HuggingFace pipeline. Downloads ~500MB on first use."""
    global _pipeline, _load_error

    if _pipeline is not None:
        return _pipeline
    if _load_error is not None:
        return None

    try:
        logger.info("Loading RoBERTa model (this may take a moment)...")
        from transformers import pipeline
        _pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            truncation=True,
            max_length=512,
        )
        logger.info("RoBERTa model loaded successfully.")
        return _pipeline
    except Exception as e:
        _load_error = str(e)
        logger.warning("RoBERTa unavailable (%s). Will fall back to VADER.", e)
        return None


def analyze(text):
    """
    Analyze sentiment using RoBERTa, falling back to VADER if unavailable.

    Args:
        text: Raw input text string.

    Returns:
        Dict matching the unified schema with word_scores array.
    """
    cleaned = clean_text(text)
    model = _load_model()

    if model is None:
        # Graceful fallback to VADER
        from models import vader_model
        result = vader_model.analyze(text)
        result["model"] = "bert (fallback: vader)"
        return result

    try:
        # Run the pipeline — returns top prediction
        truncated = cleaned[:512] if len(cleaned) > 512 else cleaned
        preds = model(truncated, top_k=3)

        # Parse predictions into pos/neg/neu scores
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for p in preds:
            mapped_label = _label_map.get(p["label"], "neutral")
            scores[mapped_label] = round(p["score"], 4)

        # Top label
        label = max(scores, key=scores.get)
        confidence = round(scores[label] * 100, 1)

        # Compound: positive - negative (range -1 to +1 approx)
        compound = round(scores["positive"] - scores["negative"], 4)

        # Word-level scoring: use VADER for word-level (BERT is sentence-level)
        word_scores = _get_word_scores_via_vader(cleaned)

        return {
            "text": text,
            "clean_text": cleaned,
            "label": label,
            "compound": compound,
            "pos": scores["positive"],
            "neg": scores["negative"],
            "neu": scores["neutral"],
            "confidence": confidence,
            "model": "bert",
            "word_scores": word_scores,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error("BERT analysis failed: %s", e)
        from models import vader_model
        result = vader_model.analyze(text)
        result["model"] = "bert (error fallback: vader)"
        return result


def _get_word_scores_via_vader(text):
    """
    Use VADER for word-level scores since BERT operates at sentence level.

    Args:
        text: Cleaned text string.

    Returns:
        List of dicts: [{"word": str, "score": float, "sentiment": str}, ...]
    """
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
    except Exception:
        return []

    words = extract_words(text)
    result = []

    for word in words:
        s = sia.polarity_scores(word)
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


def get_status():
    """Get the current status of the BERT model."""
    if _pipeline is not None:
        return {"status": "loaded", "model": "cardiffnlp/twitter-roberta-base-sentiment"}
    if _load_error is not None:
        return {"status": "fallback", "error": _load_error}
    return {"status": "not_loaded"}
