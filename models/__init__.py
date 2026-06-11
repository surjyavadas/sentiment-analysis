"""
NLP Sentiment Analysis Models Package.

All three models return the exact same JSON schema:
{
    "text": str,
    "clean_text": str,
    "label": "positive" | "negative" | "neutral",
    "compound": float,
    "pos": float,
    "neg": float,
    "neu": float,
    "confidence": float,
    "model": str,
    "word_scores": [{"word": str, "score": float, "sentiment": str}, ...],
    "timestamp": str (ISO 8601)
}
"""
