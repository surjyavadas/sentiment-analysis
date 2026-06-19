"""
BERT / RoBERTa Sentiment Analyzer — Coming Soon Stub.

This module is a placeholder for the future BERT/RoBERTa integration.
Currently delegates all analysis to VADER to keep the app lightweight
and deployment-friendly (avoids ~500MB transformers download).

Roadmap:
  - BERT/RoBERTa integration
  - Multilingual sentiment support
  - Sarcasm detection layer
"""
import logging

logger = logging.getLogger(__name__)


def analyze(text):
    """
    Stub: delegates to VADER.

    BERT/RoBERTa will be available in a future release. For now,
    this returns VADER results tagged as 'bert (coming soon)'.

    Args:
        text: Raw input text string.

    Returns:
        Dict matching the unified schema with word_scores array.
    """
    from models import vader_model
    result = vader_model.analyze(text)
    result["model"] = "bert (coming soon)"
    logger.info("BERT is a roadmap feature — using VADER fallback.")
    return result


def get_status():
    """Get the current status of the BERT model."""
    return {"status": "roadmap", "message": "BERT/RoBERTa coming in a future release."}
