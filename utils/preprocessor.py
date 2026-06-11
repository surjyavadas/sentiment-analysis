"""
Text Preprocessing Pipeline for SentimentAI.

Provides a full cleaning pipeline for social media text:
- URL removal
- Mention removal
- Hashtag cleaning
- Emoji handling
- Whitespace normalization
- HTML entity decoding
"""
import re
import html


def clean_text(text):
    """
    Clean and normalize social media text for NLP analysis.

    Args:
        text: Raw text string from a tweet or social media post.

    Returns:
        Cleaned text string ready for sentiment analysis.
    """
    if not text or not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove @mentions but keep the text after @
    text = re.sub(r'@(\w+)', r'\1', text)

    # Remove # symbol but keep the hashtag word
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove RT prefix
    text = re.sub(r'^RT\s+', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_words(text):
    """
    Extract individual words from text for word-level scoring.

    Args:
        text: Cleaned text string.

    Returns:
        List of word strings (preserving order, filtering punctuation-only tokens).
    """
    if not text:
        return []

    # Split on whitespace and filter out pure punctuation
    tokens = text.split()
    words = []
    for token in tokens:
        # Keep words that have at least one alphanumeric character
        cleaned = re.sub(r'[^\w\'-]', '', token)
        if cleaned and any(c.isalnum() for c in cleaned):
            words.append(token)  # Keep original form for display

    return words


def truncate_text(text, max_length=280):
    """
    Truncate text to a maximum length, adding ellipsis if needed.

    Args:
        text: Text string to truncate.
        max_length: Maximum character length.

    Returns:
        Truncated text with '...' appended if shortened.
    """
    if not text or len(text) <= max_length:
        return text or ""
    return text[:max_length - 3] + "..."
