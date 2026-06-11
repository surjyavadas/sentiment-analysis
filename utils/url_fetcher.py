"""
URL Fetcher for SentimentAI.

Detects social media platforms from URLs and extracts post text
using free, public APIs:
  - Twitter/X:  publish.twitter.com oEmbed API
  - Reddit:     Public .json API
  - YouTube:    youtube.com oEmbed API
  - Instagram/Facebook/LinkedIn: graceful fallback (auth required)
"""
import re
import logging
import requests

logger = logging.getLogger(__name__)

# Platform color/emoji metadata for frontend display
PLATFORM_META = {
    "twitter":   {"name": "Twitter / X",  "color": "#1DA1F2", "emoji": "🐦", "support": "full"},
    "reddit":    {"name": "Reddit",       "color": "#FF4500", "emoji": "👽", "support": "full"},
    "youtube":   {"name": "YouTube",      "color": "#FF0000", "emoji": "💬", "support": "full"},
    "instagram": {"name": "Instagram",    "color": "#E1306C", "emoji": "📸", "support": "fallback"},
    "facebook":  {"name": "Facebook",     "color": "#1877F2", "emoji": "📘", "support": "fallback"},
    "linkedin":  {"name": "LinkedIn",     "color": "#0A66C2", "emoji": "💼", "support": "fallback"},
    "unknown":   {"name": "Unknown",      "color": "#6B7280", "emoji": "🔗", "support": "none"},
}


class UnsupportedPlatformError(Exception):
    """Raised when a platform requires authentication or is unsupported."""

    def __init__(self, platform, message=None):
        self.platform = platform
        meta = PLATFORM_META.get(platform, PLATFORM_META["unknown"])
        self.message = message or (
            f"{meta['name']} requires authentication to access posts. "
            f"Please copy and paste the post text directly instead."
        )
        super().__init__(self.message)


class FetchError(Exception):
    """Raised when text extraction fails for a supported platform."""

    def __init__(self, platform, message=None):
        self.platform = platform
        self.message = message or f"Could not fetch text from {platform}. Please try again or paste the text directly."
        super().__init__(self.message)


def detect_platform(url):
    """
    Detect the social media platform from a URL.

    Args:
        url: The URL string to analyze.

    Returns:
        Platform identifier string: twitter, reddit, youtube, instagram,
        facebook, linkedin, or unknown.
    """
    if not url or not isinstance(url, str):
        return "unknown"

    url_lower = url.lower().strip()

    if "twitter.com" in url_lower or "x.com" in url_lower:
        return "twitter"
    if "instagram.com" in url_lower:
        return "instagram"
    if "reddit.com" in url_lower or "redd.it" in url_lower:
        return "reddit"
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    if "facebook.com" in url_lower or "fb.com" in url_lower:
        return "facebook"
    if "linkedin.com" in url_lower:
        return "linkedin"

    return "unknown"


def fetch_text_from_url(url):
    """
    Fetch the text content of a social media post from its URL.

    Uses free, public APIs — no authentication tokens needed:
      - Twitter: oEmbed API (publish.twitter.com)
      - Reddit:  Public JSON API (.json suffix)
      - YouTube: oEmbed API (youtube.com/oembed)

    Args:
        url: The post URL string.

    Returns:
        Dict with keys: platform, text, title (optional), author (optional)

    Raises:
        UnsupportedPlatformError: If the platform requires auth.
        FetchError: If fetching fails for a supported platform.
    """
    platform = detect_platform(url)

    if platform == "twitter":
        return _fetch_twitter(url)
    elif platform == "reddit":
        return _fetch_reddit(url)
    elif platform == "youtube":
        return _fetch_youtube(url)
    elif platform in ("instagram", "facebook", "linkedin"):
        raise UnsupportedPlatformError(platform)
    else:
        raise UnsupportedPlatformError("unknown", "Could not detect the platform from this URL. Please paste the post text directly.")


def _fetch_twitter(url):
    """
    Fetch tweet text via Twitter's free oEmbed API.

    Endpoint: https://publish.twitter.com/oembed?url=<tweet_url>
    Returns the tweet HTML, from which we extract the text.
    """
    try:
        # Normalize URL — ensure it has a scheme
        if not url.startswith("http"):
            url = "https://" + url

        oembed_url = f"https://publish.twitter.com/oembed?url={url}&omit_script=true"
        resp = requests.get(oembed_url, timeout=10)
        resp.raise_for_status()

        data = resp.json()

        # Extract text from the HTML response
        html_content = data.get("html", "")
        # The tweet text is inside <p> tags in the blockquote
        text = _extract_text_from_html(html_content)

        if not text:
            text = data.get("url", url)

        author = data.get("author_name", "")

        return {
            "platform": "twitter",
            "text": text.strip(),
            "title": None,
            "author": author,
        }

    except requests.exceptions.RequestException as e:
        logger.error("Twitter oEmbed fetch failed: %s", e)
        raise FetchError("twitter", "Could not fetch the tweet. It may be from a private account or deleted.")


def _fetch_reddit(url):
    """
    Fetch Reddit post text via the public JSON API.

    Works by appending .json to any Reddit post URL.
    """
    try:
        # Normalize URL
        if not url.startswith("http"):
            url = "https://" + url

        # Clean up URL — remove trailing slashes and query params for .json
        clean_url = url.split("?")[0].rstrip("/")

        # Append .json
        json_url = clean_url + ".json"

        headers = {"User-Agent": "SentimentAI/1.0 (Sentiment Analysis Tool)"}
        resp = requests.get(json_url, headers=headers, timeout=10)
        resp.raise_for_status()

        data = resp.json()

        # Reddit returns a list of listings
        if isinstance(data, list) and len(data) > 0:
            post_data = data[0]["data"]["children"][0]["data"]
            title = post_data.get("title", "")
            selftext = post_data.get("selftext", "")
            author = post_data.get("author", "")

            # Combine title and body
            text = title
            if selftext:
                text = f"{title}. {selftext}"

            return {
                "platform": "reddit",
                "text": text.strip(),
                "title": title,
                "author": author,
            }
        else:
            raise FetchError("reddit", "Could not parse the Reddit post data.")

    except requests.exceptions.RequestException as e:
        logger.error("Reddit JSON fetch failed: %s", e)
        raise FetchError("reddit", "Could not fetch the Reddit post. It may be private or deleted.")
    except (KeyError, IndexError, TypeError) as e:
        logger.error("Reddit JSON parsing failed: %s", e)
        raise FetchError("reddit", "Could not parse the Reddit post. The URL format may not be supported.")


def _fetch_youtube(url):
    """
    Fetch YouTube video title via the free oEmbed API.

    Endpoint: https://www.youtube.com/oembed?url=<video_url>&format=json
    """
    try:
        if not url.startswith("http"):
            url = "https://" + url

        oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
        resp = requests.get(oembed_url, timeout=10)
        resp.raise_for_status()

        data = resp.json()
        title = data.get("title", "")
        author = data.get("author_name", "")

        if not title:
            raise FetchError("youtube", "Could not extract the video title.")

        return {
            "platform": "youtube",
            "text": title.strip(),
            "title": title.strip(),
            "author": author,
        }

    except requests.exceptions.RequestException as e:
        logger.error("YouTube oEmbed fetch failed: %s", e)
        raise FetchError("youtube", "Could not fetch the YouTube video info. It may be private or deleted.")


def _extract_text_from_html(html):
    """
    Extract plain text from Twitter oEmbed HTML.

    The oEmbed HTML is a <blockquote> containing <p> tags with the tweet text.
    We strip all HTML tags and extract the text content.
    """
    if not html:
        return ""

    # Remove <a> tags but keep their text content
    text = re.sub(r'<a[^>]*>(.*?)</a>', r'\1', html)

    # Remove <br> and replace with spaces
    text = re.sub(r'<br\s*/?>', ' ', text)

    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove the trailing "— author (@handle) date" attribution
    text = re.sub(r'\s*—\s*[^—]+$', '', text)

    return text


def get_platform_meta(platform):
    """
    Get display metadata for a platform.

    Args:
        platform: Platform identifier string.

    Returns:
        Dict with name, color, emoji, support keys.
    """
    return PLATFORM_META.get(platform, PLATFORM_META["unknown"])
