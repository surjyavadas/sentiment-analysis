"""
SentimentAI — Flask Backend

Production-quality REST API for multi-model sentiment analysis.
Serves the SPA frontend and exposes 6 API endpoints including
URL-based social media analysis.
"""
import os
import io
import json
import logging
from datetime import datetime, timezone

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd

from utils.database import init_db, save_analysis, save_analyses_bulk, get_history, clear_history, get_stats
from utils.url_fetcher import detect_platform, fetch_text_from_url, get_platform_meta, UnsupportedPlatformError, FetchError
from models import vader_model, textblob_model, bert_model

# ─── App Setup ────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize database
init_db()
logger.info("Database initialized.")

# Model dispatch table
MODEL_MAP = {
    "vader": vader_model,
    "textblob": textblob_model,
    "bert": bert_model,
}


# ─── Page Route ───────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page application."""
    return render_template("index.html")


# ─── POST /api/analyze ────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Analyze a single text.

    Body: { "text": "...", "model": "vader|textblob|bert|all" }
    Returns: Single result dict, or array of 3 results if model="all".
    """
    data = request.get_json(silent=True)
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "Missing 'text' field."}), 400

    text = data["text"].strip()
    model_name = data.get("model", "vader").lower()

    try:
        if model_name == "all":
            results = []
            for name, mod in MODEL_MAP.items():
                result = mod.analyze(text)
                save_analysis(result)
                results.append(result)
            return jsonify(results), 200
        else:
            mod = MODEL_MAP.get(model_name)
            if not mod:
                return jsonify({"error": f"Unknown model '{model_name}'. Use: vader, textblob, bert, all"}), 400
            result = mod.analyze(text)
            save_analysis(result)
            return jsonify(result), 200

    except Exception as e:
        logger.error("Analysis error: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


# ─── POST /api/analyze-url ───────────────────────────────────────────

@app.route("/api/analyze-url", methods=["POST"])
def api_analyze_url():
    """
    Analyze a social media post by URL.

    Body: { "url": "https://twitter.com/...", "model": "vader|textblob|all" }
    Steps: detect platform → fetch text via platform API/oEmbed → run sentiment → return result
    Returns: { platform, platform_meta, extracted_text, label, compound, pos, neg, neu,
               confidence, word_scores[], url, model, timestamp }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("url", "").strip():
        return jsonify({"error": "Missing 'url' field."}), 400

    url = data["url"].strip()
    model_name = data.get("model", "vader").lower()
    platform = detect_platform(url)

    try:
        # Fetch text from URL
        fetch_result = fetch_text_from_url(url)
        extracted_text = fetch_result["text"]

        if not extracted_text:
            return jsonify({
                "error": "Could not extract text from this URL.",
                "platform": platform,
                "platform_meta": get_platform_meta(platform),
            }), 400

        # Run sentiment analysis
        mod = MODEL_MAP.get(model_name, vader_model)
        result = mod.analyze(extracted_text)

        # Enrich result with URL/platform data
        result["platform"] = platform
        result["platform_meta"] = get_platform_meta(platform)
        result["url"] = url
        result["extracted_text"] = extracted_text
        result["author"] = fetch_result.get("author", "")
        result["post_title"] = fetch_result.get("title", "")

        # Save to database
        save_analysis(result)

        return jsonify(result), 200

    except UnsupportedPlatformError as e:
        return jsonify({
            "error": e.message,
            "error_type": "unsupported_platform",
            "platform": e.platform,
            "platform_meta": get_platform_meta(e.platform),
        }), 422

    except FetchError as e:
        return jsonify({
            "error": e.message,
            "error_type": "fetch_error",
            "platform": e.platform,
            "platform_meta": get_platform_meta(e.platform),
        }), 502

    except Exception as e:
        logger.error("URL analysis error: %s", e, exc_info=True)
        return jsonify({
            "error": str(e),
            "error_type": "unknown",
            "platform": platform,
            "platform_meta": get_platform_meta(platform),
        }), 500


# ─── POST /api/bulk ───────────────────────────────────────────────────

@app.route("/api/bulk", methods=["POST"])
def api_bulk():
    """
    Bulk analyze a CSV file.

    Body: multipart/form-data with 'file' field (CSV with 'text' column).
    Returns: { results: [...], summary: { total, positive, negative, neutral } }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use 'file' field."}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted."}), 400

    try:
        content = file.read().decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(content))

        if df.empty:
            return jsonify({"error": "CSV file is empty."}), 400

        # Find text column
        text_col = None
        for col in ["text", "Text", "TEXT", "tweet", "Tweet", "content", "Content", "message"]:
            if col in df.columns:
                text_col = col
                break
        if text_col is None:
            text_col = df.columns[0]

        texts = df[text_col].dropna().astype(str).tolist()

        if len(texts) > 10000:
            return jsonify({"error": f"CSV has {len(texts)} rows. Maximum is 10,000."}), 400

        model_name = request.form.get("model", "vader").lower()
        mod = MODEL_MAP.get(model_name, vader_model)

        results = []
        for t in texts:
            result = mod.analyze(t)
            results.append(result)

        # Save all to database
        save_analyses_bulk(results)

        # Summary
        labels = [r["label"] for r in results]
        summary = {
            "total": len(results),
            "positive": labels.count("positive"),
            "negative": labels.count("negative"),
            "neutral": labels.count("neutral"),
        }

        return jsonify({"results": results, "summary": summary}), 200

    except pd.errors.EmptyDataError:
        return jsonify({"error": "CSV is empty or malformed."}), 400
    except Exception as e:
        logger.error("Bulk analysis error: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


# ─── GET /api/history ─────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def api_history():
    """
    Get the last 100 analyses.

    Returns: Array of analysis result objects.
    """
    try:
        items = get_history(limit=100)
        return jsonify(items), 200
    except Exception as e:
        logger.error("History error: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


# ─── GET /api/stats ───────────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def api_stats():
    """
    Get aggregate statistics.

    Returns: { total_count, positive_pct, negative_pct, neutral_pct, by_platform{}, trend[] }
    """
    try:
        stats = get_stats()
        return jsonify(stats), 200
    except Exception as e:
        logger.error("Stats error: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


# ─── DELETE /api/history ──────────────────────────────────────────────

@app.route("/api/history", methods=["DELETE"])
def api_clear_history():
    """
    Clear all analysis history.

    Returns: { message: "Cleared N records." }
    """
    try:
        count = clear_history()
        return jsonify({"message": f"Cleared {count} records."}), 200
    except Exception as e:
        logger.error("Clear history error: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("=" * 60)
    logger.info("  SentimentAI starting on http://0.0.0.0:%s", port)
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=True)
