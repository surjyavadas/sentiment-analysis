"""
SQLite Database Helper for SentimentAI.

Manages all persistent storage: analysis history, statistics,
and word frequency data. Auto-creates the database on first run.
Supports platform/URL fields for URL-based analysis.
"""
import os
import sqlite3
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "history.db")


def _get_conn():
    """Get a SQLite connection with row factory."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create the analyses table if it doesn't exist, and migrate if needed."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            clean_text TEXT,
            label TEXT NOT NULL,
            compound REAL,
            pos REAL,
            neg REAL,
            neu REAL,
            confidence REAL,
            model TEXT NOT NULL,
            word_scores TEXT,
            timestamp TEXT NOT NULL,
            platform TEXT DEFAULT '',
            url TEXT DEFAULT '',
            extracted_text TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp ON analyses(timestamp DESC)
    """)

    # Migration: add platform/url/extracted_text columns to existing tables
    _migrate_add_columns(conn)

    conn.commit()
    conn.close()


def _migrate_add_columns(conn):
    """Add platform, url, extracted_text columns if they don't exist yet."""
    cursor = conn.execute("PRAGMA table_info(analyses)")
    existing_cols = {row["name"] for row in cursor.fetchall()}

    migrations = [
        ("platform", "TEXT DEFAULT ''"),
        ("url", "TEXT DEFAULT ''"),
        ("extracted_text", "TEXT DEFAULT ''"),
    ]

    for col_name, col_type in migrations:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE analyses ADD COLUMN {col_name} {col_type}")


def save_analysis(result):
    """
    Save a single analysis result to the database.

    Args:
        result: Dict with keys: text, clean_text, label, compound, pos, neg, neu,
                confidence, model, word_scores, timestamp, platform, url, extracted_text.

    Returns:
        The row ID of the saved record.
    """
    conn = _get_conn()
    cur = conn.execute("""
        INSERT INTO analyses (text, clean_text, label, compound, pos, neg, neu,
                              confidence, model, word_scores, timestamp,
                              platform, url, extracted_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.get("text", ""),
        result.get("clean_text", ""),
        result.get("label", "neutral"),
        result.get("compound", 0.0),
        result.get("pos", 0.0),
        result.get("neg", 0.0),
        result.get("neu", 0.0),
        result.get("confidence", 0.0),
        result.get("model", "vader"),
        json.dumps(result.get("word_scores", [])),
        result.get("timestamp", datetime.utcnow().isoformat()),
        result.get("platform", ""),
        result.get("url", ""),
        result.get("extracted_text", ""),
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def save_analyses_bulk(results):
    """
    Save multiple analysis results in a single transaction.

    Args:
        results: List of result dicts.

    Returns:
        Number of records inserted.
    """
    conn = _get_conn()
    rows = [
        (
            r.get("text", ""), r.get("clean_text", ""), r.get("label", "neutral"),
            r.get("compound", 0.0), r.get("pos", 0.0), r.get("neg", 0.0),
            r.get("neu", 0.0), r.get("confidence", 0.0), r.get("model", "vader"),
            json.dumps(r.get("word_scores", [])),
            r.get("timestamp", datetime.utcnow().isoformat()),
            r.get("platform", ""),
            r.get("url", ""),
            r.get("extracted_text", ""),
        )
        for r in results
    ]
    conn.executemany("""
        INSERT INTO analyses (text, clean_text, label, compound, pos, neg, neu,
                              confidence, model, word_scores, timestamp,
                              platform, url, extracted_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    count = len(rows)
    conn.commit()
    conn.close()
    return count


def get_history(limit=100):
    """
    Fetch the most recent analyses.

    Args:
        limit: Maximum records to return (default 100).

    Returns:
        List of dicts with all analysis fields including platform/url.
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM analyses ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    items = []
    for row in rows:
        item = dict(row)
        try:
            item["word_scores"] = json.loads(item.get("word_scores") or "[]")
        except (json.JSONDecodeError, TypeError):
            item["word_scores"] = []
        # Ensure platform/url fields exist even for old records
        item.setdefault("platform", "")
        item.setdefault("url", "")
        item.setdefault("extracted_text", "")
        items.append(item)
    return items


def clear_history():
    """Delete all analysis records. Returns the count of deleted rows."""
    conn = _get_conn()
    cur = conn.execute("DELETE FROM analyses")
    count = cur.rowcount
    conn.commit()
    conn.close()
    return count


def get_stats():
    """
    Get aggregate statistics from all analyses, including per-platform breakdown.

    Returns:
        Dict with total_count, positive_pct, negative_pct, neutral_pct,
        by_platform{}, and trend[].
    """
    conn = _get_conn()

    # Total count
    total = conn.execute("SELECT COUNT(*) as c FROM analyses").fetchone()["c"]

    if total == 0:
        conn.close()
        return {
            "total_count": 0,
            "positive_pct": 0,
            "negative_pct": 0,
            "neutral_pct": 0,
            "by_platform": {},
            "trend": [],
        }

    # Sentiment counts
    pos = conn.execute("SELECT COUNT(*) as c FROM analyses WHERE label='positive'").fetchone()["c"]
    neg = conn.execute("SELECT COUNT(*) as c FROM analyses WHERE label='negative'").fetchone()["c"]
    neu = conn.execute("SELECT COUNT(*) as c FROM analyses WHERE label='neutral'").fetchone()["c"]

    # Per-platform breakdown
    by_platform = {}
    try:
        platform_rows = conn.execute("""
            SELECT platform, label, COUNT(*) as cnt
            FROM analyses
            WHERE platform != '' AND platform IS NOT NULL
            GROUP BY platform, label
        """).fetchall()

        for row in platform_rows:
            p = row["platform"]
            if p not in by_platform:
                by_platform[p] = {"total": 0, "positive": 0, "negative": 0, "neutral": 0}
            by_platform[p][row["label"]] = row["cnt"]
            by_platform[p]["total"] += row["cnt"]
    except Exception:
        # Old DB without platform column — silently skip
        pass

    # Trend: last 30 analyses in chronological order
    trend_rows = conn.execute("""
        SELECT label, compound, confidence, timestamp
        FROM analyses ORDER BY timestamp DESC LIMIT 30
    """).fetchall()
    conn.close()

    trend = [
        {
            "label": r["label"],
            "compound": r["compound"],
            "confidence": r["confidence"],
            "timestamp": r["timestamp"],
        }
        for r in reversed(trend_rows)
    ]

    return {
        "total_count": total,
        "positive_pct": round((pos / total) * 100, 1),
        "negative_pct": round((neg / total) * 100, 1),
        "neutral_pct": round((neu / total) * 100, 1),
        "by_platform": by_platform,
        "trend": trend,
    }
