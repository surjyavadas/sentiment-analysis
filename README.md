# 🧠 SentimentAI — AI-Powered Sentiment Analysis

A production-quality, full-stack web application for analyzing the sentiment of social media posts using **VADER and TextBlob** — with a vibrant colorful UI, URL-based analysis, and full demo mode.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Coming%20Soon-purple?style=for-the-badge&logo=render)](https://sentimentai.onrender.com)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![VADER](https://img.shields.io/badge/VADER-Sentiment-purple)
![TextBlob](https://img.shields.io/badge/TextBlob-NLP-cyan)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## ✨ Features

| Feature | Description |
|---|---|
| **URL Analyzer** | Paste a Twitter, Reddit, or YouTube URL — the app fetches and analyzes the post automatically |
| **Text Analyzer** | Paste any text directly with live character count and model selection |
| **Bulk CSV Upload** | Upload thousands of posts, analyze all at once with drag-and-drop |
| **Demo Mode** | Toggle ON to preload 30 realistic sample analyses — app always looks populated |
| **Model Comparison** | Run VADER and TextBlob side-by-side with confidence scores |
| **Word Highlighting** | Every word color-coded: green (positive), red (negative), grey (neutral) |
| **Confidence Arc Gauge** | Animated SVG semi-circle gauge showing analysis confidence |
| **Dashboard** | Doughnut chart, platform breakdown, trend line, animated stat counters |
| **History Feed** | Searchable, filterable history with platform and sentiment filters |
| **Platform Detection** | Auto-detects Twitter, Reddit, YouTube, Instagram, Facebook, LinkedIn from URL |
| **Graceful Fallbacks** | Beautiful fallback UI for platforms requiring authentication |
| **Share & Copy** | Copy results as text, JSON, or shareable URL |
| **Deployment Ready** | Procfile + render.yaml for one-click Render.com deploy |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.9+, Flask 3.0, Flask-CORS |
| **NLP — Rule-based** | NLTK VADER |
| **NLP — Lexicon** | TextBlob |
| **Database** | SQLite 3 |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript (all inline SPA) |
| **Charts** | Chart.js 4.x (lazy-loaded) |
| **Icons** | Font Awesome 6 |
| **Typography** | Outfit (headings) + Inter (body) via Google Fonts |
| **URL APIs** | Twitter oEmbed, Reddit JSON API, YouTube oEmbed |

---

## 📁 Project Structure

```
sentiment-analyzer/
├── app.py                    # Flask server with 7 API endpoints
├── models/
│   ├── __init__.py
│   ├── vader_model.py        # VADER with word-level scoring
│   ├── textblob_model.py     # TextBlob analyzer
│   └── bert_model.py         # Stub — roadmap item, delegates to VADER
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py       # Text cleaning pipeline
│   ├── url_fetcher.py        # Platform detection + text extraction
│   └── database.py           # SQLite operations
├── templates/
│   └── index.html            # Complete SPA (CSS+JS inline, all 7 sections)
├── data/
│   ├── sample_tweets.csv     # Sample test data
│   └── demo_posts.json       # Demo mode preloaded data (30 entries)
├── requirements.txt
├── Procfile                  # Render.com deployment
├── render.yaml               # One-click deploy config
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone <your-repo-url>
cd sentiment-analyzer
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate   # macOS/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon punkt_tab
python -m textblob.download_corpora
```

### 3. Run
```bash
python app.py
```

Open **http://localhost:5000** 🎉

> **Demo Mode** is ON by default — the app will auto-seed 30 sample analyses so the dashboard and history sections look populated immediately.

---

## 📊 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyze` | POST | Analyze single text |
| `/api/analyze-url` | POST | Analyze social media post by URL |
| `/api/bulk` | POST | Bulk CSV analysis |
| `/api/history` | GET | Last 100 analyses |
| `/api/stats` | GET | Aggregate statistics |
| `/api/history` | DELETE | Clear all history |
| `/api/demo-seed` | POST | Seed demo data (30 entries) |

### Example: Analyze Text
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!", "model": "vader"}'
```

### Example: Analyze URL
```bash
curl -X POST http://localhost:5000/api/analyze-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://twitter.com/elonmusk/status/1234567890"}'
```

### Response Schema
```json
{
  "text": "I love this product!",
  "label": "positive",
  "compound": 0.6369,
  "pos": 0.571,
  "neg": 0.0,
  "neu": 0.429,
  "confidence": 63.7,
  "model": "vader",
  "word_scores": [
    {"word": "I", "score": 0.0, "sentiment": "neutral"},
    {"word": "love", "score": 0.6369, "sentiment": "positive"},
    {"word": "this", "score": 0.0, "sentiment": "neutral"},
    {"word": "product!", "score": 0.0, "sentiment": "neutral"}
  ]
}
```

---

## 🧪 Model Comparison

| Model | Type | Speed | Accuracy | Best For |
|---|---|---|---|---|
| **VADER** | Rule-based | ⚡ Very Fast | ~87% | Social media, emojis, slang |
| **TextBlob** | Lexicon | ⚡ Fast | ~82% | General text, subjectivity |
| **BERT** *(roadmap)* | Deep Learning | 🐢 Slower | ~94% | Nuanced, complex text |

---

## ⚠️ Known Limitations

Transparency is important — here are the current constraints:

| Limitation | Details |
|---|---|
| **Twitter oEmbed** | May be unreliable for very old tweets or if Twitter changes their API. Works well for recent public tweets. |
| **Reddit API** | Requires `User-Agent` header (included). May be rate-limited under heavy use. |
| **Instagram/Facebook/LinkedIn** | Cannot be scraped — these platforms require authentication. The app provides a graceful fallback UI prompting users to paste text manually. |
| **VADER limitations** | Rule-based, so it can miss sarcasm, irony, and complex context. Best for straightforward social media text. |
| **TextBlob limitations** | Pattern-based, so accuracy drops on informal text with heavy slang or emojis. |
| **No BERT yet** | Deep learning model is on the roadmap but excluded from this release to keep deployments lightweight (~500MB+ saved). |

---

## 🗺️ Future Roadmap

- [ ] **BERT/RoBERTa Integration** — Deep learning sentiment with ~94% accuracy (Q3 2025)
- [ ] **Multilingual Support** — Analyze posts in Spanish, French, German, Japanese (Q4 2025)
- [ ] **Sarcasm Detection** — AI layer to detect irony and implied sentiment (2026)
- [ ] **Real-time Monitoring** — Stream and analyze posts in real-time from Twitter/Reddit
- [ ] **Export to PDF** — Generate beautiful sentiment reports
- [ ] **Team Collaboration** — Share dashboards and history across teams

---

## 🚀 Deploy to Render

1. Push to GitHub
2. Connect to [Render.com](https://render.com)
3. Use the included `render.yaml` for one-click deploy

Or manually:
- **Build command:** `pip install -r requirements.txt && python -m nltk.downloader vader_lexicon punkt_tab && python -m textblob.download_corpora`
- **Start command:** `gunicorn app:app`

> ✅ No heavy dependencies — no `transformers` or `torch` means fast builds and reliable free-tier deployments.

---

## 📝 License

MIT License — free for personal and commercial use.

---

Built with ❤️ using Python, Flask, and modern NLP.
