# 🧠 SentimentAI — AI-Powered Sentiment Analysis

A production-quality, full-stack web application for analyzing the sentiment of social media posts using **three NLP models** — VADER, TextBlob, and RoBERTa — with a stunning Three.js-powered dark-mode dashboard.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![HuggingFace](https://img.shields.io/badge/HuggingFace-RoBERTa-yellow?logo=huggingface)
![Three.js](https://img.shields.io/badge/Three.js-Particles-black?logo=three.js)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Single Post Analyzer** | Paste text, choose model, get instant sentiment with confidence + word highlighting |
| **Bulk CSV Upload** | Upload thousands of posts, analyze all at once with drag-and-drop |
| **3D Particle Hero** | 2000-particle Three.js background with mouse parallax |
| **3D Sentiment Orb** | Color-shifting CSS 3D orb (green/red/grey) |
| **Model Comparison** | Run VADER, TextBlob, and BERT side-by-side |
| **Word Highlighting** | Every word color-coded: green (positive), red (negative), grey (neutral) |
| **Dashboard** | Doughnut chart, trend line, word cloud, animated stat counters |
| **History Feed** | Searchable, filterable history of all past analyses |
| **API Docs** | Built-in API reference with copy-to-clipboard code examples |
| **Share Links** | Generate shareable URLs with query parameters |
| **Deployment Ready** | Procfile + render.yaml for one-click Render.com deploy |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.9+, Flask 3.0, Flask-CORS |
| **NLP — Rule-based** | NLTK VADER |
| **NLP — Lexicon** | TextBlob |
| **NLP — Deep Learning** | HuggingFace Transformers (cardiffnlp/twitter-roberta-base-sentiment) |
| **Database** | SQLite 3 |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript (all inline) |
| **3D Graphics** | Three.js r128 |
| **Charts** | Chart.js 4.x |
| **Icons** | Font Awesome 6 |
| **Typography** | Space Grotesk + Inter + JetBrains Mono |

---

## 📁 Project Structure

```
sentiment-analyzer/
├── app.py                    # Flask server with 5 API endpoints
├── models/
│   ├── __init__.py
│   ├── vader_model.py        # VADER with word-level scoring
│   ├── textblob_model.py     # TextBlob analyzer
│   └── bert_model.py         # RoBERTa with VADER fallback
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py       # Text cleaning pipeline
│   └── database.py           # SQLite operations
├── templates/
│   └── index.html            # Complete SPA (CSS+JS inline)
├── data/
│   └── sample_tweets.csv     # 20 test tweets
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

> **Note:** RoBERTa downloads ~500 MB on first analysis. Falls back to VADER if unavailable.

---

## 📊 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyze` | POST | Analyze single text |
| `/api/bulk` | POST | Bulk CSV analysis |
| `/api/history` | GET | Last 100 analyses |
| `/api/stats` | GET | Aggregate statistics |
| `/api/history` | DELETE | Clear all history |

### Example: Analyze Text
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!", "model": "all"}'
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
| **VADER** | Rule-based | ⚡ Very Fast | Good | Social media, emojis, slang |
| **TextBlob** | Lexicon | ⚡ Fast | Moderate | General text, subjectivity |
| **RoBERTa** | Deep Learning | 🐢 Slower | High | Nuanced, complex text |

---

## 🚀 Deploy to Render

1. Push to GitHub
2. Connect to [Render.com](https://render.com)
3. Use the included `render.yaml` for one-click deploy

Or manually:
- **Build command:** `pip install -r requirements.txt && python -m nltk.downloader vader_lexicon punkt_tab`
- **Start command:** `gunicorn app:app`

---

## 📝 License

MIT License — free for personal and commercial use.

---

Built with ❤️ using Python, Flask, Three.js, and modern NLP.
