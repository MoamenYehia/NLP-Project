# ◈ SentimentLens

Local NLP sentiment and emotion analysis built with Flask, NLTK, VADER, TextBlob, and scikit-learn. The app runs fully offline after the initial NLTK data download and does not use external AI APIs.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8-4B8BBE?style=flat)
![VADER](https://img.shields.io/badge/VADER-Sentiment-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

## Features

- Sentiment label with confidence using a VADER + TextBlob ensemble
- Eight-emotion scoring with a curated NRC-style lexicon
- Keyword extraction with TF-IDF and POS fallback
- Readability and text statistics such as TTR and Flesch Reading Ease
- Browser UI with charts and sample inputs

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK data once

```bash
python setup.py
```

### 4. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## Windows Notes

- If you already have a `.venv` folder, you do not need to recreate it.
- The repository includes a real [.gitignore](.gitignore) so the virtual environment stays out of version control.
- The app is configured to serve the existing root-level `index.html`, `main.js`, and `style.css` files directly.

## How It Works

1. Paste or type text into the UI.
2. Flask sends it to the `/analyze` endpoint.
3. The analysis engine returns sentiment, emotion scores, keywords, and statistics.
4. The frontend renders the result with charts and summary cards.

## API

### `POST /analyze`

Request body:

```json
{ "text": "Your text here" }
```

Example response fields:

```json
{
  "sentiment": {
    "label": "Positive",
    "score": 0.87,
    "explanation": "The text carries a clear positive tone..."
  },
  "emotions": [
    { "name": "Joy", "score": 0.61 },
    { "name": "Trust", "score": 0.38 }
  ],
  "keywords": ["incredible", "food", "kind"],
  "summary": "This text was classified as Positive..."
}
```

## Project Structure

```text
SentimentLens/
├── app.py
├── sentiment_engine.py
├── setup.py
├── requirements.txt
├── index.html
├── main.js
├── style.css
├── README.md
└── .gitignore
```

## Dependencies

- Flask for the web server
- NLTK for tokenization, POS tagging, and sentiment resources
- VADER for rule-based sentiment scoring
- TextBlob for polarity and subjectivity
- scikit-learn for TF-IDF keyword extraction

## License

MIT License. See [LICENSE](LICENSE).
