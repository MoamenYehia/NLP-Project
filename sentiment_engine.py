"""
Sentiment & Emotion Analysis Engine
------------------------------------
Techniques used:
  - VADER  : rule-based lexicon sentiment (Hutto & Gilbert, 2014)
  - TextBlob: pattern-based polarity & subjectivity
  - Ensemble: weighted combination of both for final sentiment
  - TF-IDF  : keyword extraction (scikit-learn)
  - NLTK    : tokenization, POS tagging, stopword removal
  - Lexicon-based emotion detection (NRC Emotion mapping via custom rules)
  - Readability: sentence stats, avg word length, type-token ratio
"""

import re
import math
import string
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# ── NRC-style emotion lexicon (curated subset) ──────────────────────────────
# Each word maps to one or more emotions it evokes.
EMOTION_LEXICON: dict[str, list[str]] = {
    # Joy
    "happy": ["joy"], "happiness": ["joy"], "joy": ["joy"], "excited": ["joy", "anticipation"],
    "love": ["joy", "trust"], "wonderful": ["joy"], "amazing": ["joy", "surprise"],
    "fantastic": ["joy"], "great": ["joy"], "excellent": ["joy"], "good": ["joy"],
    "delightful": ["joy"], "glad": ["joy"], "pleasure": ["joy"], "enjoy": ["joy"],
    "laugh": ["joy"], "smile": ["joy"], "beautiful": ["joy"], "brilliant": ["joy"],
    "celebrate": ["joy", "anticipation"], "success": ["joy", "trust"],
    "incredible": ["joy", "surprise"], "thrilled": ["joy", "anticipation"],
    "awesome": ["joy"], "fun": ["joy"], "cheer": ["joy"], "bliss": ["joy"],

    # Sadness
    "sad": ["sadness"], "sadness": ["sadness"], "unhappy": ["sadness"],
    "depressed": ["sadness"], "miserable": ["sadness"], "cry": ["sadness"],
    "grief": ["sadness"], "sorrow": ["sadness"], "mourn": ["sadness"],
    "loss": ["sadness"], "lonely": ["sadness"], "alone": ["sadness"],
    "hopeless": ["sadness", "fear"], "heartbreak": ["sadness"], "regret": ["sadness"],
    "disappointed": ["sadness"], "hurt": ["sadness", "anger"], "pain": ["sadness"],
    "suffer": ["sadness"], "despair": ["sadness", "fear"], "tragic": ["sadness"],

    # Anger
    "angry": ["anger"], "anger": ["anger"], "furious": ["anger"], "rage": ["anger"],
    "hate": ["anger", "disgust"], "frustrate": ["anger"], "frustrated": ["anger"],
    "annoyed": ["anger"], "irritated": ["anger"], "outrage": ["anger"],
    "resentment": ["anger"], "hostility": ["anger"], "mad": ["anger"],
    "aggressive": ["anger"], "violent": ["anger", "fear"], "cruel": ["anger", "disgust"],
    "unfair": ["anger"], "betrayed": ["anger", "sadness"], "cheat": ["anger", "disgust"],
    "lie": ["anger", "disgust"], "scam": ["anger", "disgust"],

    # Fear
    "fear": ["fear"], "afraid": ["fear"], "scared": ["fear"], "terrified": ["fear"],
    "horror": ["fear"], "dread": ["fear"], "panic": ["fear"], "anxiety": ["fear"],
    "worry": ["fear"], "threat": ["fear"], "danger": ["fear"], "risk": ["fear"],
    "nightmare": ["fear"], "phobia": ["fear"], "nervous": ["fear"],
    "uncertain": ["fear"], "doubt": ["fear"], "overwhelmed": ["fear", "sadness"],

    # Surprise
    "surprised": ["surprise"], "surprise": ["surprise"], "astonished": ["surprise"],
    "shocked": ["surprise"], "amazed": ["surprise"], "unexpected": ["surprise"],
    "unbelievable": ["surprise"], "stunning": ["surprise"], "suddenly": ["surprise"],
    "discovered": ["surprise"], "reveal": ["surprise"], "twist": ["surprise"],
    "extraordinary": ["surprise", "joy"], "remarkable": ["surprise", "joy"],

    # Disgust
    "disgusting": ["disgust"], "disgust": ["disgust"], "gross": ["disgust"],
    "revolting": ["disgust"], "nasty": ["disgust"], "horrible": ["disgust"],
    "awful": ["disgust"], "terrible": ["disgust"], "vile": ["disgust"],
    "repulsive": ["disgust"], "filthy": ["disgust"], "corrupt": ["disgust", "anger"],
    "toxic": ["disgust"], "poisonous": ["disgust", "fear"],

    # Trust
    "trust": ["trust"], "honest": ["trust"], "reliable": ["trust"],
    "faithful": ["trust"], "loyal": ["trust"], "dependable": ["trust"],
    "confident": ["trust", "joy"], "secure": ["trust"], "safe": ["trust"],
    "genuine": ["trust"], "authentic": ["trust"], "integrity": ["trust"],
    "believe": ["trust"], "faith": ["trust"], "committed": ["trust"],

    # Anticipation
    "hope": ["anticipation", "joy"], "expect": ["anticipation"],
    "anticipate": ["anticipation"], "waiting": ["anticipation"],
    "future": ["anticipation"], "plan": ["anticipation"], "goal": ["anticipation"],
    "dream": ["anticipation", "joy"], "eager": ["anticipation", "joy"],
    "looking forward": ["anticipation", "joy"], "soon": ["anticipation"],
    "upcoming": ["anticipation"], "prepare": ["anticipation"],
}

ALL_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
EMOTION_LABELS = {e: e.capitalize() for e in ALL_EMOTIONS}

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
vader = SentimentIntensityAnalyzer()


# ── Text Preprocessing ───────────────────────────────────────────────────────

def preprocess(text: str) -> dict:
    """Tokenize, clean, POS-tag the input text."""
    # Raw sentences
    sentences = sent_tokenize(text)
    # Lowercase tokens, remove punctuation
    tokens_raw = word_tokenize(text.lower())
    tokens_clean = [t for t in tokens_raw if t.isalpha() and t not in STOP_WORDS]
    tokens_all   = [t for t in tokens_raw if t.isalpha()]   # with stopwords for POS
    # POS tags
    pos_tags = pos_tag(tokens_all)
    # Stems
    stems = [stemmer.stem(t) for t in tokens_clean]

    return {
        "sentences": sentences,
        "tokens_raw": tokens_raw,
        "tokens_clean": tokens_clean,
        "tokens_all": tokens_all,
        "pos_tags": pos_tags,
        "stems": stems,
        "word_count": len(tokens_all),
        "sentence_count": len(sentences),
    }


# ── Sentiment Analysis ───────────────────────────────────────────────────────

def analyze_sentiment(text: str, preprocessed: dict) -> dict:
    """
    Ensemble sentiment:
      - VADER (weight 0.55): handles social text, punctuation, emojis, negations
      - TextBlob (weight 0.45): pattern-based, good on formal text
    Combined compound score → label + confidence.
    """
    # VADER
    vs = vader.polarity_scores(text)
    vader_compound = vs["compound"]  # -1 to +1

    # TextBlob
    tb = TextBlob(text)
    tb_polarity = tb.sentiment.polarity  # -1 to +1
    tb_subjectivity = tb.sentiment.subjectivity  # 0 to 1

    # Ensemble (weighted average of compound scores)
    ensemble = 0.55 * vader_compound + 0.45 * tb_polarity

    # Sentence-level variance (detect mixed sentiment)
    sent_scores = [vader.polarity_scores(s)["compound"] for s in preprocessed["sentences"]]
    variance = _variance(sent_scores) if len(sent_scores) > 1 else 0.0
    has_mix   = variance > 0.15 and any(s > 0.2 for s in sent_scores) and any(s < -0.2 for s in sent_scores)

    # Label
    if has_mix:
        label = "Mixed"
        confidence = 0.5 + variance * 0.5  # higher variance = more confident it's mixed
    elif ensemble >= 0.05:
        label = "Positive"
        confidence = _norm(ensemble, 0.05, 1.0)
    elif ensemble <= -0.05:
        label = "Negative"
        confidence = _norm(-ensemble, 0.05, 1.0)
    else:
        label = "Neutral"
        confidence = 1.0 - abs(ensemble) * 4  # closer to 0 = more confident neutral

    confidence = round(min(max(confidence, 0.45), 0.99), 3)

    explanation = _sentiment_explanation(label, ensemble, tb_subjectivity, preprocessed)

    return {
        "label": label,
        "score": confidence,
        "explanation": explanation,
        "details": {
            "vader_compound": round(vader_compound, 4),
            "vader_pos": round(vs["pos"], 4),
            "vader_neg": round(vs["neg"], 4),
            "vader_neu": round(vs["neu"], 4),
            "textblob_polarity": round(tb_polarity, 4),
            "textblob_subjectivity": round(tb_subjectivity, 4),
            "ensemble_score": round(ensemble, 4),
            "sentence_variance": round(variance, 4),
        },
    }


def _sentiment_explanation(label, score, subjectivity, preprocessed):
    subj_str = "highly subjective" if subjectivity > 0.6 else "fairly objective" if subjectivity < 0.35 else "moderately subjective"
    wc = preprocessed["word_count"]
    sc = preprocessed["sentence_count"]

    if label == "Positive":
        return f"The text carries a clear positive tone across its {sc} sentence(s). It is {subj_str} with an ensemble polarity score of {score:+.2f}."
    elif label == "Negative":
        return f"The text expresses negative sentiment throughout its {wc} words. It is {subj_str} with a polarity score of {score:+.2f}."
    elif label == "Mixed":
        return f"Sentence-level analysis reveals conflicting sentiments — some strongly positive, some negative — giving this {wc}-word text a mixed emotional profile."
    else:
        return f"The text reads as largely neutral or factual across {sc} sentence(s), with minimal emotional loading (polarity ≈ {score:+.2f})."


# ── Emotion Detection ────────────────────────────────────────────────────────

def detect_emotions(preprocessed: dict) -> list[dict]:
    """
    Lexicon-based emotion detection using NRC-style mapping.
    Applies:
      - Direct token matching against EMOTION_LEXICON
      - Stem-based fallback matching
      - Sentence-level negation handling
      - Score normalisation via softmax-inspired smoothing
    Returns list of {name, score} sorted by score desc.
    """
    emotion_hits: dict[str, float] = {e: 0.0 for e in ALL_EMOTIONS}
    tokens = preprocessed["tokens_clean"]
    sentences = preprocessed["sentences"]

    # Negation words — if present before an emotion word, dampen it
    NEGATIONS = {"not", "no", "never", "neither", "nor", "nothing", "nobody",
                 "nowhere", "hardly", "barely", "scarcely", "dont", "doesnt",
                 "didnt", "isnt", "wasnt", "shouldnt", "wouldnt", "couldnt"}

    for sentence in sentences:
        sent_tokens = word_tokenize(sentence.lower())
        for i, token in enumerate(sent_tokens):
            if not token.isalpha():
                continue
            # Check negation window (3 words before)
            window = sent_tokens[max(0, i - 3): i]
            negated = any(w in NEGATIONS for w in window)

            # Direct match
            emotions_for_token = EMOTION_LEXICON.get(token, [])
            # Stem fallback
            if not emotions_for_token:
                stem = stemmer.stem(token)
                for key, emots in EMOTION_LEXICON.items():
                    if stemmer.stem(key) == stem:
                        emotions_for_token = emots
                        break

            for em in emotions_for_token:
                weight = -0.4 if negated else 1.0
                emotion_hits[em] += weight

    # Intensity boost: VADER sentence-level scores feed into joy/sadness/anger
    vs = vader.polarity_scores(" ".join(preprocessed["tokens_all"]))
    emotion_hits["joy"]     += max(vs["pos"] * 3, 0)
    emotion_hits["sadness"] += max(vs["neg"] * 2, 0)
    emotion_hits["anger"]   += max(vs["neg"] * 1, 0)

    # Clamp negatives to 0
    emotion_hits = {k: max(v, 0.0) for k, v in emotion_hits.items()}

    total = sum(emotion_hits.values()) or 1.0

    # Softmax-inspired normalisation (avoid all-zero edge case)
    base_score = 0.05  # every emotion gets at least a tiny baseline
    scores = {}
    for em in ALL_EMOTIONS:
        raw = emotion_hits[em] / total
        scores[em] = round(min(base_score + raw * 0.95, 1.0), 3)

    return [
        {"name": EMOTION_LABELS[em], "score": scores[em]}
        for em in ALL_EMOTIONS
    ]


# ── Keyword Extraction (TF-IDF) ──────────────────────────────────────────────

def extract_keywords(text: str, preprocessed: dict, top_n: int = 6) -> list[str]:
    """
    TF-IDF on sentence level — picks words that are important within
    this document relative to a pseudo-corpus of its own sentences.
    Falls back to frequency-based if fewer than 2 sentences.
    """
    sentences = preprocessed["sentences"]

    if len(sentences) >= 2:
        try:
            tfidf = TfidfVectorizer(
                stop_words="english",
                token_pattern=r"\b[a-zA-Z]{3,}\b",
                max_features=200,
            )
            tfidf.fit_transform(sentences)
            scores = dict(zip(tfidf.get_feature_names_out(),
                               tfidf.idf_))  # IDF as proxy for uniqueness
            # Score words by TF (in full text) × IDF uniqueness
            word_freq = Counter(preprocessed["tokens_clean"])
            combined = {
                w: (word_freq.get(w, 0) / len(preprocessed["tokens_clean"])) * idf
                for w, idf in scores.items()
                if w in word_freq
            }
            keywords = sorted(combined, key=combined.get, reverse=True)[:top_n]
            return keywords if keywords else _freq_keywords(preprocessed, top_n)
        except Exception:
            pass

    return _freq_keywords(preprocessed, top_n)


def _freq_keywords(preprocessed: dict, top_n: int) -> list[str]:
    """Frequency fallback — return most common content words."""
    # Prefer nouns and adjectives via POS filter
    good_pos = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VBG", "VBD"}
    pos_words = [w for w, p in preprocessed["pos_tags"]
                 if p in good_pos and w.lower() not in STOP_WORDS and len(w) > 3]
    freq = Counter(pos_words)
    return [w for w, _ in freq.most_common(top_n)] or \
           [w for w, _ in Counter(preprocessed["tokens_clean"]).most_common(top_n)]


# ── Text Statistics ───────────────────────────────────────────────────────────

def text_statistics(text: str, preprocessed: dict) -> dict:
    """
    Readability and linguistic statistics:
      - Type-Token Ratio (lexical diversity)
      - Average sentence length
      - Average word length
      - Flesch Reading Ease (approximate)
      - POS distribution
    """
    tokens = preprocessed["tokens_all"]
    sentences = preprocessed["sentences"]

    # Type-Token Ratio
    ttr = len(set(tokens)) / len(tokens) if tokens else 0

    # Average lengths
    avg_sent_len = sum(len(word_tokenize(s)) for s in sentences) / len(sentences) if sentences else 0
    avg_word_len = sum(len(w) for w in tokens) / len(tokens) if tokens else 0

    # Syllable count approximation for Flesch
    total_syllables = sum(_count_syllables(w) for w in tokens)
    wc = len(tokens)
    sc = len(sentences)
    flesch = 206.835 - 1.015 * (wc / sc) - 84.6 * (total_syllables / wc) if wc and sc else 0
    flesch = round(max(0, min(flesch, 100)), 1)

    # POS distribution
    pos_dist: dict[str, int] = {}
    for _, tag in preprocessed["pos_tags"]:
        group = _simplify_pos(tag)
        pos_dist[group] = pos_dist.get(group, 0) + 1

    return {
        "word_count": wc,
        "sentence_count": sc,
        "unique_words": len(set(t.lower() for t in tokens)),
        "type_token_ratio": round(ttr, 3),
        "avg_sentence_length": round(avg_sent_len, 1),
        "avg_word_length": round(avg_word_len, 2),
        "flesch_reading_ease": flesch,
        "pos_distribution": pos_dist,
    }


def _count_syllables(word: str) -> int:
    """Simple syllable counter using vowel groups."""
    word = word.lower()
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _simplify_pos(tag: str) -> str:
    if tag.startswith("NN"):  return "Nouns"
    if tag.startswith("VB"):  return "Verbs"
    if tag.startswith("JJ"):  return "Adjectives"
    if tag.startswith("RB"):  return "Adverbs"
    if tag.startswith("PRP"): return "Pronouns"
    return "Other"


# ── Summary Generator ─────────────────────────────────────────────────────────

def generate_summary(sentiment: dict, emotions: list[dict], stats: dict, keywords: list[str]) -> str:
    """
    Rule-based NLG summary — no LLM, uses analysis outputs to compose prose.
    """
    label = sentiment["label"]
    conf  = int(sentiment["score"] * 100)
    top_emotions = sorted(emotions, key=lambda e: e["score"], reverse=True)[:2]
    top_em_names = " and ".join(e["name"] for e in top_emotions)
    flesch = stats["flesch_reading_ease"]
    ttr    = stats["type_token_ratio"]

    if flesch > 70:   read_str = "easy to read"
    elif flesch > 50: read_str = "moderately readable"
    else:              read_str = "dense or complex"

    diversity = "rich" if ttr > 0.7 else "moderate" if ttr > 0.45 else "repetitive"

    kw_str = ", ".join(f'"{k}"' for k in keywords[:4]) if keywords else "none identified"

    return (
        f"This {stats['word_count']}-word text was classified as {label} with {conf}% confidence "
        f"using an ensemble of VADER and TextBlob sentiment models. "
        f"Emotion analysis surfaced dominant themes of {top_em_names} "
        f"via lexicon-based NRC mapping. "
        f"Key signal words include {kw_str} (extracted via TF-IDF). "
        f"The writing style is {read_str} (Flesch score: {flesch}) "
        f"with {diversity} vocabulary diversity (TTR: {ttr:.2f})."
    )


# ── Main Analysis Entry Point ─────────────────────────────────────────────────

def analyze(text: str) -> dict:
    """Run the full NLP pipeline on input text."""
    preprocessed = preprocess(text)
    sentiment    = analyze_sentiment(text, preprocessed)
    emotions     = detect_emotions(preprocessed)
    keywords     = extract_keywords(text, preprocessed)
    stats        = text_statistics(text, preprocessed)
    summary      = generate_summary(sentiment, emotions, stats, keywords)

    return {
        "sentiment": sentiment,
        "emotions": emotions,
        "keywords": keywords,
        "stats": stats,
        "summary": summary,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(value: float, lo: float, hi: float) -> float:
    """Normalise value from [lo, hi] to [0, 1]."""
    if hi == lo:
        return 0.5
    return (value - lo) / (hi - lo)


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)
