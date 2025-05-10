import re
import string
import nltk
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text: str) -> str:
    """
    Lowercases text, removes punctuation, numbers, and special characters.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def tokenize_and_filter(text: str, min_token_len=2) -> list[str]:
    """
    Tokenizes text, removes stopwords and short tokens.
    """
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered = [
        t for t in tokens
        if t not in stop_words and len(t) >= min_token_len
    ]
    return filtered

def preprocess_text(text: str) -> str:
    """
    Applies full preprocessing pipeline:
    1) Clean text
    2) Tokenize & remove stopwords/short tokens
    3) Rejoin for vectorization
    """
    cleaned = clean_text(text)
    tokens = tokenize_and_filter(cleaned)
    return " ".join(tokens)
