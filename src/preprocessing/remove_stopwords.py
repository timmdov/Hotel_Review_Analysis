"""
Stopword Removal Module for Turkish Hotel Reviews (after Lemmatization)

This module removes stopwords from lemmatized Turkish hotel reviews.
- It uses Turkish NLTK stopwords (excluding sentiment-carrying words)
- Adds extra fillers and domain-specific vocabulary
"""
import os

import pandas as pd

from src.utils.config.paths import STEP5_LEMMATIZED, STEP6_NO_STOPWORDS
from src.utils.io.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

with open(os.path.join(BASE_DIR, "resources", "stopwords.txt"), "r", encoding="utf-8") as f:
    stopwords = set(line.strip() for line in f if line.strip())


def remove_stopwords(text: str) -> str:
    words = text.split()
    filtered = [word for word in words if word not in stopwords]
    return " ".join(filtered)


def clean_stopwords_from_dataset(input_path: str, output_path: str) -> None:
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with {len(df)} rows.")

        if "Review_Lemma" not in df.columns:
            raise ValueError("Missing 'Review_Lemma' column.")

        df["Review_Clean"] = df["Review_Lemma"].astype(str).apply(remove_stopwords)

        df.to_csv(output_path, index=False)
        logger.info(f"Stopwords removed. Saved to: {output_path}")

    except Exception as e:
        logger.error(f"Stopword cleaning failed: {e}")


if __name__ == "__main__":
    clean_stopwords_from_dataset(
        input_path=STEP5_LEMMATIZED,
        output_path=STEP6_NO_STOPWORDS
    )
