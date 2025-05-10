import pandas as pd
from trnlp import TrnlpWord

from src.utils.config.paths import STEP4_NORMALIZED, STEP5_LEMMATIZED
from src.utils.io.logger import get_logger

logger = get_logger(__name__)


def lemmatize_text(text: str) -> str:
    words = text.split()
    lemmatized_words = []

    for word in words:
        lemma = TrnlpWord()
        lemma.setword(word)
        # Fix: get_stem() is a method
        stem = lemma.get_stem
        lemmatized_words.append(stem if stem else word)

    return " ".join(lemmatized_words)


def lemmatize_review_dataset(input_path: str, output_path: str) -> None:
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with {len(df)} rows.")

        if "Review_Normalized" not in df.columns:
            raise ValueError("Missing 'Review_Normalized' column in input CSV.")

        df["Review_Lemma"] = df["Review_Normalized"].astype(str).apply(lemmatize_text)

        df.to_csv(output_path, index=False)
        logger.info(f"Lemmatized reviews saved to: {output_path}")

    except Exception as e:
        logger.error(f"Lemmatization failed: {e}")


if __name__ == "__main__":
    lemmatize_review_dataset(
        input_path=STEP4_NORMALIZED,
        output_path=STEP5_LEMMATIZED
    )