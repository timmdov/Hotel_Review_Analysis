import pandas as pd
from trnlp import TrnlpWord

<<<<<<< Updated upstream:ML_SentimentAnalysis/preprocessing/lemmatize_reviews.py
# TODO: try other tool, too
from config import LEMMATIZED_PATH, NO_STOP_WORDS_PATH
from logger import get_logger
=======
from src.utils.config import LEMMATIZED_PATH, NO_STOP_WORDS_PATH
from src.utils.logger import get_logger
>>>>>>> Stashed changes:ML_SentimentAnalysis/src/preprocessing/lemmatize_reviews.py

logger = get_logger(__name__)


def lemmatize_text(text: str) -> str:
    words = text.split()
    lemmatized_words = []

    for word in words:
        lemma = TrnlpWord()
        lemma.setword(word)
        lemmatized_words.append(lemma.get_stem if lemma.get_stem else word)

    return " ".join(lemmatized_words)


def lemmatize_review_dataset(input_path: str, output_path: str) -> None:
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with {len(df)} rows.")

        df["Review_Lemma"] = df["Review_Clean"].astype(str).apply(lemmatize_text)
        df.to_csv(output_path, index=False)
        logger.info(f"Lemmatized reviews saved to: {output_path}")

    except Exception as e:
        logger.error(f"Lemmatization failed: {e}")


if __name__ == "__main__":
    lemmatize_review_dataset(
        input_path=NO_STOP_WORDS_PATH,
        output_path=LEMMATIZED_PATH
    )
