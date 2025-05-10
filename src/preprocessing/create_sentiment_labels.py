import pandas as pd

from src.utils.config import STEP6_LEMMATIZED, STEP7_LABELED
from src.utils.logger import get_logger

logger = get_logger(__name__)


def map_rating_to_sentiment(rating: int) -> int:
    if rating <= 3:  # Changed from rating <= 2 to include rating 3
        return 0  # Negative
    else:
        return 1  # Positive


def create_sentiment_labels(input_path: str, output_path: str) -> None:
    """
    Maps hotel review ratings (1–5) to sentiment labels:
    - 0: Negative (ratings 1-3)
    - 1: Positive (ratings 4-5)

    Adds a new column: 'Sentiment'

    Parameters:
    - input_path (str): Path to lemmatized review CSV
    - output_path (str): Path to save dataset with sentiment labels
    """
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with {len(df)} rows.")

        df["Sentiment"] = df["Rating"].apply(map_rating_to_sentiment)
        logger.info(f"Sentiment label distribution:\n{df['Sentiment'].value_counts()}")

        df.to_csv(output_path, index=False)
        logger.info(f"Sentiment-labeled dataset saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error during sentiment label creation: {e}")


if __name__ == "__main__":
    create_sentiment_labels(
        input_path=STEP6_LEMMATIZED,
        output_path=STEP7_LABELED
    )