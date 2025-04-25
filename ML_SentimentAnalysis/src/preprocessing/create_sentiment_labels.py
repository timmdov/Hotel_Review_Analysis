import pandas as pd

from src.utils.config import LEMMATIZED_PATH, LABELED_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def map_rating_to_sentiment(rating: int) -> int:
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive


def create_sentiment_labels(input_path: str, output_path: str) -> None:
    """
    Maps hotel review ratings (1–5) to sentiment labels:
    - 0: Negative
    - 1: Neutral
    - 2: Positive

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
        input_path=LEMMATIZED_PATH,
        output_path=LABELED_PATH
    )
