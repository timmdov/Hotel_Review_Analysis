import pandas as pd

from src.utils.config import TURKISH_ONLY_PATH, CLEANED_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_review_dataset(input_path: str, output_path: str) -> None:
    """
    Cleans a hotel review dataset by removing duplicates and rows with missing or empty reviews/ratings.

    Parameters:
    - input_path (str): Path to the raw Turkish review CSV file
    - output_path (str): Path where the cleaned CSV will be saved
    """
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with {len(df)} rows.")

        df = df.drop_duplicates()
        logger.info(f"Removed duplicates. Rows left: {len(df)}")

        df = df.dropna(subset=["Review", "Rating"])
        logger.info(f"Removed rows with missing Review/Rating. Rows left: {len(df)}")

        df["Review"] = df["Review"].astype(str)
        df = df[df["Review"].str.strip() != ""]
        logger.info(f"Removed empty-string reviews. Rows left: {len(df)}")

        df = df.reset_index(drop=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned dataset saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error while cleaning dataset: {e}")


if __name__ == "__main__":
    clean_review_dataset(
        input_path=TURKISH_ONLY_PATH,
        output_path=CLEANED_PATH
    )
