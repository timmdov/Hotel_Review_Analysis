import re

import pandas as pd

from src.utils.config import STEP3_CLEANED, STEP4_NORMALIZED
from src.utils.logger import get_logger

logger = get_logger(__name__)

def normalize_text(text: str) -> str:
    """
    Normalizes a Turkish text string by:
    - Lowercasing
    - Removing digits and punctuation
    - Removing extra whitespace

    Parameters:
    - text (str): The input text string

    Returns:
    - str: Normalized clean text
    """
    text = text.lower()

    text = re.sub(r"[^a-zçğıöşü\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_review_dataset(input_path: str, output_path: str) -> None:
    """
    Normalizes the review text in a CSV file and saves the updated dataset.

    Parameters:
    - input_path (str): Path to the cleaned CSV file
    - output_path (str): Path to save the normalized version
    """
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with {len(df)} rows.")

        df["Review_Normalized"] = df["Review"].astype(str).apply(normalize_text)
        logger.info("Text normalization complete.")

        df.to_csv(output_path, index=False)
        logger.info(f"Saved normalized reviews to: {output_path}")

    except Exception as e:
        logger.error(f"Error during normalization: {e}")


if __name__ == "__main__":
    normalize_review_dataset(
        input_path=STEP3_CLEANED,
        output_path=STEP4_NORMALIZED
    )
