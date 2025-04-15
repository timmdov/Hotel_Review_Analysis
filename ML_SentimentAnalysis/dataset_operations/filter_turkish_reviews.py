import logging

import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from config import TURKISH_ONLY_PATH, FILTERED_PATH
from dataset_operations.config import COLUMN_TO_DETECT_LANGUAGE, LANGUAGE_TO_FILTER


def filter_reviews_for_specific_language(input_path, output_path, column_to_detect_language=COLUMN_TO_DETECT_LANGUAGE,
                                         language=LANGUAGE_TO_FILTER):
    """
        Detects the language of reviews in the given column and filters only Turkish ('tr') reviews.
        Saves the result to a new CSV file.

        Parameters:
        - input_path (str): Path to the input CSV file.
        - output_path (str): Path to save the filtered Turkish-only reviews.
        - review_column (str): The column that contains review text.
        - language (str): The language of the review text.
        """

    def detect_language(text):
        try:
            return detect(str(text))
        except LangDetectException:
            return "unknown"

    try:
        df = pd.read_csv(input_path)
        df["language"] = df[column_to_detect_language].apply(detect_language)
        df_tr = df[df["language"] == language]
        if "language" in df_tr.columns:
            df_tr = df_tr.drop(columns=["language"])
        df_tr.to_csv(output_path, index=False)
        logging.info(f"Saved Turkish reviews to: {output_path}")
    except Exception as e:
        logging.info(f"Error: {e}")


def run_filter_turkish_reviews():
    filter_reviews_for_specific_language(
        input_path=FILTERED_PATH,
        output_path=TURKISH_ONLY_PATH,
        column_to_detect_language=COLUMN_TO_DETECT_LANGUAGE,
    )


if __name__ == "__main__":
    run_filter_turkish_reviews()
