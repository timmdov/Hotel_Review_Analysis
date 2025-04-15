import logging

from dataset_operations import *
from dataset_operations.config import RAW_DATASET_PATH, FILTERED_PATH, TURKISH_ONLY_PATH, COLUMNS_TO_KEEP, \
    COLUMN_TO_DETECT_LANGUAGE

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


def clean_and_filter_turkish_reviews(original_path, filtered_columns_path, final_output_path, columns_to_keep,
                                     review_column="Review"):
    """
    Sequentially runs:
    1. `remove_unnecessary_columns()` to keep only specified columns
    2. `filter_turkish_reviews()` to keep only Turkish reviews
    """
    remove_unnecessary_columns(original_path, filtered_columns_path, columns_to_keep)
    filter_reviews_for_specific_language(filtered_columns_path, final_output_path, review_column)


def run_dataset_pipeline():
    clean_and_filter_turkish_reviews(
        original_path=RAW_DATASET_PATH,
        filtered_columns_path=FILTERED_PATH,
        final_output_path=TURKISH_ONLY_PATH,
        columns_to_keep=COLUMNS_TO_KEEP,
        review_column=COLUMN_TO_DETECT_LANGUAGE
    )


if __name__ == "__main__":
    run_dataset_pipeline()
