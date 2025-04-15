from clean_reviews import clean_review_dataset
from config import *
from dataset_operations import remove_unnecessary_columns
from dataset_operations.filter_language import filter_reviews_for_specific_language
from lemmatize_reviews import lemmatize_review_dataset
from preprocessing.create_sentiment_labels import create_sentiment_labels
from remove_stopwords import clean_stopwords_from_dataset
from text_normalization import normalize_review_dataset


def run_text_preprocessing_pipeline():
    """
    Runs the full text preprocessing pipeline on hotel review dataset:
    1. Remove unnecessary columns
    2. Keep only Turkish reviews
    3. Remove nulls, duplicates, empty data
    4. Normalize text
    5. Remove stopwords (including domain-specific)
    6. Lemmatize text
    7. Label rating
    """
    # Step 1
    remove_unnecessary_columns(
        input_path=RAW_DATASET_PATH,
        output_path=FILTERED_PATH,
        columns_to_keep=COLUMNS_TO_KEEP
    )

    # Step 2
    filter_reviews_for_specific_language(
        input_path=FILTERED_PATH,
        output_path=TURKISH_ONLY_PATH,
        column_to_detect_language=COLUMN_TO_DETECT_LANGUAGE,
        language=LANGUAGE_TO_FILTER
    )

    # Step 3
    clean_review_dataset(
        input_path=TURKISH_ONLY_PATH,
        output_path=CLEANED_PATH
    )

    # Step 4
    normalize_review_dataset(
        input_path=CLEANED_PATH,
        output_path=NORMALIZED_PATH
    )

    # Step 5
    clean_stopwords_from_dataset(
        input_path=NORMALIZED_PATH,
        output_path=NO_STOP_WORDS_PATH
    )

    # Step 6
    lemmatize_review_dataset(
        input_path=NO_STOP_WORDS_PATH,
        output_path=LEMMATIZED_PATH
    )

    # Step 7
    create_sentiment_labels(
        input_path=LEMMATIZED_PATH,
        output_path=LABELED_PATH
    )


if __name__ == "__main__":
    run_text_preprocessing_pipeline()
