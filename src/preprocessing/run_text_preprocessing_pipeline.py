from clean_reviews import clean_review_dataset
from lemmatize import lemmatize_review_dataset
from prepare_final_dataset import prepare_dataset_for_model
from remove_stopwords import clean_stopwords_from_dataset
from src.preprocessing.label_sentiment import create_sentiment_labels
from src.utils.config import *
from src.utils.remove_columns import remove_unnecessary_columns
from src.utils.filter_language import filter_reviews_for_language
from normalize_text import normalize_review_dataset


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
        input_path=RAW_PATH,
        output_path=STEP1_FILTERED_COLUMNS,
        columns_to_keep=COLUMNS_TO_KEEP
    )

    # Step 2
    filter_reviews_for_language(
        input_path=STEP1_FILTERED_COLUMNS,
        output_path=STEP2_TURKISH_ONLY,
        column_to_detect_language=COLUMN_TO_DETECT_LANGUAGE,
        language=LANGUAGE_TO_FILTER
    )

    # Step 3
    clean_review_dataset(
        input_path=STEP2_TURKISH_ONLY,
        output_path=STEP3_CLEANED
    )

    # Step 4
    normalize_review_dataset(
        input_path=STEP3_CLEANED,
        output_path=STEP4_NORMALIZED
    )

    # Step 5
    clean_stopwords_from_dataset(
        input_path=STEP4_NORMALIZED,
        output_path=STEP5_NO_STOPWORDS
    )

    # Step 6
    lemmatize_review_dataset(
        input_path=STEP5_NO_STOPWORDS,
        output_path=STEP6_LEMMATIZED
    )

    # Step 7
    create_sentiment_labels(
        input_path=STEP6_LEMMATIZED,
        output_path=STEP7_LABELED
    )

    # Step 8
    prepare_dataset_for_model(
        input_path=STEP7_LABELED,
        output_path=STEP8_MODEL_READY,
    )


if __name__ == "__main__":
    run_text_preprocessing_pipeline()
