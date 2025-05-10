import os

# Project root directory (you can change this to whatever your project root is)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "reviews_raw.csv")
DATA_INTERIM = os.path.join(BASE_DIR, "data", "interim")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

RAW_PATH = DATA_RAW
STEP1_FILTERED_COLUMNS = os.path.join(DATA_INTERIM, "reviews_filtered_columns.csv")
STEP2_TURKISH_ONLY = os.path.join(DATA_INTERIM, "reviews_tr_only.csv")
STEP3_CLEANED = os.path.join(DATA_INTERIM, "reviews_cleaned.csv")
STEP4_NORMALIZED = os.path.join(DATA_INTERIM, "reviews_normalized.csv")
STEP5_NO_STOPWORDS = os.path.join(DATA_INTERIM, "reviews_no_stopwords.csv")
STEP6_LEMMATIZED = os.path.join(DATA_INTERIM, "reviews_lemmatized.csv")
STEP7_LABELED = os.path.join(DATA_INTERIM, "reviews_labeled_sentiment.csv")
STEP8_MODEL_READY = os.path.join(DATA_PROCESSED, "reviews_model_ready.csv")

COLUMNS_TO_KEEP = ["HotelName", "Review", "Rating"] # columns to keep for step 1
COLUMNS_TO_KEEP_FOR_MODEL = ["Review_Lemma", "Sentiment"] # columns to keep for model
COLUMN_TO_DETECT_LANGUAGE = "Review" # column to keep only language specific reviews. step 2
LANGUAGE_TO_FILTER = "tr" # language to filter. step 2
