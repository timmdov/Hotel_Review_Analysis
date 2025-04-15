RAW_DATASET_PATH = "../dataset/tripadvisor_raw_reviews.csv" # raw dataset
FILTERED_PATH = "../dataset/tripadvisor_filtered_columns.csv" # step 1: removing unnecessary columns
TURKISH_ONLY_PATH = "../dataset/tripadvisor_tr_reviews.csv" # step 2: keeping only language specific reviews
CLEANED_PATH = "../dataset/tripadvisor_cleaned_reviews.csv" # step 3: cleaning data
NORMALIZED_PATH = "../dataset/tripadvisor_normalized_reviews.csv" # step 4: normalizing data
NO_STOP_WORDS_PATH = "../dataset/tripadvisor_cleaned_nostopwords.csv" # step 5: removing stop words
LEMMATIZED_PATH = "../dataset/tripadvisor_lemmatized_reviews.csv" # step 6: lemmatization

COLUMNS_TO_KEEP = ["HotelName", "Review", "Rating"] # columns to keep for step 1
COLUMN_TO_DETECT_LANGUAGE = "Review" # column to keep only language specific reviews. step 2
LANGUAGE_TO_FILTER = "tr" # language to filter. step 2
