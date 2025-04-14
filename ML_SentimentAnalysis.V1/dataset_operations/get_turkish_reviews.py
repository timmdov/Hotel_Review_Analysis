import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Load your dataset
df = pd.read_csv("~/Projects/Pycharm/HotelReviewAnalysis/ML_SentimentAnalysis.V1/dataset/TripAdvisor Reviews Scraper.csv")

# Define a function to detect language
def detect_language(text):
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"

# Apply language detection
df["language"] = df["Review"].apply(detect_language)

# Filter only Turkish reviews
df_tr = df[df["language"] == "tr"]

# Save filtered dataset
df_tr.to_csv("~/Projects/Pycharm/HotelReviewAnalysis/ML_SentimentAnalysis.V1/dataset/turkish_reviews_only.csv", index=False)