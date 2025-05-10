import os
import pandas as pd
from src.preprocessing.preprocessing import preprocess_text
from src.sentiment_annotation.VADER import vader_sentiment_label
from src.text_vectorization.text_vectorization_sklearn import vectorize_reviews
from src.models.Logistics_Regression_Model import train_logistic_regression


def run_model_one(data_path: str):
    """
    End-to-end pipeline for Model One:
    1. Load CSV
    2. Preprocess Negative/Positive Reviews
    3. (Optional) VADER annotation => Negative_Review_Sentiment, Positive_Review_Sentiment
    4. Vectorize
    5. Train Logistic Regression
    6. Print metrics
    """
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} rows from {data_path}")
    required_cols = ["Negative_Review", "Positive_Review"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["Negative_Review"] = df["Negative_Review"].fillna("").apply(preprocess_text)
    df["Positive_Review"] = df["Positive_Review"].fillna("").apply(preprocess_text)
    df["Negative_Review_Sentiment"] = df["Negative_Review"].apply(vader_sentiment_label)
    df["Positive_Review_Sentiment"] = df["Positive_Review"].apply(vader_sentiment_label)
    y = df["Negative_Review_Sentiment"]
    X, vectorizer = vectorize_reviews(df)
    model = train_logistic_regression(X, y)
    print("Model One pipeline complete.")


if __name__ == "__main__":
    # Adjust your path to the CSV file as needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "..", "data", "Booking.com", "Hotels_Reviews.csv")

    run_model_one(data_path)
