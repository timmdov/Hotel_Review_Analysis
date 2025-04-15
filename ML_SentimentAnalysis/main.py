import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

from dataset_operations.run_dataset_pipeline import clean_and_filter_turkish_reviews
from logger import get_logger
from text_vectorization.text_vectorization_sklearn import vectorize_reviews

logger = get_logger(__name__)


def train_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        multi_class='auto'
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='macro')
    cm = confusion_matrix(y_test, predictions)

    logger.info("Evaluation Results:")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-score:  {f1:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(cm)

    return model


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Run dataset pipeline (filter columns + Turkish-only)
    clean_and_filter_turkish_reviews(
        original_path=os.path.join(current_dir, "dataset", "tripadvisor_raw_reviews.csv"),
        filtered_columns_path=os.path.join(current_dir, "dataset", "filtered_columns.csv"),
        final_output_path=os.path.join(current_dir, "dataset", "turkish_reviews_only.csv"),
        columns_to_keep=["HotelName", "Review", "Rating"],
        review_column="Review"
    )

    # Step 2: Load cleaned data
    df = pd.read_csv(os.path.join(current_dir, "dataset", "turkish_reviews_only.csv"))
    X, vectorizer = vectorize_reviews(df)
    y = df["Rating"]  # or another column like Sentiment if you generate one
    model = train_logistic_regression(X, y)


if __name__ == "__main__":
    main()
