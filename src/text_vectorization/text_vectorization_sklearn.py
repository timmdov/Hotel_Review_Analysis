import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def vectorize_reviews(df: pd.DataFrame) -> tuple:
    """
    Vectorizes text from the DataFrame using CountVectorizer with:
      - unigrams and bigrams (ngram_range=(1,2))
      - max_features=10000
      - min_df=2  (terms must appear in >=2 documents)

    Returns:
      X: The sparse feature matrix
      vectorizer: The fitted CountVectorizer
    """
    df["combined_reviews"] = (
            df["Negative_Review"].fillna("") + " " + df["Positive_Review"].fillna("")
    )
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2
    )
    X = vectorizer.fit_transform(df["combined_reviews"])

    return X, vectorizer
