import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def vectorize_reviews(df: pd.DataFrame, text_column: str = "Review_Lemma") -> tuple:
    """
    Vectorizes text from a single preprocessed column using CountVectorizer.

    Parameters:
        df (pd.DataFrame): The input DataFrame with text data.
        text_column (str): The name of the column to vectorize.

    Returns:
        X: The sparse feature matrix
        vectorizer: The fitted CountVectorizer
    """
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2
    )
    X = vectorizer.fit_transform(df[text_column].fillna(""))

    return X, vectorizer
