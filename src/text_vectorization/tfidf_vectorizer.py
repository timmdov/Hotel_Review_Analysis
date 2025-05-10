import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.io.logger import get_logger
from src.utils.config.paths import STEP5_NO_STOPWORDS

logger = get_logger(__name__)

def vectorize_reviews(df: pd.DataFrame, text_column: str = "Review_Clean"):
    """
    Applies TF-IDF vectorization to the specified text column.

    Parameters:
    - df (pd.DataFrame): DataFrame with cleaned text column
    - text_column (str): Column name with text to vectorize

    Returns:
    - X: TF-IDF sparse matrix
    - vectorizer: The fitted TfidfVectorizer instance
    """
    try:
        logger.info("Starting TF-IDF vectorization...")

        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigrams + bigrams
            min_df=5,
            max_df=0.85,
        )
        X = tfidf.fit_transform(df[text_column])
        logger.info(f"TF-IDF complete. Matrix shape: {X.shape}")
        return X, tfidf

    except Exception as e:
        logger.error(f"Error during vectorization: {e}")
        return None, None


if __name__ == "__main__":
    df = pd.read_csv(STEP5_NO_STOPWORDS)
    X, tfidf_vectorizer = vectorize_reviews(df)