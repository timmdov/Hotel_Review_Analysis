import pandas as pd
from nltk.corpus import stopwords

from config import NORMALIZED_PATH, NO_STOP_WORDS_PATH
from logger import get_logger

logger = get_logger(__name__)

import nltk

nltk.download("stopwords")

turkish_stopwords = set(stopwords.words("turkish"))

domain_stopwords = [
    "otel", "hotelde", "otelin", "otelde", "otelden",
    "konakladık", "konaklama", "tesis", "çalışan", "çalışanlar",
    "personel", "personelin", "personeller", "çok", "gayet", "herşey",
    "her şey", "otelimiz", "oteli", "otelinde", "otelimizi",
    "kaldık", "kaldığımız", "memnun", "memnunuz", "memnuniyet",
    "güzeldi", "iyiydi", "süper", "mükemmel", "harika", "güzel",
    "yer", "mekan", "mekân", "yeriydi", "yeri", "bir", "daha",
    "yine", "için", "bize", "bizi", "bizim", "bizi", "çok", "şey",
    # Added after 1st EDA process
    "teşekkür", "bey", "hanım", "eder", "et", "ol", "var", "olarak",
    "kadar", "ayrı", "gül", "tekrar", "ekip","olma", "gel"
]

custom_stopwords = turkish_stopwords.union(domain_stopwords)

def remove_stopwords(text: str) -> str:
    """
    Removes Turkish stopwords from normalized review text.

    Parameters:
    - text (str): Normalized review string

    Returns:
    - str: Cleaned text without stopwords
    """
    words = text.split()
    filtered = [word for word in words if word not in custom_stopwords]
    return " ".join(filtered)


def clean_stopwords_from_dataset(input_path: str, output_path: str) -> None:
    """
    Applies stopword removal to the 'Review_Normalized' column and saves result.

    Parameters:
    - input_path (str): Path to the normalized dataset
    - output_path (str): Path to save stopword-free dataset
    """
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with {len(df)} rows.")

        if "Review_Normalized" not in df.columns:
            raise ValueError("Missing 'Review_Normalized' column in input CSV.")

        df["Review_Clean"] = df["Review_Normalized"].astype(str).apply(remove_stopwords)

        df.to_csv(output_path, index=False)
        logger.info(f"Stopwords removed. Saved cleaned reviews to: {output_path}")

    except Exception as e:
        logger.error(f"Error during stopword removal: {e}")


if __name__ == "__main__":
    clean_stopwords_from_dataset(
        input_path=NORMALIZED_PATH,
        output_path=NO_STOP_WORDS_PATH
    )
