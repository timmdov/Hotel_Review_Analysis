"""
Stopword Removal Module for Turkish Hotel Reviews

This module provides functions for removing stopwords from Turkish hotel reviews.
The stopwords list consists of:
1. Standard Turkish stopwords from NLTK (with sentiment-bearing words removed)
2. Additional common Turkish stopwords not included in NLTK
3. Domain-specific stopwords related to hotels and travel

The stopwords list has been carefully curated to:
- Remove common words that don't contribute to sentiment analysis
- Preserve words that are important for sentiment classification
- Remove domain-specific terms that are common across all reviews
"""

import pandas as pd
from nltk.corpus import stopwords

from src.utils.config import STEP4_NORMALIZED, STEP5_NO_STOPWORDS
from src.utils.logger import get_logger

logger = get_logger(__name__)

import nltk

# Download NLTK stopwords if not already downloaded
nltk.download("stopwords", quiet=True)

turkish_stopwords = set(stopwords.words("turkish"))

# Remove some stopwords that might be important for sentiment analysis
sentiment_bearing_words = {
    "güzel", "iyi", "kötü", "harika", "berbat", "mükemmel", 
    "temiz", "kirli", "rahat", "rahatsız", "lezzetli", "lezzetsiz",
    "nazik", "kaba", "yardımsever", "ilgisiz", "pahalı", "ucuz"
}

turkish_stopwords = turkish_stopwords - sentiment_bearing_words

# Add more common Turkish stopwords that aren't in NLTK's list
additional_turkish_stopwords = {
    "acaba", "aslında", "belki", "beri", "bile", "böyle", "çok", "çünkü",
    "diye", "eğer", "fakat", "gene", "gibi", "hatta", "hem", "hep",
    "hepsi", "hiç", "ise", "işte", "kaç", "kez", "ki", "kim",
    "madem", "nasıl", "neden", "nerede", "nereye", "niçin", "niye", "rağmen",
    "sanki", "şayet", "şekilde", "şimdi", "tüm", "üzere", "ya", "yani",
    "yok", "zaten", "zira"
}

turkish_stopwords = turkish_stopwords.union(additional_turkish_stopwords)

domain_stopwords = [
    # Hotel and accommodation related terms
    "otel", "hotelde", "otelin", "otelde", "otelden", "hotel",
    "konakladık", "konaklama", "tesis", "tesisi", "tesiste",
    "oda", "odada", "odalar", "odaları", "odamız", "odası",
    "resepsiyon", "lobi", "lobby", "restoran", "restaurant",
    "kahvaltı", "yemek", "havuz", "plaj", "deniz", "spa", "hamam",
    "rezervasyon", "booking", "check", "giriş", "çıkış",

    # Staff related terms
    "çalışan", "çalışanlar", "personel", "personelin", "personeller", 
    "görevli", "görevliler", "resepsiyonist", "garson", "temizlikçi",

    # General hotel review terms
    "herşey", "her şey", "otelimiz", "oteli", "otelinde", "otelimizi",
    "kaldık", "kaldığımız", "konakladığımız", "yer", "mekan", "mekân", 
    "yeriydi", "yeri", "gece", "gün", "hafta", "tatil", "seyahat",

    # Common Turkish filler words
    "bir", "daha", "yine", "için", "bize", "bizi", "bizim", "şey",
    "bey", "hanım", "eder", "et", "ol", "var", "olarak",
    "kadar", "ayrı", "gül", "tekrar", "ekip", "olma", "gel",

    # Additional common words in hotel reviews
    "kere", "kez", "defa", "kişi", "kişilik", "aile", "arkadaş",
    "genel", "özel", "hizmet", "servis", "fiyat", "ücret", "para",
    "süre", "zaman", "saat", "dakika", "gün", "gece", "sabah", "akşam"
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
        input_path=STEP4_NORMALIZED,
        output_path=STEP5_NO_STOPWORDS
    )