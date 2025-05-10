from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import STEP6_LEMMATIZED, REVIEW_WORDCOUNT_PNG
from src.utils.logger import get_logger

logger = get_logger(__name__)


def basic_text_eda(input_path: str, text_column: str = "Review_Lemma"):
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with {len(df)} rows.")

        # Top 30 most common words
        all_words = []
        for text in df[text_column].dropna():
            all_words.extend(text.split())

        counter = Counter(all_words)
        most_common = counter.most_common(30)

        print("\nTop 30 Most Common Words:")
        for word, count in most_common:
            print(f"{word:<15} {count}")

        # Word count per review
        df["word_count"] = df[text_column].apply(lambda x: len(str(x).split()))

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df["word_count"], bins=30, color='skyblue', edgecolor='black')
        plt.title("Distribution of Review Word Counts")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(REVIEW_WORDCOUNT_PNG)
        plt.close()

        logger.info(f"EDA complete. Histogram saved to: {REVIEW_WORDCOUNT_PNG}")

    except Exception as e:
        logger.error(f"Error in EDA: {e}")


if __name__ == "__main__":
    basic_text_eda(
        input_path=STEP6_LEMMATIZED,
        text_column="Review_Lemma"
    )
